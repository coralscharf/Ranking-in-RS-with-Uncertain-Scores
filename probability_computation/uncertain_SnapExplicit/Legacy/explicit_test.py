import torch
import numpy as np

from uncertain.models import FunkSVD, CPMF, OrdRec, Ensemble
from uncertain.models import UncertainWrapper, ItemPopularity, UserProfileLength, ItemVariance
from uncertain.datasets.movielens import get_movielens_dataset
from evaluation import evaluate_ratings, evaluate_recommendations

ML = get_movielens_dataset(variant='100K')
train, test = ML.split(test_percentage=0.1, min_profile_length=3, seed=0)
train, val = ML.split(test_percentage=0.1, min_profile_length=0, seed=0)

item_popularity = ItemPopularity(train)
item_popularity.evaluation = {}
evaluate_recommendations(item_popularity, test, train)
print(item_popularity.evaluation)

baseline = FunkSVD(train, embedding_dim=50, weight_decay=0, lr=1, batch_size=512)
baseline.initialize(train)
# baseline.load()
baseline.fit(train, val)
baseline.evaluation = {}
evaluate_ratings(baseline, test)
evaluate_recommendations(baseline, test, train)
print(baseline.evaluation)
'''
cpmf = CPMF(embedding_dim=50, l2_penalty=0, initial_lr=5, batch_size=512,
            tolerance=2, max_epochs=100, path='test_models/cpmf.pth')
cpmf.initialize(train)
# cpmf.load()
cpmf.fit(train, val)
cpmf.evaluation = {}
evaluate_ratings(cpmf, test)
evaluate_recommendations(cpmf, test, train)
print(cpmf.evaluation)

ensemble = Ensemble(base_model=baseline, n_models=5)
ensemble.fit(train, val)
ensemble.evaluation = {}
evaluate_ratings(ensemble, test)
evaluate_recommendations(ensemble, test, train)
print(ensemble.evaluation)

resample = Resample(base_model=baseline, n_models=5)
resample.fit(train, val)
resample.evaluation = {}
evaluate_ratings(resample, test)
evaluate_recommendations(resample, test, train)
print(resample.evaluation)

user_support = UncertainWrapper(baseline, UserProfileLength(train))
user_support.evaluation = {}
evaluate_ratings(user_support, test)
evaluate_recommendations(user_support, test, train)
print(user_support.evaluation)

item_support = UncertainWrapper(baseline, ItemPopularity(train))
item_support.evaluation = {}
evaluate_ratings(item_support, test)
evaluate_recommendations(item_support, test, train)
print(item_support.evaluation)

item_variance = UncertainWrapper(baseline, ItemVariance(train))
item_variance.evaluation = {}
evaluate_ratings(item_variance, test)
evaluate_recommendations(item_variance, test, train)
print(item_variance.evaluation)

uncertainty = FunkSVD(embedding_dim=50, l2_penalty=0, initial_lr=10, batch_size=512,
                      tolerance=1, max_epochs=100, path='test_models/baseline.pth')
funksvdcv = CVUncertainty(recommender=baseline, uncertainty=uncertainty)
funksvdcv.fit(train, val)
funksvdcv = UncertainWrapper(baseline, funksvdcv.uncertainty)
funksvdcv.evaluation = {}
evaluate_ratings(funksvdcv, test)
evaluate_recommendations(funksvdcv, test, train)
print(funksvdcv.evaluation)

uncertainty = FunkSVD(embedding_dim=50, l2_penalty=0, initial_lr=10, batch_size=512,
                      tolerance=1, max_epochs=100, path='test_models/baseline.pth')
biascv = CVUncertainty(recommender=baseline, uncertainty=uncertainty)
biascv.fit(train, val)
biascv = UncertainWrapper(baseline, biascv.uncertainty)
biascv.evaluation = {}
evaluate_ratings(biascv, test)
evaluate_recommendations(biascv, train, test)
print(biascv.evaluation)

train.score_labels, train.scores = torch.unique(train.scores, return_inverse=True)
val.scores = torch.unique(val.scores, return_inverse=True)[1]
ordrec = OrdRec(embedding_dim=50, l2_penalty=0, initial_lr=10,
                batch_size=512, tolerance=1, max_epochs=100, path='test_models/ordrec.pth')
ordrec.initialize(train)
# ordrec.load()
ordrec.fit(train, val)
ordrec.evaluation = {}
evaluate_ratings(ordrec, test)
evaluate_recommendations(ordrec, test, train)
print(ordrec.evaluation)


from pandas import DataFrame as df
from matplotlib import pyplot as plt
from copy import deepcopy
from uncertain.data_structures import Recommendations
from uncertain.metrics import precision, hit_rate, surprise

unc = []
for u in range(test.num_users):
  users = torch.full([10], u).cuda()
  items = torch.randint(low=1, high=train.num_items, size=[10]).cuda()
  with torch.no_grad():
    unc.append(cpmf._net(users, items)[1])
unc = torch.hstack(unc).cpu().detach().numpy()
cuts = np.quantile(unc, [0.75, 0.5, 0.25])
unc = unc[unc < 5]
count, bins, _ = plt.hist(unc, 30, density=True)
count = [0] + list(count) + [0]
bins = list(bins) + [bins[-1] + (bins[-1] - bins[-2])]

cuts = {'Permissive': {'cut': cuts[0], 'color': 'g'},
        'Median': {'cut': cuts[1], 'color': 'b'},
        'Restrictive': {'cut': cuts[2], 'color': 'r'}}
f, ax = plt.subplots()
ax.plot(bins, count, color='k')
for key, value in cuts.items():
    ax.axvline(x=value['cut'], color=value['color'], linestyle='dashed', label=key)
ax.set_xlabel('Uncertainty')
ax.set_ylabel('Density')
ax.legend()
f.tight_layout()
#f.savefig('Movielens/distribution.pdf')
#f.show()

for quartile in cuts:
    cuts[quartile].update({'Drop': {'Coverage': 0, 'Hit rate': [], 'Surprise': []},
                           'Split certain': {'% Recommendations': 0, 'Hit rate': [], 'Surprise': []},
                           'Split uncertain': {'% Recommendations': 0, 'Hit rate': [], 'Surprise': []}})

baseline = {'cut': float('inf'), 'color': 'k', 'Hit rate': [], 'Surprise': []}

relevant_users = 0
for u in range(test.num_users):

    targets = test.items()[torch.logical_and(test.users() == u,
                                             test.ratings >= 4)]
    if not len(targets):
        continue
    else:
        relevant_users += 1

    rated = train.get_rated_items(u)
    with torch.no_grad():
        rated_var = cpmf._net.item_embeddings(rated)

    pred, unc = cpmf.predict_user(u)
    pred[rated] = -float('inf')
    ranking = pred.argsort(descending=True)[:-len(rated)]
    unc = unc[ranking]
    top10 = ranking[:10]
    unc_top10 = unc[top10]
    
    recommendations = Recommendations(u, top10, unc_top10)
    baseline['Hit rate'].append(hit_rate(recommendations, targets))
    baseline['Surprise'].append(surprise(recommendations, cpmf, rated_var))

    for quartile in cuts:

        idx = unc < cuts[quartile]['cut']
        ranking_ = ranking[idx]
        unc_ = unc[idx]
        if idx.sum() > 0:
            recommendations = Recommendations(u, ranking_[:10], unc_[:10])
            cuts[quartile]['Drop']['Hit rate'].append(hit_rate(recommendations, targets))
            cuts[quartile]['Drop']['Surprise'].append(surprise(recommendations, cpmf, rated_var))
        if idx.sum() >= 10:
            cuts[quartile]['Drop']['Coverage'] += 1

        if quartile != 'Free':
            is_certain = unc_top10 < cuts[quartile]['cut']
            if is_certain.sum() > 0:
                cuts[quartile]['Split certain']['% Recommendations'] += is_certain.sum()
                recommendations = Recommendations(u, top10[is_certain], unc_top10[is_certain])
                cuts[quartile]['Split certain']['Hit rate'].append(hit_rate(recommendations, targets))
                cuts[quartile]['Split certain']['Surprise'].append(surprise(recommendations, cpmf, rated_var))

            is_uncertain = torch.logical_not(is_certain)
            if is_uncertain.sum() > 0:
                cuts[quartile]['Split uncertain']['% Recommendations'] += is_uncertain.sum()
                recommendations = Recommendations(u, top10[is_uncertain], unc_top10[is_uncertain])
                cuts[quartile]['Split uncertain']['Hit rate'].append(hit_rate(recommendations, targets))
                cuts[quartile]['Split uncertain']['Surprise'].append(surprise(recommendations, cpmf, rated_var))

baseline['Hit rate'] = np.mean(baseline['Hit rate'])
baseline['Surprise'] = np.mean(np.hstack(baseline['Surprise']))
print(baseline)

dropping = df(([cuts[quartile]['Drop']['Coverage'] * 100 / relevant_users for quartile in cuts],
               [np.mean(cuts[quartile]['Drop']['Hit rate']) for quartile in cuts],
               [np.mean(np.hstack(cuts[quartile]['Drop']['Surprise'])) for quartile in cuts]),
              index=['Coverage', 'Hit rate', 'Surprise'], columns=cuts.keys()).T
print(dropping)

certain = df(([cuts[quartile]['Split certain']['% Recommendations'].item() * 10 / relevant_users for quartile in cuts],
              [np.mean(cuts[quartile]['Split certain']['Hit rate']) for quartile in cuts],
              [np.mean(np.hstack(cuts[quartile]['Split certain']['Surprise'])) for quartile in cuts]),
             index=['% Recommendations', 'Hit rate', 'Surprise'], columns=cuts.keys()).T
print(certain)

uncertain = df(([cuts[quartile]['Split uncertain']['% Recommendations'].item() * 10 / relevant_users for quartile in cuts],
                [np.mean(cuts[quartile]['Split uncertain']['Hit rate']) for quartile in cuts],
                [np.mean(np.hstack(cuts[quartile]['Split uncertain']['Surprise'])) for quartile in cuts]),
               index=['% Recommendations', 'Hit rate', 'Surprise'], columns=cuts.keys()).T
print(uncertain)
'''
