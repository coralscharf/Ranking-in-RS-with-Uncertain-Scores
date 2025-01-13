import torch
import numpy as np
from pandas import factorize
from pandas import DataFrame as df
from matplotlib import pyplot as plt

from uncertain.models import Linear, FunkSVD, CPMF, OrdRec, Ensemble, Resample
from uncertain.models import LinearUncertainty, CVUncertainty, PlugIn
from uncertain.datasets.movielens import get_movielens_dataset
from uncertain.cross_validation import random_train_test_split
from uncertain.metrics import recommendation_score, rmse_score, correlation, rpi_score, classification, diversity
from uncertain.utils import minibatch

ML = get_movielens_dataset(variant='100K').cuda()
train, test = random_train_test_split(ML, test_percentage=0.2, random_state=0)
test, val = random_train_test_split(test, test_percentage=0.5, random_state=0)

CPMF_params = {'embedding_dim': 50, 'l2': 0, 'learning_rate': 4,
               'batch_size': 512, 'path': 'tests/test_models/cpmf.pth', 'use_cuda': True}
cpmf = CPMF(**CPMF_params)
cpmf.initialize(train)
cpmf.load()

unc = []
for u in range(test.num_users):
    unc.append(cpmf.predict(user_id=1)[1])
unc = torch.hstack(unc).cpu().detach().numpy()
unc = unc[unc < 4]
count, bins, _ = plt.hist(unc, 20, density=True)
count = [0] + list(count) + [0]
bins = list(bins) + [bins[-1] + (bins[-1] - bins[-2])]
f, ax = plt.subplots()
ax.plot(bins, count, color='k')
ax.axvline(x=2, color='g', linestyle='dashed', label='Permissive cutoff')
ax.axvline(x=1, color='r', linestyle='dashed', label='Restrictive cutoff')
ax.set_xlabel('Uncertainty')
ax.set_ylabel('Density')
ax.legend()
f.tight_layout()
f.savefig('tests/sample_dist.pdf')

cuts = {'Free': float('inf'), 'Permissive': 2.0, 'Restrictive': 1.0}
precision = {'Free': [], 'Permissive': [], 'Restrictive': []}
surprise = {'Free': [], 'Permissive': [], 'Restrictive': []}
for u in range(test.num_users):

    targets = test.items()[torch.logical_and(test.users() == u,
                                             test.ratings >= 4)]
    if not len(targets):
        continue

    rated = train.items()[train.users() == u]
    with torch.no_grad():
        rated_var = cpmf._net.item_embeddings(rated)

    recommendation, uncertainties = cpmf.recommend(u, train, top=test.num_items)

    for cut in cuts:
        idx = uncertainties < cuts[cut]
        recom, unc = recommendation[idx][:10], uncertainties[idx][:10]
        hits = np.zeros(10)
        for k in range(len(recom)):
            if recom[k] in targets:
                hits[k] = 1
            with torch.no_grad():
                recom_var = cpmf._net.item_embeddings(recom[k])
                surprise[cut].append((1. - torch.cosine_similarity(recom_var, rated_var, dim=-1)).min().item())
        precision[cut].append(hits.cumsum(0) / np.arange(1, 11))


from matplotlib import pyplot as plt
f, ax = plt.subplots()
ax.plot(range(1, 11), np.vstack(precision['Free']).mean(0), label='No cut')
ax.plot(range(1, 11), np.vstack(precision['Permissive']).mean(0), color='g', label='Permissive cut')
ax.plot(range(1, 11), np.vstack(precision['Restrictive']).mean(0), color='r', label='Restrictive cut')
ax.legend()
f.tight_layout()
f.savefig('tests/cuts.pdf')

for cut in cuts:
    print(np.mean(surprise[cut]))