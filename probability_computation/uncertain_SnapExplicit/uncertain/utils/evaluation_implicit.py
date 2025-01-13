import torch
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from scipy.special import comb
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from uncertain.core import VanillaRecommender, UncertainRecommender
    

def get_AP(hits):
    n_hits = hits.cumsum(0)
    if n_hits[-1] > 0:
        precision = n_hits / np.arange(1, len(hits) + 1)
        return np.sum(precision * hits) / n_hits[-1]
    else:
        return 0
    

def val_MAP(model, data):
    AP = np.zeros(data.n_user)
    for user in tqdm(range(data.n_user)):
        targets = data.val[data.val[:, 0] == user, 1]
        rated = data.train[:, 1][data.train[:, 0] == user]
        rec =  model.recommend(user, remove_items=rated, n=5)
        AP[user] = get_AP(rec.index.isin(targets))
    return np.mean(AP)


def accuracy_metrics(size, max_k, is_uncertain):
    out = {'MAP': np.zeros((size, max_k)),
           'Recall': np.zeros((size, max_k))}
    if is_uncertain:
        out['URI'] = np.full((size, max_k), np.nan)
    return out


def test(model, data, max_k, name):
    
    metrics = {}
    Rating_rec = accuracy_metrics(len(data.test_users), max_k, isinstance(model, UncertainRecommender))
    
    pred = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test[:, 1]).long())
    neg = model.predict(torch.tensor(data.test[:, 0]).long(), torch.tensor(data.test_negative_items).long())
    
    if isinstance(model, UncertainRecommender):
        pred, unc = pred[0], pred[1]
        is_concordant = pred - neg[0] > 0
        metrics['FCP'] = is_concordant.sum().item() / len(data.test)
        metrics['Concordant uncertainty'] = unc[is_concordant].mean()
        metrics['Discordant uncertainty'] = unc[~is_concordant].mean()
        
        URI = {'hits': 0, 'hits_unc': 0, 'avg': np.zeros(len(data.test_users))}
        
        rand_preds = model.predict(data.rand['users'], data.rand['items'])
        metrics['corr_usup'] = stats.spearmanr(rand_preds[1], data.user_support[data.rand['users']].flatten())[0]
        # metrics['corr_udiv'] = stats.spearmanr(rand_preds[1], data.user_diversity[data.rand['users']].flatten())[0]
        metrics['corr_isup'] = stats.spearmanr(rand_preds[1], data.item_support[data.rand['items']].flatten())[0]
        
        '''
        metrics['unc_plot'], ax = plt.subplots(ncols=2)
        ax[0].hist(rand_preds[1])
        ax[0].set_xlabel('Uncertainty')
        ax[0].set_ylabel('Density')
        ax[1].plot(rand_preds[0], rand_preds[1], 'o')
        ax[1].set_xlabel('Score')
        ax[1].set_ylabel('Uncertainty')
        metrics['unc_plot'].tight_layout()
        '''
        
        Cuts = {'Values': np.nanquantile(rand_preds[1], np.linspace(0.8, 0.2, 4)),
                'Coverage': np.ones((len(data.test_users), 5)),
                'Relevance': np.zeros((len(data.test_users), 5)),
                'MAP': np.zeros((len(data.test_users), 5))}
        
    else:
        is_concordant = pred - neg > 0
        metrics['FCP'] = is_concordant.sum().item() / len(data.test)
    
    if hasattr(model, 'uncertain_predict_user'):
        Uncertain_rec = accuracy_metrics(len(data.test_users), max_k, True)
        
    precision_denom = np.arange(1, max_k + 1)
        
    for idxu, user in enumerate(tqdm(data.test_users, desc=name+' - Recommending')):
        targets = data.test[data.test[:, 0] == user, 1]
        rated = data.train[:, 1][data.train[:, 0] == user]
        rec = model.recommend(user, remove_items=rated, n=data.n_item - len(rated))
        top_k = rec[:max_k]
        hits = top_k.index.isin(targets)
        n_hits = hits.cumsum(0)

        if n_hits[-1] > 0:
            precision = n_hits / precision_denom
            Rating_rec['MAP'][idxu] = np.cumsum(precision * hits) / np.maximum(1, n_hits)
            Rating_rec['Recall'][idxu] = n_hits / len(targets)

        if isinstance(model, UncertainRecommender):
            unc = rec.uncertainties.to_numpy()
            top_k_unc = unc[:max_k]
            URI['avg'][idxu] = top_k_unc.mean()
            hits_unc = np.sum(top_k_unc * hits)
            URI['hits'] += n_hits[-1]
            URI['hits_unc'] += hits_unc
            if n_hits[-1] > 0:
                avg_hits_unc = hits_unc / n_hits[-1]
                Rating_rec['URI'][idxu] = (URI['avg'][idxu] - avg_hits_unc) / top_k_unc.std()

            Cuts['MAP'][idxu, 0] = Rating_rec['MAP'][idxu][-1]
            Cuts['Relevance'][idxu, 0] = top_k['scores'].mean()

            for idxc, cut in enumerate(Cuts['Values']):
                if np.sum(top_k_unc < cut) == len(top_k_unc):
                    Cuts['MAP'][idxu, idxc + 1] = Cuts['MAP'][idxu, idxc]
                    Cuts['Coverage'][idxu, idxc + 1] = Cuts['Coverage'][idxu, idxc]
                    Cuts['Relevance'][idxu, idxc + 1] = Cuts['Relevance'][idxu, idxc]
                else:
                    top_k_constrained = rec[unc < cut][:max_k]
                    top_k_unc = top_k_constrained.uncertainties.to_numpy()
                    Cuts['Coverage'][idxu, idxc + 1] = len(top_k_unc) / max_k
                    if len(top_k_constrained > 0):
                        Cuts['Relevance'][idxu, idxc + 1] = top_k_constrained['scores'].mean()
                        hits_ = top_k_constrained.index.isin(targets)
                        if hits_.sum() > 0:
                            Cuts['MAP'][idxu, idxc + 1] = get_AP(hits_)

        if hasattr(model, 'uncertain_predict_user'):
            top_k_unc = model.uncertain_recommend(user, remove_items=rated, n=max_k)
            top_k_unc_unc = 1 - top_k_unc.scores.to_numpy()
            if np.sum(top_k_unc.index.to_numpy() == top_k.index.to_numpy()) == max_k:
                Uncertain_rec['MAP'][idxu] = Rating_rec['MAP'][idxu]
                Uncertain_rec['Recall'][idxu] = Rating_rec['Recall'][idxu]
            else:
                hits = top_k_unc.index.isin(targets)
                n_hits = hits.cumsum(0)

            if n_hits[-1] > 0:
                precision = n_hits / precision_denom
                Uncertain_rec['MAP'][idxu] = np.cumsum(precision * hits) / np.maximum(1, n_hits)
                Uncertain_rec['Recall'][idxu] = n_hits / len(targets)
    
    metrics['Rating_rec'] = {'MAP': np.mean(Rating_rec['MAP'], 0),
                             'Recall': np.mean(Rating_rec['Recall'], 0)}

    if isinstance(model, UncertainRecommender):
        metrics['Rating_rec']['Unc_MAP_corr'] = stats.spearmanr(Rating_rec['MAP'][:, -1], URI['avg'])[0]
        metrics['Rating_rec']['URI'] = np.nanmean(Rating_rec['URI'])

        metrics['Cuts'] = {'Values': Cuts['Values'],
                           'Coverage': Cuts['Coverage'].mean(0),
                           'Relevance': Cuts['Relevance'].mean(0),
                           'MAP': Cuts['MAP'].mean(0),
                           'Map*': np.array([Cuts['MAP'][Cuts['Coverage'][:, k] == 1, k].mean() for k in range(5)])}

    if hasattr(model, 'uncertain_predict_user'):
        metrics['Uncertain_rec'] = {'MAP': np.mean(Uncertain_rec['MAP'], 0),
                                    'Recall': np.mean(Uncertain_rec['Recall'], 0)}

    if name is not None:
        with open('results/' + name + '.pkl', 'wb') as f:
            pickle.dump(metrics, file=f)

    return metrics
