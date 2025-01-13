import torch
import numpy as np
from uncertain import Interactions
from matplotlib import pyplot as plt
from uncertain.metrics import rmse_score, error_uncertainty_correlation, rpi_score, classification


def predict_interactions(self, interactions, batch_size=1e100):
    if batch_size > len(interactions):
        return self.predict(interactions.users, interactions.items)
    else:
        batch_size = int(batch_size)
        est = torch.empty(len(interactions), device=self.device)
        if self.is_uncertain:
            unc = torch.empty(len(interactions), device=self.device)

        loader = interactions.minibatch(batch_size)
        for minibatch_num, (users, items, _) in enumerate(loader):
            preds = self.predict(users, items)
            if self.is_uncertain:
                est[(minibatch_num * batch_size):((minibatch_num + 1) * batch_size)] = preds[0]
                unc[(minibatch_num * batch_size):((minibatch_num + 1) * batch_size)] = preds[1]
            else:
                est[(minibatch_num * batch_size):((minibatch_num + 1) * batch_size)] = preds

        if self.is_uncertain:
            return est, unc
        else:
            return est

def uncertainty_distributions(model, size, upper_bound):

    users = torch.randint(0, model.num_users, (size,))
    items = torch.randint(0, model.num_items, (size,))
    uncertainties = model.forward(users, items).numpy()
    cuts = np.quantile(uncertainties, [0.25, 0.5, 0.75])
    count, bins, _ = plt.hist(uncertainties[uncertainties < upper_bound], 30, density=True)
    count = [0] + list(count) + [0]
    bins = list(bins) + [bins[-1] + (bins[-1] - bins[-2])]

    cuts = {'First quartile': {'cut': cuts[0], 'color': 'r'},
            'Median': {'cut': cuts[1], 'color': 'b'},
            'Third quartile': {'cut': cuts[2], 'color': 'g'}}

    f, ax = plt.subplots()
    ax.plot(bins, count, color='k')
    for key, value in cuts.items():
        ax.axvline(x=value['cut'], color=value['color'], linestyle='dashed', label=key)
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Density')
    ax.legend()
    f.tight_layout()

    return f


def evaluate_ratings(model, test):
    model.evaluation['Ratings'] = {}
    predictions = model.forward(test.users, test.items)

    if not model.is_uncertain:
        model.evaluation['Ratings']['RMSE'] = rmse_score(predictions, test.scores)
    else:
        model.evaluation['Ratings']['RMSE'] = rmse_score(predictions[0], test.scores)

        error = torch.abs(test.scores - predictions[0])
        model.evaluation['Ratings']['RPI'] = rpi_score(error, predictions[1])
        model.evaluation['Ratings']['Classification'] = classification(error, predictions[1])
        idx = torch.randperm(len(predictions[1]))[:int(1e5)]
        quantiles = torch.quantile(predictions[1][idx],
                                   torch.linspace(0, 1, 21, device=predictions[1].device, dtype=predictions[1].dtype))
        model.evaluation['Ratings']['Quantile RMSE'] = torch.zeros(20)
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= predictions[1], predictions[1] < quantiles[idx + 1])
            model.evaluation['Ratings']['Quantile RMSE'][idx] = torch.sqrt(torch.square(error[ind]).mean())
        model.evaluation['Ratings']['Correlation'] = error_uncertainty_correlation((error, predictions[1]))


def evaluate_recommendations(model, test, train, relevance_threshold=4, max_k=10):
    model.evaluation['Recommendations'] = {}

    precision = []
    recall = []
    ndcg = []
    rri = []
    arange = torch.arange(1, max_k + 1, dtype=torch.float64)

    for user in range(test.num_users):

        if relevance_threshold is not None and test.type is not 'Implicit':
            targets = test.items[torch.logical_and(test.users == user,
                                                   test.scores >= relevance_threshold)]
        else:
            targets = test.items[test.users == user]

        if not len(targets):
            continue

        rec = model.recommend(user, train.get_rated_items(user))
        hits = get_hits(rec, targets)
        num_hit = hits.cumsum(0)

        precision.append(num_hit / arange)
        recall.append(num_hit / len(targets))
        denom = torch.log2(arange + 1)
        ndcg.append(torch.cumsum(hits / denom, 0) / torch.cumsum(hits.sort(descending=True)[0] / denom, 0))

        if model.is_uncertain and hits.sum().item() > 0:
            rri_ = torch.empty(max_k - 1)
            for i in range(2, max_k + 1):
                unc = rec.uncertainties[:i]
                rri_[i - 2] = (unc.mean() - unc[hits[:i]]).mean() / unc.std()
            rri.append(rri_)

    model.evaluation['Recommendations']['Precision'] = torch.vstack(precision).mean(0)
    model.evaluation['Recommendations']['Recall'] = torch.vstack(recall).mean(0)
    ndcg = torch.vstack(ndcg)
    model.evaluation['Recommendations']['NDCG'] = ndcg.nansum(0) / (~ndcg.isnan()).float().sum(0)
    if len(rri) > 0:
        rri = torch.vstack(rri)
        model.evaluation['Recommendations']['RRI'] = rri.nansum(0) / (~rri.isnan()).float().sum(0)


def evaluate_strategies():
    return 0

'''
def evaluate_zhu(model, test, relevance_threshold=None, max_k=20):
    out = {}
    loader = minibatch(test, batch_size=int(1e5))
    est = []
    if model.is_uncertain:
        unc = []
        for interactions, _ in loader:
            predictions = model.predict(interactions[:, 0], interactions[:, 1])
            est.append(predictions[0])
            unc.append(predictions[1])
        unc = torch.hstack(unc)
    else:
        for interactions, _ in loader:
            est.append(model.predict(interactions[:, 0], interactions[:, 1]))
    est = torch.hstack(est)

    error = torch.abs(test.scores - est)
    out['MAE'] = error.mean().item()
    out['RMSE'] = error.square().mean().sqrt().item()

    if model.is_uncertain:
        unc_avg, unc_ste = unc.mean(), (unc - unc.mean()).abs().mean()
        precision = []
        rri = []
        denominator = torch.arange(1, max_k + 1, device=test.interactions.device)
        for user in range(model.num_users):
            idx = test.interactions[:, 0] == user
            if idx.sum() == 0:
                continue
            relevance = test.scores[idx] >= relevance_threshold
            est_, unc_ = est[idx], unc[idx]
            recommendations = est_.argsort(descending=True)
            hit_indicator = torch.full([max_k], np.nan, device=test.interactions.device)
            hit_indicator[:len(recommendations)] = relevance[recommendations][:max_k]
            precision.append(hit_indicator.cumsum(0) / denominator)
            recommendation_unc = torch.full([max_k], np.nan, device=test.interactions.device)
            recommendation_unc[:len(recommendations)] = unc_[recommendations][:max_k]
            rri.append((hit_indicator * (unc_avg - recommendation_unc).cumsum(0)) /
                       hit_indicator.cumsum(0) / unc_ste)

        out['Precision'] = np.nanmean(torch.vstack(precision).cpu().numpy(), axis=0).tolist()
        out['RRI'] = np.nanmean(torch.vstack(rri).cpu().numpy(), axis=0).tolist()

        out['RPI'] = rpi_score(error, unc).item()
        p, s = correlation(error, unc)
        out['Pearson'], out['Spearman'] = p.item(), s.item()
        out['Quantile RMSE'] = []
        idx = torch.randperm(len(unc))[:int(1e5)]
        quantiles = torch.quantile(unc[idx], torch.linspace(0, 1, 21, device=unc.device, dtype=unc.dtype))
        for idx in range(20):
            ind = torch.bitwise_and(quantiles[idx] <= unc, unc < quantiles[idx + 1])
            out['Quantile RMSE'].append(torch.sqrt(torch.square(error[ind]).mean()).item())

        out['AUC'] = classification(error, unc).item()

    else:
        precision = []
        denominator = torch.arange(1, max_k + 1, device=test.interactions.device)
        for user in range(model.num_users):
            idx = test.interactions[:, 0] == user
            if idx.sum() == 0:
                continue
            relevance = test.scores[idx] >= relevance_threshold
            est_ = est[idx]

            hits = torch.full([max_k], np.nan, device=test.interactions.device)
            hits_ = relevance[est_.argsort(descending=True)][:max_k]
            hits[:len(hits_)] = hits_
            precision.append(hits.cumsum(0) / denominator)

        out['Precision'] = np.nanmean(torch.vstack(precision).cpu().numpy(), axis=0).tolist()

    return out
'''
