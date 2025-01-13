import numpy as np
from uncertain.core import UncertainRecommender
from scipy.stats import norm


class UserHeuristic(UncertainRecommender):

    def __init__(self, base_MF, uncertainty):
        self.MF = base_MF
        self.uncertainty = uncertainty

    def predict(self, user_ids, item_ids):
        return self.MF.predict(user_ids, item_ids), self.uncertainty[user_ids]

    def predict_user(self, user):
        return self.MF.predict_user(user), np.full(self.MF.n_item, self.uncertainty[user])


class ItemHeuristic(UncertainRecommender):

    def __init__(self, base_MF, uncertainty):
        self.MF = base_MF
        self.uncertainty = uncertainty

    def predict(self, user_ids, item_ids):
        return self.MF.predict(user_ids, item_ids), self.uncertainty[item_ids]

    def predict_user(self, user):
        return self.MF.predict_user(user), self.uncertainty


class Ensemble(UncertainRecommender):

    def __init__(self, models):
        self.models = models

    @property
    def is_uncertain(self):
        return True

    def predict(self, user_ids, item_ids):
        predictions = np.empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict(user_ids, item_ids)
        scores = predictions.mean(1)
        uncertainties = predictions.std(1)
        return scores, uncertainties

    def predict_user(self, user):
        predictions = np.empty((self.models[0].n_item, len(self.models)))
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict_user(user)
        scores = predictions.mean(1)
        uncertainties = predictions.std(1)
        return scores, uncertainties

    def uncertain_predict(self, user_ids, item_ids, threshold):
        scores, uncertainties = self.predict(user_ids, item_ids)
        return norm.sf(threshold, scores, uncertainties)

    def uncertain_predict_user(self, user, threshold):
        predictions = np.empty((self.models[0].n_item, len(self.models)))
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict_user(user)
        scores = predictions.mean(1)
        uncertainties = predictions.std(1)
        if threshold is not None:
            return norm.sf(threshold, scores, uncertainties)
        else:
            return scores - 4*uncertainties


class Resample(UncertainRecommender):

    def __init__(self, base_MF, models):
        self.MF = base_MF
        self.models = models

    @property
    def is_uncertain(self):
        return True

    def predict(self, user_ids, item_ids):
        predictions = np.empty((len(user_ids), len(self.models)))
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict(user_ids, item_ids)
        uncertainties = predictions.std(1)
        return self.MF.predict(user_ids, item_ids), uncertainties

    def predict_user(self, user):
        predictions = np.empty((self.models[0].n_item, len(self.models)))
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict_user(user)
        uncertainties = predictions.std(1)
        return self.MF.predict_user(user), uncertainties

    def uncertain_predict(self, user_ids, item_ids, threshold):
        scores, uncertainties = self.predict(user_ids, item_ids)
        return norm.sf(threshold, scores, uncertainties)
    
    def uncertain_predict_user(self, user, threshold):
        predictions = np.empty((self.models[0].n_item, len(self.models)))
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict_user(user)
        uncertainties = predictions.std(1)
        if threshold is not None:
            return norm.sf(threshold, self.MF.predict_user(user), uncertainties)
        else:
            return scores - 4*uncertainties

    

class UncertainWrapper(UncertainRecommender):
    """
    Wraps a score estimator with an uncertainty estimator.

    Parameters
    ----------
    scores: :class:`uncertain.models.base`
        A recommendation model.
    uncertainty: :class:`uncertain.models.base`
        An uncertainty estimator: A class containing a predict
        function that returns an uncertainty estimate for the
        given user.
    """
    def __init__(self, scores, uncertainty):
        self.scores = scores
        self.uncertainty = uncertainty

    def predict(self, user_ids, item_ids):
        return self.scores.predict(user_ids, item_ids), self.uncertainty.predict(user_ids, item_ids)

    def predict_user(self, user):
        return self.scores.predict_user(user), self.uncertainty.predict_user(user)

    def uncertain_predict(self, user_ids, item_ids, threshold):
        scores, uncertainties = self.predict(user_ids, item_ids)
        return norm.sf(threshold, scores, uncertainties)

    def uncertain_predict_user(self, user, threshold):
        return norm.sf(threshold, self.scores.predict_user(user), self.uncertainty.predict_user(user))
