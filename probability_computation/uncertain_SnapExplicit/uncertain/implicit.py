import math
import torch
import numpy as np
from pandas import DataFrame
from scipy.stats import beta
from .core import FactorizationModel, VanillaRecommender, UncertainRecommender


def cross_entropy(positive, negative=None):
    if negative is None:
        return - positive.sigmoid().log().mean()
    else:
        return - torch.cat((positive.sigmoid(), 1 - negative.sigmoid())).log().mean()


class Implicit:

    def get_negative_prediction(self, batch):
        item_ids = torch.randint(0, self.n_item, (len(batch) * self.n_negative,), device=batch.device)
        negative_prediction = self.forward(batch.repeat(self.n_negative), item_ids)
        return negative_prediction

    def training_step(self, batch, batch_idx):
        output = self.forward(batch[:, 0], batch[:, 1])
        loss = self.loss_func(output, self.get_negative_prediction(batch[:, 0]))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user, rated, targets = batch
        rec = self._val_pred(user)
        rec[rated] = -float('inf')
        _, rec = torch.sort(rec, descending=True)
        hits = torch.isin(rec[:5], targets, assume_unique=True)
        n_hits = hits.cumsum(0)
        if n_hits[-1] > 0:
            precision = n_hits / torch.arange(1, 6, device=n_hits.device)
            self.log('val_MAP', torch.sum(precision * hits) / n_hits[-1], prog_bar=True)
        else:
            self.log('val_MAP', torch.tensor(0, dtype=torch.float32), prog_bar=True)
            
    def _val_pred(self, user):
        user_embedding = self.user_embeddings(user)
        return (user_embedding * self.item_embeddings.weight).sum(1)
        

class logMF(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, n_negative=1):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, **dict(loss_func=cross_entropy, n_negative=n_negative))
        

class CAMF(Implicit, FactorizationModel, UncertainRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, n_negative=1):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, **dict(n_negative=n_negative))
        self.linear = torch.nn.Linear(embedding_dim, 2, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        self.activation = torch.nn.Softplus()
        # self.quantile = 0.5
        # self.penalty = penalty

    def configure_optimizers(self):
        params = [{'params': self.user_embeddings.parameters(), 'weight_decay': self.weight_decay},
                  {'params': self.item_embeddings.parameters(), 'weight_decay': self.weight_decay},
                  {'params': self.linear.parameters(), 'weight_decay': 0}]
        return torch.optim.Adam(params, lr=self.lr)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)
        dot = user_embedding * item_embedding
        alpha_beta = self.activation(self.linear(dot))
        return alpha_beta

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            alpha_beta = self(user_ids, item_ids) # .numpy()
            return self.expectation(alpha_beta).numpy(), self.variance(alpha_beta).numpy()
            # return beta.ppf(self.quantile, a=dists[:, 0], b=dists[:, 1])

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            dot = user_embedding * self.item_embeddings.weight
            alpha_beta = self.activation(self.linear(dot))
            return self.expectation(alpha_beta).numpy(), self.variance(alpha_beta).numpy()

    def _val_pred(self, user):
        user_embedding = self.user_embeddings(user)
        dot = user_embedding * self.item_embeddings.weight
        alpha_beta = self.activation(self.linear(dot))
        return self.expectation(alpha_beta)
        
    @staticmethod
    def expectation(alpha_beta):
        return torch.divide(alpha_beta[:, 0], alpha_beta.sum(1)).flatten()

    @staticmethod
    def variance(alpha_beta):
        sum_params = alpha_beta.sum(1)
        return torch.divide(alpha_beta.prod(1), (sum_params**2 * (sum_params + 1))).flatten()

    def loss_func(self, positive_params, negative_params, train=True):
        prob_positive = self.expectation(positive_params)
        prob_negative = self.expectation(negative_params)
        loss = torch.cat((prob_positive, 1 - prob_negative))

        '''
        if train:
            penalty_pos = (positive_params[:, 0] - positive_params[:, 1]).square().sum()
            penalty_neg = (negative_params[:, 0] - negative_params[:, 1]).square().sum()
            penalty = self.penalty * (penalty_pos + penalty_neg)
            return - loss.log().mean() + penalty
        else:
            return loss.mean()
        '''
        return - loss.log().mean()


class GMF(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, n_negative=1):
        super().__init__(n_user, n_item, embedding_dim, lr, 0, cross_entropy, n_negative)
        self.init_net()

    def init_net(self):
        self.linear = torch.nn.Linear(self.embedding_dim, 1, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        return self.linear(user_embeddings * item_embeddings)

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            return self.linear(user_embedding * self.item_embeddings.weight).numpy()

    def _val_pred(self, user):
        user_embedding = self.user_embeddings(user)
        return self.linear(user_embedding * self.item_embeddings.weight)
        

class MLP(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, n_negative=1):
        super().__init__(n_user, n_item, embedding_dim, lr, 0, cross_entropy, n_negative)
        self.init_net()

    def init_net(self):
        self.linear0 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=False)
        self.linear1 = torch.nn.Linear(self.embedding_dim, int(self.embedding_dim/2), bias=False)
        self.linear2 = torch.nn.Linear(int(self.embedding_dim/2), 1, bias=False)
        torch.nn.init.normal_(self.linear0.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        input_embeddings = torch.cat((user_embeddings, item_embeddings), dim=1)
        return self.linear2(self.linear1(self.linear0(input_embeddings)))

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user).expand(self.n_item, self.embedding_dim)
            input_embedding = torch.cat((user_embedding, self.item_embeddings.weight), dim=1)
            return self.linear2(self.linear1(self.linear0(input_embedding))).numpy()

    def _val_pred(self, user):
        user_embedding = self.user_embeddings(user).expand(self.n_item, self.embedding_dim)
        input_embedding = torch.cat((user_embedding, self.item_embeddings.weight), dim=1)
        return self.linear2(self.linear1(self.linear0(input_embedding)))