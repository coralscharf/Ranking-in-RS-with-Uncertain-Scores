import torch
from pandas import DataFrame
from numpy import column_stack
from pytorch_lightning import LightningModule


class BiasModel(LightningModule):

    def __init__(self, n_user, n_item, lr):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.lr = lr
        self.user_bias = torch.nn.Embedding(self.n_user, 1)
        self.item_bias = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.normal_(self.user_bias.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, mean=0, std=0.01)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, user_ids, item_ids):
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)
        return (user_bias + item_bias).flatten()

    @staticmethod
    def loss_func(predicted, observed):
        return ((observed - predicted) ** 2).sum()

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_bias = self.user_bias(user)
            return (user_bias + self.item_bias.weight).flatten().numpy()


class FactorizationModel(LightningModule):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, **kwargs):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.user_embeddings = torch.nn.Embedding(self.n_user, self.embedding_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_item, self.embedding_dim)
        torch.nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embeddings.weight, mean=0, std=0.01)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def dot(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        return (user_embeddings * item_embeddings).sum(1)

    def forward(self, user_ids, item_ids):
        return self.dot(user_ids, item_ids)

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids).numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            return (user_embedding * self.item_embeddings.weight).sum(1).numpy()


class VanillaRecommender:

    def recommend(self, user, remove_items=None, n=10):
        out = DataFrame(self.predict_user(torch.tensor(user)), columns=['scores'])
        if remove_items is not None:
            out.loc[remove_items, 'scores'] = -float('inf')
        out = out.sort_values(by='scores', ascending=False)[:n]
        return out


class UncertainRecommender:

    def recommend(self, user, remove_items=None, n=10):
        out = DataFrame(column_stack(self.predict_user(torch.tensor(user))))
        out.columns = ['scores', 'uncertainties']
        if remove_items is not None:
            out.loc[remove_items, 'scores'] = -float('inf')
        out = out.sort_values(by='scores', ascending=False)[:n]
        return out

    def uncertain_recommend(self, user, threshold=None, remove_items=None, n=10):
        out = DataFrame(self.uncertain_predict_user(torch.tensor(user), threshold), columns=['scores'])
        if remove_items is not None:
            out.loc[remove_items, 'scores'] = -float('inf')
        out = out.sort_values(by='scores', ascending=False)[:n]
        return out
