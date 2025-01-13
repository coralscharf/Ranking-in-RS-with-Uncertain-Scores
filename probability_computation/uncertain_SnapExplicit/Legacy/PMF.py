"""
Factorization models for explicit feedback problems.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.evaluation import rmse_score, precision_recall_score
from spotlight.torch_utils import gpu, assert_no_grad


class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_embeddings.weight.data.uniform_(-.01, .01)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_embeddings.weight.data.uniform_(-.01, .01)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot


class PMF(object):

    def __init__(self,
                 embedding_dim,
                 n_iter,
                 lambda_u,
                 lambda_v,
                 learning_rate,
                 batch_size,
                 use_cuda):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._lambda_u = lambda_u
        self._lambda_v = lambda_v
        self._batch_size = batch_size
        self._use_cuda = use_cuda

        self._num_users = None
        self._num_items = None
        self._num_ratings = None
        self._net = None
        self._optimizer = None

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items,
         self._num_ratings) = (interactions.num_users,
                               interactions.num_items,
                               len(interactions.ratings))

        self._net = gpu(BilinearNet(self._num_users,
                                    self._num_items,
                                    self._embedding_dim),
                        self._use_cuda)

        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._learning_rate
            )

        self.train_loss = []

    def _mse(self, observed_ratings, predicted_ratings):

        assert_no_grad(observed_ratings)
        return ((observed_ratings - predicted_ratings) ** 2).sum()

    def _add_penalty(self, size):

        penalty_u = (self._net.user_embeddings.weight ** 2).sum() * (self._lambda_u*size/self._num_ratings)
        penalty_v = (self._net.item_embeddings.weight ** 2).sum() * (self._lambda_v*size/self._num_ratings)

        return penalty_u + penalty_v

    def fit(self, interactions, test, verbose=False):

        if not self._initialized:
            self._initialize(interactions)
            if test:
                self.test_loss = []

        user_ids_tensor = gpu(torch.from_numpy(interactions.user_ids.astype(np.int64)), self._use_cuda)
        item_ids_tensor = gpu(torch.from_numpy(interactions.item_ids.astype(np.int64)), self._use_cuda)
        ratings_tensor = gpu(torch.from_numpy(interactions.ratings), self._use_cuda)

        for epoch_num in range(self._n_iter):

            idx = np.arange(0, self._num_ratings)
            np.random.shuffle(idx)
            idx = gpu(torch.tensor(idx), self._use_cuda)
            user_ids_tensor = user_ids_tensor[idx]
            item_ids_tensor = item_ids_tensor[idx]
            ratings_tensor = ratings_tensor[idx]

            self.train_loss.append(0)

            for i in range(0, self._num_ratings, self._batch_size):

                self._optimizer.zero_grad()

                predictions = self._net(user_ids_tensor[i:i + self._batch_size],
                                        item_ids_tensor[i:i + self._batch_size])

                loss = self._mse(ratings_tensor[i:i + self._batch_size], predictions)
                self.train_loss[-1] += loss.item()
                loss += self._add_penalty(len(user_ids_tensor[i:i + self._batch_size]))
                loss.backward()
                self._optimizer.step()

            self.train_loss[-1] /= self._num_ratings
            if test:
                predictions = model.predict(test.user_ids, test.item_ids)
                loss = self._mse(torch.tensor(test.ratings), torch.tensor(predictions))
                self.test_loss.append(loss.item()/len(test.ratings))

            if verbose:
                print('{}-th epoch loss: '.format(epoch_num), (self.train_loss[-1], self.test_loss[-1]))

    def predict(self, user_ids, item_ids=None):

        self._net.train(False)
        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self._num_items,
                                                  self._use_cuda)
        out = self._net(user_ids, item_ids)

        return out.cpu().detach().numpy().flatten()

    def evaluate(self, test, train):

        self.rmse = rmse_score(self, test)
        p, r = precision_recall_score(self, test, train, np.arange(1, 11))
        self.precision = p.mean(axis=0)
        self.recall = r.mean(axis=0)



from utils import dataset_loader
train, test = dataset_loader('100K')

model = PMF(embedding_dim=50, n_iter=200, lambda_u=.5, lambda_v=.5, learning_rate=.001, batch_size=8196, use_cuda=True)
model.fit(train, test, verbose=True)
model.evaluate(test, train)
