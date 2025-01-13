import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
from spotlight.evaluation import precision_recall_score
from spotlight.factorization._components import _predict_process_ids
import torch


class PureSVD(object):

    def __init__(self, embedding_dim, n_iter, random_state):

        self.embedding_dim = embedding_dim
        self.n_iter = n_iter
        self.random_state = random_state

        self.user_embeddings = None
        self.item_embeddings = None

    def fit(self, interactions):

        U, Sigma, VT = randomized_svd(train.tocsr(),
                                      n_components=self.embedding_dim,
                                      n_iter=self.n_iter,
                                      random_state=self.random_state)

        self.user_embeddings = U * sps.diags(Sigma)
        self.item_embeddings = VT.T

    def predict(self, user_ids, item_ids=None):

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids,
                                                  self.item_embeddings.shape[0], False)

        u = self.user_embeddings[user_ids]
        i = self.item_embeddings[item_ids]

        return np.multiply(u, i).sum(1)

    def evaluate(self, test, train):

        self.precision, self.recall = precision_recall_score(self, test, train, 10)


class EigenvalueNet(torch.nn.Module):

    def __init__(self, ratings_matrix, sim_matrix, lr):

        super().__init__()

        self.R = torch.tensor(ratings_matrix.tocsr()[1:, 1:].todense()).cuda()
        self.sim_matrix = torch.tensor(sim_matrix).cuda()

        self.eigen = torch.nn.Embedding(ratings_matrix.shape[0]-1, 1).cuda()
        self.eigen.weight.data.uniform_(0, .01)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(100):
            self.optimizer.zero_grad()
            loss = self()
            loss.backward()
            self.optimizer.step()
            print(loss)

        self.avg = self.eigen.weight.mean().item()
        self.std = self.eigen.weight.std().item()
        self.confidence = (self.eigen.weight.cpu().detach().numpy().flatten() - self.avg) / (self.std)

    def forward(self):

            return ((torch.mm(self.R, self.sim_matrix) - torch.mm(self.eigen.weight.diagflat(), self.R))**2).mean()


class SimilarityRecommender(object):

    def __init__(self):

        self.sim_matrix = None
        self.train_csr = None

    def fit(self, train_csr):

        self.sim_matrix = cosine_similarity(train_csr.T, dense_output=False)
        self.train_csr = train_csr

    def predict(self, user_ids):

        return np.insert(self.train_csr[user_ids].dot(self.sim_matrix).toarray().flatten(), 0, 0)

    def evaluate(self, test, train):

        self.precision, self.recall = precision_recall_score(self, test, train, 10)


from utils import dataset_loader
train, test = dataset_loader('1M')
train_csr = train.tocsr()[1:, 1:]
test_nz = test.tocsr()[1:].getnnz(1) > 0

PureSVDModel = PureSVD(embedding_dim=50, n_iter=10, random_state=0)
PureSVDModel.fit(train)
PureSVDModel.evaluate(test, train)
sim_matrix = np.matmul(PureSVDModel.item_embeddings[1:], PureSVDModel.item_embeddings.T[:, 1:])
PureSVDEigen = EigenvalueNet(train.tocsr(), sim_matrix, 0.01)
SVDConfidence = PureSVDEigen.confidence[test_nz]
np.corrcoef(PureSVDModel.precision, SVDConfidence)

KNN = SimilarityRecommender()
KNN.fit(train_csr)
KNN.evaluate(test.tocsr()[1:, 1:], train_csr)
KNNEigen = EigenvalueNet(train.tocsr(), KNN.sim_matrix.todense(), 100)
KNNConfidence = KNNEigen.confidence[test_nz]
np.corrcoef(KNN.precision, KNNConfidence)

which = (PureSVDEigen.confidence[test_nz] > KNNEigen.confidence[test_nz]).flatten()
model1 = PureSVDModel.precision[which]
model2 = KNN.precision[~which]
precision_merge = np.concatenate((model1, model2))
precision_merge.mean()