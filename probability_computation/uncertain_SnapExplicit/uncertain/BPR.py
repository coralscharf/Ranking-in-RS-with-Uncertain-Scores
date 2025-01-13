import math
import torch
import numpy as np
from .core import FactorizationModel, VanillaRecommender, UncertainRecommender
from .implicit import Implicit


class ProbabilisticLoss:

    def __init__(self, log_scale=True):
        self.log_scale = log_scale

    def get_prob(self, x, sigma):
        """Override this"""
        return None

    def __call__(self, *args):
        prob = self.get_prob(*args)
        if self.log_scale:
            prob = prob.log()
        return - prob.mean()


class BPR(ProbabilisticLoss):

    def __init__(self, log_scale=True):
        super().__init__(log_scale)

    def get_prob(self, positive, negative):
        x = positive - negative
        return torch.sigmoid(x)


class ABPR(ProbabilisticLoss):

    def __init__(self, log_scale=True):
        super().__init__(log_scale)

    def get_prob(self, positive, negative):
        x = positive[0] - negative[0]
        rho = positive[1] + negative[1]
        return torch.sigmoid(x / rho)


class GPR(ProbabilisticLoss):

    def __init__(self, log_scale=True):
        super().__init__(log_scale)

    def get_prob(self, positive, negative):
        x = positive[0] - negative[0]
        rho = positive[1] + negative[1]
        return 0.5 * (1 + torch.erf(x / torch.sqrt(2*rho)))

    
class bprMF(Implicit, FactorizationModel, VanillaRecommender):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, **dict(loss_func=BPR(), n_negative=1))
        self.rho_activation = torch.nn.Softplus()
        
        
class UncertainBPR(Implicit, FactorizationModel, UncertainRecommender):
    
    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, **dict(loss_func=loss, n_negative=1, padding=0))

        
class biasMF(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.user_bias = torch.nn.Embedding(self.n_user, 1)
        self.item_bias = torch.nn.Embedding(self.n_item, 1)
        torch.nn.init.normal_(self.user_bias.weight, mean=1, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, mean=1, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()
        
    def forward(self, user_ids, item_ids):
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(item_ids)
        return self.dot(user_ids, item_ids), self.rho_activation(user_bias + item_bias).flatten() + self.padding
    
    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            user_bias = self.user_bias(user)
            mean = (user_embedding * self.item_embeddings.weight).sum(1).numpy()
            unc = self.rho_activation(user_bias + self.item_bias.weight).flatten().numpy() + self.padding
            return mean, unc
        
    def uncertain_predict_user(self, user, threshold):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            user_bias = self.user_bias(user)
            mean = (user_embedding * self.item_embeddings.weight).sum(1).numpy()
            unc = self.rho_activation(user_bias + self.item_bias.weight).flatten().numpy() + self.padding
            return mean - 4*np.sqrt(unc)

        
class TwoWayMF(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.user_embeddings_rho = torch.nn.Embedding(self.n_user, embedding_dim)
        self.item_embeddings_rho = torch.nn.Embedding(self.n_item, embedding_dim)
        torch.nn.init.normal_(self.user_embeddings_rho.weight, mean=1/self.embedding_dim, std=0.01)
        torch.nn.init.normal_(self.item_embeddings_rho.weight, mean=1/self.embedding_dim, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        params = [{'params': self.user_embeddings.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
                  {'params': self.item_embeddings.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
                  {'params': self.user_embeddings_rho.parameters(), 'lr': self.lr*10, 'weight_decay': self.weight_decay},
                  {'params': self.item_embeddings_rho.parameters(), 'lr': self.lr*10, 'weight_decay': self.weight_decay},]
        return torch.optim.Adam(params)

    def forward(self, user_ids, item_ids):
        user_embedding_rho = self.user_embeddings_rho(user_ids)
        item_embedding_rho = self.item_embeddings_rho(item_ids)
        return self.dot(user_ids, item_ids), self.rho_activation((user_embedding_rho * item_embedding_rho).sum(1)) + self.padding
    
    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            user_embedding_rho = self.user_embeddings_rho(user)
            mean = (user_embedding * self.item_embeddings.weight).sum(1).numpy()
            unc = self.rho_activation((user_embedding_rho * self.item_embeddings_rho.weight).sum(1)).numpy() + self.padding
            return mean, unc
        
    def uncertain_predict_user(self, user, threshold):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            user_embedding_rho = self.user_embeddings_rho(user)
            mean = (user_embedding * self.item_embeddings.weight).sum(1).numpy()
            unc = self.rho_activation((user_embedding_rho * self.item_embeddings_rho.weight).sum(1)).numpy() + self.padding
            return mean - 4*np.sqrt(unc)


class bprGMF(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.linear = torch.nn.Linear(self.embedding_dim, 2, bias=False)
        torch.nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        out = self.linear(user_embeddings * item_embeddings)
        return out[:, 0], self.rho_activation(out[:, 1])

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            out = self.linear(user_embedding * self.item_embeddings.weight)
            return out[:, 0].numpy(), self.rho_activation(out[:, 1]).numpy()

    def uncertain_predict_user(self, user, threshold):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user)
            out = self.linear(user_embedding * self.item_embeddings.weight)
            return out[:, 0].numpy() - 4*np.sqrt(self.rho_activation(out[:, 1])).numpy()


class bprMLP(UncertainBPR):

    def __init__(self, n_user, n_item, embedding_dim, lr, weight_decay, loss):
        super().__init__(n_user, n_item, embedding_dim, lr, weight_decay, loss)
        self.init_net()

    def init_net(self):
        self.linear0 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=False)
        self.linear1 = torch.nn.Linear(self.embedding_dim, int(self.embedding_dim/2), bias=False)
        self.linear2 = torch.nn.Linear(int(self.embedding_dim/2), 2, bias=False)
        torch.nn.init.normal_(self.linear0.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.01)
        self.rho_activation = torch.nn.Softplus()
        self.save_hyperparameters()

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        input_embeddings = torch.cat((user_embeddings, item_embeddings), dim=1)
        out = self.linear2(self.linear1(self.linear0(input_embeddings)))
        return out[:, 0], self.rho_activation(out[:, 1])

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            out = self(user_ids, item_ids)
            return out[0].numpy(), out[1].numpy()

    def predict_user(self, user):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user).expand(self.n_item, self.embedding_dim)
            input_embedding = torch.cat((user_embedding, self.item_embeddings.weight), dim=1)
            out = self.linear2(self.linear1(self.linear0(input_embedding)))
            return out[:, 0].numpy(), self.rho_activation(out[:, 1]).numpy()
        
    def uncertain_predict_user(self, user, threshold):
        with torch.no_grad():
            user_embedding = self.user_embeddings(user).expand(self.n_item, self.embedding_dim)
            input_embedding = torch.cat((user_embedding, self.item_embeddings.weight), dim=1)
            out = self.linear2(self.linear1(self.linear0(input_embedding)))
            return out[:, 0].numpy() - 4*np.sqrt(self.rho_activation(out[:, 1])).numpy()
