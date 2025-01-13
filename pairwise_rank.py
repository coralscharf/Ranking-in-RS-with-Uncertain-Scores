from typing import Tuple, List
import time
import torch
from utils import generate_universe


class PairwiseRank:
    def __init__(self, score_tuples: List[Tuple], k: int, device='cpu'):
        """
        :param score_tuples: tuples of (sigma_m, sigma_n)
        """

        self.score_tuples = score_tuples
        self.device = device
        self.upper_scores = torch.Tensor([x[0] for x in self.score_tuples]).to(device)
        self.lower_scores = torch.Tensor([x[1] for x in self.score_tuples]).to(device)
        self.k = k
        self.result = None

    def calculate_pairwise_rank(self, score_probabilities: torch.Tensor, score_values: torch.Tensor):
        """
        Function computing probability of each tuple (sigma_n, sigma_m) being at any rank <= k in a provided universe

        :param score_probabilities: distribution of scores. N lines, #possible_scores columns.
        Entry (i,j) represents probability of tuple i to have a j'th score
        :param score_values: scores of tuples. N lines, #possible_scores columns.
        Entry (i,j) represents j'th score of tuple i.

        :return: array of probabilities (n_tuples, rank_m, rank_n).
        P[i,m,n] represents P(rank(upper score in tuple i) = m, P(rank(lower score in tuple i) = n) in universe U+i
        """
        self.result = torch.zeros(len(self.score_tuples), self.k, self.k+1).to(self.device)
        self.result[:, 0, 1] = 1
        self.result = torch.log(self.result)
        for j, j_score_values in enumerate(score_values):
            self.update_result(score_probabilities[j], score_values[j])
        return self.result

    def update_result(self, score_probabilities, score_values):
        """
        Function performing "addition" of an item with given probabilities/values to tuples worlds.
        :param previous_result: probability array (n_tuples, rank_m, rank_n)
        :param score_probabilities: probabilities of scores for a new item
        :param score_values: values of scores for a new item
        :return: updated probability array (n_tuples, rank_m, rank_n)
        """
        log_p_lower, log_p_between, log_p_higher = self._get_relative_log_probabilities(score_probabilities, score_values)

        new_result = self.result + log_p_lower.reshape(-1, 1, 1)

        between_result = self.result + log_p_between.reshape(-1, 1, 1)
        new_result[:, :, 1:] = torch.logaddexp(new_result[:, :, 1:], between_result[:, :, :-1])
        new_result[:, :, -1] = torch.logaddexp(new_result[:, :, -1], between_result[:, :, -1])  # values in the last column (P[rank_n > k]) should only increase

        higher_result = self.result + log_p_higher.reshape(-1, 1, 1)
        new_result[:, 1:, 1:] = torch.logaddexp(new_result[:, 1:, 1:], higher_result[:, :-1, :-1])
        new_result[:, 1:, -1] = torch.logaddexp(new_result[:, 1:, -1], higher_result[:, :-1, -1])  # values in the last column (P[rank_n > k]) should only increase

        self.result = new_result

    def _get_relative_log_probabilities(self, j_score_probabilities, j_score_values):
        j_score_values = j_score_values.reshape(-1, 1)
        lower_equality = (j_score_values == self.lower_scores)
        upper_equality = (j_score_values == self.upper_scores)

        lower_score_mask = lower_equality*0.5 + (j_score_values < self.lower_scores)

        between_score_mask = upper_equality*0.5 + (j_score_values < self.upper_scores)
        between_score_mask *= lower_equality*0.5 + (j_score_values > self.lower_scores)

        higher_score_mask = upper_equality*0.5 + (j_score_values > self.upper_scores)

        full_equality = lower_equality & upper_equality
        lower_score_mask[full_equality] = between_score_mask[full_equality] = higher_score_mask[full_equality] = 1/3

        p_lower = j_score_probabilities.matmul(lower_score_mask)
        p_between = j_score_probabilities.matmul(between_score_mask)
        p_higher = j_score_probabilities.matmul(higher_score_mask)

        return torch.log(p_lower), torch.log(p_between), torch.log(p_higher)


def performance_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    k = 10
    n_tuples = 50000
    n_grades = 50
    size = 50000
    for i in range(1):
        score_probabilities, score_values = generate_universe(size, n_grades)
        score_probabilities = score_probabilities.to(device).type(torch.float32)
        score_values = score_values.to(device).type(torch.float32)
        torch.manual_seed(0)
        tuples = torch.rand(n_tuples*2).reshape(n_tuples, 2) * 100
        tuples.sort(axis=1)

        rank_dist = PairwiseRank(tuples, k, device)
        start = time.time()
        result = rank_dist.calculate_pairwise_rank(score_probabilities, score_values)
        print(f'k={k}. n_grades={n_grades}. universe_size={size}. n_tuples={n_tuples}. Took {time.time() - start} seconds')
        start = time.time()
        #result = rank_dist.pairwise_rank_full()
        #print(f'k={k}. n_grades={n_grades}. universe_size={size}. n_tuples={n_tuples}. Full run took {time.time() - start} seconds')


def run_small_test():
    device = 'cpu'
    score_probabilities = torch.Tensor([[0.5, 0.5], [0.5, 0.5]]).to(device)
    score_values = torch.Tensor([[2, 5], [0, 6]]).to(device)
    tuples = [(3, 4)]
    k = 2

    rank_dist = PairwiseRank(tuples, k, device)

    result = rank_dist.calculate_pairwise_rank(score_probabilities, score_values)
    print(result)
    print(torch.exp(result))


if __name__ == "__main__":
    performance_test()
    # run_small_test()
