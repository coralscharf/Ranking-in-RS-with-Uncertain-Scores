import numpy as np
import time
from utils import generate_universe
import torch
from collections import defaultdict


class ConsecutivePairRank:
    def __init__(self, score_probabilities: torch.Tensor, score_values=torch.Tensor([]), device='cpu'):
        """
        :param data: distribution of scores. N lines, #possible_scores columns.
        Entry (i,j) represents probability of tuple i to have a j'th score
        :param scores: scores of tuples. N lines, #possible_scores columns.
        Entry (i,j) represents j'th score of tuple i. If empty then scores are assumed to be equal to indices
        """

        self.universe_len = score_probabilities.shape[0]
        self.score_probabilities = score_probabilities.cpu().numpy()
        self.n_scores = self.score_probabilities.shape[1]
        if len(score_values) == 0:
            self.score_values = np.repeat(np.arange(self.n_scores).reshape(1,-1), self.universe_len, axis=0)
        else:
            self.score_values = score_values.cpu().numpy()
        self.p_lower = defaultdict(int)
        all_possible_scores = set(self.score_values.flatten())
        self.device = device
        for j in range(self.universe_len):
            for score in all_possible_scores:
                self.p_lower[j, score] = ((self.score_values[j] < score)*self.score_probabilities[j]).sum()
                self.p_lower[j, score] += ((self.score_values[j] == score)*self.score_probabilities[j]).sum()/2  # random tie-breaking
                self.p_lower[j, score] = min(self.p_lower[j, score], 1)

    def calculate_consecutive_rank(self):
        """
        Computes probability of each tuple (s_m,s_n) being ranked at places (i, i+1) for each i < k
        :return: dict {(m,n,s_m,s_n): array of k-1 log probabilities}
        """
        results = dict()
        for m in range(self.universe_len):
            for n in range(self.universe_len):
                if n == m:
                    continue
                for s_m, probability_of_s_m in zip(self.score_values[m], self.score_probabilities[m]):
                    for s_n, probability_of_s_n in zip(self.score_values[n], self.score_probabilities[n]):
                        if s_n > s_m:
                            continue
                        result = np.zeros(self.universe_len - 1)
                        result[0] = probability_of_s_n * probability_of_s_m
                        if s_n == s_m:
                            result[0] /= 2  # random tie-breaking
                        for j in range(self.universe_len):
                            if j == n or j == m:
                                continue
                            new_result = result * self.p_lower[j, s_n]
                            new_result[1:] += (result * (1 - self.p_lower[j, s_m]))[:-1]
                            result = new_result
                        if np.any(result):
                            results[m, n, s_m, s_n] = np.log(result)
        return results


def performance_test():
    device = 'cpu'
    for k in [5, 10, 20, 50]:
        for n_scores in [5, 10, 20, 50, 100]:
            score_probabilities, score_values = generate_universe(k, n_scores)
            rank_dist = ConsecutivePairRank(score_probabilities.to(device), score_values.to(device))
            start = time.time()
            result = rank_dist.calculate_consecutive_rank()
            print(f'k={k}. n_scores={n_scores}. Took {time.time() - start} seconds')


def run_small_test():
    device = 'cpu'
    score_probabilities = torch.from_numpy(np.array([[0.4, 0.6],
                     [0.3, 0.7],
                     [0.2, 0.8]])).to(device)
    score_values = torch.from_numpy(np.array([[5, 10],
                     [2, 7],
                     [3, 8]])).to(device)
    rank_dist = ConsecutivePairRank(score_probabilities, score_values)
    result = rank_dist.calculate_consecutive_rank()
    print(result)


if __name__ == "__main__":
    performance_test()
    # run_small_test()
