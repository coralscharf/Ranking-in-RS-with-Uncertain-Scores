import time
import pandas as pd
import ast
import numpy as np
import logging
import torch
from utils import generate_universe
from tqdm import tqdm
from constants import *
import math
import csv


class RankDist:
    def __init__(self, score_probabilities: torch.Tensor, score_values=torch.Tensor([]), device='cpu'):
        """
        :param data: distribution of scores. N lines, #possible_scores columns.
        Entry (i,j) represents probability of tuple i to have a j'th score
        :param scores: scores of tuples. N lines, #possible_scores columns.
        Entry (i,j) represents j'th score of tuple i. If empty then scores are assumed to be equal to indices
        """

        self.universe_len = score_probabilities.shape[0]
        self.score_probabilities = score_probabilities.float().to(device)
        self.score_probabilities = torch.nn.functional.normalize(self.score_probabilities, dim=1, p=1)
        self.log_score_probabilities = torch.log(score_probabilities)
        self.n_scores = self.score_probabilities.shape[1]
        if len(score_values):
            self.score_values = score_values.float().to(device)
        else:
            self.score_values = torch.arange(self.n_scores).reshape(1, -1).repeat(self.universe_len, 1).to(device)
        self.device = device

        # mask = (np.tril(np.ones((self.n_scores, self.n_scores))) - np.eye(self.n_scores, self.n_scores) * 0.5).T
        # self.lower_score_probability = self.score_probabilities.dot(mask)  # P(S_j < l)

    def rank_dist_with_tiebreaking_one_by_one(self, k) -> torch.Tensor:
        lower_score_probability, equal_score_probability, higher_score_probability = self._calculate_score_probabilities()
        result = torch.log(torch.zeros(self.universe_len, k, self.n_scores).to(self.device))

        for i in tqdm(range(self.universe_len)):
            # Matrix storing log probabilities of each score of example i being in the first k places with e ties
            conditioned_matrix = torch.zeros(self.universe_len, k, self.n_scores).to(self.device)
            conditioned_matrix[0, 0, :] = self.score_probabilities[i]
            conditioned_matrix = torch.log(conditioned_matrix)
            for j in range(self.universe_len):
                if j == i:
                    # Skip the example itself
                    continue
                # Apply probabilities of example j on conditioned matrix
                conditioned_matrix = self._apply_matrices(conditioned_matrix, lower_score_probability[j],
                                                          equal_score_probability[j], higher_score_probability[j])

            # Resolve ties by diving each probability by the number of ties
            multiplier = torch.log(torch.tensor([1 / i for i in range(1, self.universe_len + 1)])).to(self.device)
            conditioned_matrix = conditioned_matrix + multiplier.reshape(-1, 1, 1)

            # In case of tie we need to shift probabilities to lower places. Number of shifts is equal to number of ties
            for n_ties in range(1, k):
                tmp = conditioned_matrix[n_ties, :, :]
                for start_index in range(1, n_ties + 1):
                    conditioned_matrix[n_ties, start_index:, :] = torch.logaddexp(conditioned_matrix[n_ties, start_index:, :],
                                                                                tmp[:-start_index, :])
            # If there were more than k ties we perform cumulative sum, as each tie affects all the following ranks
            conditioned_matrix[k:, :, :] = torch.logcumsumexp(conditioned_matrix[k:, :, :], dim=1)

            # Sum up adjusted probabilities for each score
            result[i, :, :] = torch.logsumexp(conditioned_matrix, dim=0)

        return result

    def rank_dist_with_tiebreaking_batch(self, k, batch_size=1000) -> torch.Tensor:
        lower_score_probability, equal_score_probability, higher_score_probability = self._calculate_score_probabilities()
        result = torch.zeros(self.universe_len, k, self.n_scores).to(self.device)

        for start_index in tqdm(range(0, self.universe_len, batch_size)):
            # The trick here is that for each batch we first calculate probabilities induced by all examples except the batch
            # And only when we iterate over the batch itself. This way we avoid recalculating probabilities for all examples
            effective_batch_size = min(batch_size, self.universe_len - start_index)
            end_index = start_index + effective_batch_size

            # Matrix storing log probabilities of each score being in the first k places with e ties in universe WITHOUT BATCH
            conditioned_matrix = torch.zeros(self.universe_len, k, self.n_scores).to(self.device)  # e, k, l
            conditioned_matrix[0, 0, :] = 1
            conditioned_matrix = torch.log(conditioned_matrix)
            for j in range(self.universe_len):
                if start_index <= j < end_index:
                    # Skip batch examples
                    continue
                # Apply probabilities of example j on conditioned matrix
                conditioned_matrix = self._apply_matrices(conditioned_matrix, lower_score_probability[j], equal_score_probability[j], higher_score_probability[j])

            # Now we create a large matrix storing probabilities of each score of each example IN BATCH being in the first k places with e ties
            big_matrix = torch.zeros(effective_batch_size, self.universe_len, k, self.n_scores).to(self.device)  # i, e, k, l
            # Initiate it using precalculated probabilities for universe without batch
            big_matrix[:] = conditioned_matrix
            # Adjust for probabilities of score assignment
            big_matrix = big_matrix + equal_score_probability[start_index:end_index, :].reshape(effective_batch_size, 1, 1, -1)

            for j in range(effective_batch_size):
                # Apply probabilities of each example in batch on big matrix
                new_matrix = big_matrix + lower_score_probability[start_index+j].reshape(1, 1, 1, -1)
                equal_matrix = big_matrix + equal_score_probability[start_index+j].reshape(1, 1, 1, -1)
                higher_matrix = big_matrix + higher_score_probability[start_index+j].reshape(1, 1, 1, -1)

                new_matrix[:, 1:, :, :] = torch.logaddexp(new_matrix[:, 1:, :, :], equal_matrix[:, :-1, :, :])
                new_matrix[:, :,  1:, :] = torch.logaddexp(new_matrix[:, :, 1:, :], higher_matrix[:, :, :-1, :])
                # Example shouldn't affect itself
                new_matrix[j] = big_matrix[j]
                big_matrix = new_matrix

            # Resolve ties by diving each probability by the number of ties
            multiplier = torch.log(torch.tensor([1 / i for i in range(1, self.universe_len + 1)])).to(self.device)
            big_matrix = big_matrix + multiplier.reshape(1, -1, 1, 1)

            # In case of tie we need to shift probabilities to lower places. Number of shifts is equal to number of ties
            for n_ties in range(1, k):
                tmp = big_matrix[:, n_ties, :, :]
                for shift_index in range(1, n_ties + 1):
                    big_matrix[:, n_ties, shift_index:, :] = torch.logaddexp(big_matrix[:, n_ties, shift_index:, :],
                                                                              tmp[:, :-shift_index, :])
            # If there were more than k ties we perform cumulative sum, as each tie affects all the following ranks
            big_matrix[:, k:, :, :] = torch.logcumsumexp(big_matrix[:, k:, :, :], dim=2)
            # Sum up adjusted probabilities for each score of each example in batch
            result[start_index:start_index+effective_batch_size] = torch.logsumexp(big_matrix, dim=1)
        return result

    def rank_dist(self, k) -> torch.Tensor:
        result = torch.zeros(self.universe_len, k, self.n_scores).to(self.device)
        result[:, 0, :] = self.score_probabilities  # i, k, l
        result = torch.log(result)

        reshaped_score_values = self.score_values.reshape(-1, 1, self.n_scores)
        for j, j_score_values in enumerate(self.score_values):
            lower_score_mask = (j_score_values.reshape(-1, 1) < reshaped_score_values).float() # 3d array (i, l_j, l_i). Entry i,l_j,l_i defines if l_i'th score of i larger than l_j'th score of j
            lower_score_mask += (j_score_values.reshape(-1, 1) == reshaped_score_values)*0.5
            p_lower_score = (lower_score_mask*self.score_probabilities[j].reshape(1, self.n_scores, 1)).sum(dim=1)  # 2d array (i,l). P(s_j < s_i^l)
            p_lower_score = torch.clamp(p_lower_score, max=1)
            new_result = result + torch.log(p_lower_score)[:, None, :]  # p_{i,j,k,l} = p(j_smaller)*p_{i,j-1,k,l} +
            new_result[:, 1:, :] = torch.logaddexp(new_result[:, 1:, :],
                                                   (result + torch.log(1 - p_lower_score)[:, None, :])[:, :-1, :])  # p(j_larger)*p_{i,j-1,k-1,l}
            new_result[j] = result[j]  # j'th probabilities shouldn't change if j is already in the sub-universe

            if torch.any(torch.isnan(new_result)):
                logging.critical('ERROR WITH PROBABILITY IN ILLEGAL RANGE. RESULTS MAY BE FLAWED. PLEASE REPORT TO ALEX')

            result = new_result

        return result

    def _calculate_score_probabilities(self):
        """
        Calculates log probabilities for each example in the universe being lower/equal/higher to each score
        :return: 3 matrices of shape (universe_len, n_scores) containing log probabilities
        """
        lower_score_probability = torch.zeros((self.universe_len, self.n_scores)).to(self.device)
        lower_score_probability[:, 1:] = self.score_probabilities.cumsum(dim=1)[:, :-1]  # P(S_j < l)
        equal_score_probability = torch.log(self.score_probabilities).to(self.device)
        higher_score_probability = torch.ones((self.universe_len, self.n_scores)).to(
            self.device) - self.score_probabilities - lower_score_probability
        higher_score_probability = torch.log(torch.clamp(higher_score_probability, min=0, max=1)).to(self.device)
        lower_score_probability = torch.log(lower_score_probability).to(self.device)
        return lower_score_probability, equal_score_probability, higher_score_probability

    @staticmethod
    def _apply_matrices(conditioned_matrix, lower_score_probability, equal_score_probability, higher_score_probability):
        new_matrix = conditioned_matrix + lower_score_probability.reshape(1, 1, -1)
        equal_matrix = conditioned_matrix + equal_score_probability.reshape(1, 1, -1)
        higher_matrix = conditioned_matrix + higher_score_probability.reshape(1, 1, -1)

        new_matrix[1:, :, :] = torch.logaddexp(new_matrix[1:, :, :], equal_matrix[:-1, :, :])
        new_matrix[:, 1:, :] = torch.logaddexp(new_matrix[:, 1:, :], higher_matrix[:, :-1, :])
        return new_matrix


def performance_test_for_different_scores():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = 10000
    n_scores = 10
    k = 5
    for i in range(2):
        score_probabilities, score_values = generate_universe(size, n_scores)
        rank_dist = RankDist(score_probabilities, score_values, device)
        start = time.time()
        result = rank_dist.rank_dist(k)
        print(f'k={k}. n_scores={n_scores}. universe_size={size}. Took {time.time() - start} seconds')



def run_small_test():
    device = 'cpu'
    score_probabilities = torch.from_numpy(np.array([[0.4, 0.6],
                     [0.3, 0.7],
                     [0.2, 0.8]]))
    score_values = torch.from_numpy(np.array([[5, 10],
                     [2, 7],
                     [3, 8]]))
    rank_dist = RankDist(score_probabilities, score_values, device)
    result = rank_dist.rank_dist(2)
    print(result)
    print(torch.exp(result))


def run_small_test2():
    device = 'cpu'
    k = 2
    data = torch.Tensor([[0.0, 0.4, 0.6, 0.0],
                     [0.3, 0.0, 0.0, 0.7],
                     [0.1, 0.0, 0.4, 0.5]])
    data2 = torch.Tensor([[0.5,  0.5],
                     [0.5, 0.5],
                     [0.5, 0.5]])

    rank_dist = RankDist(data, device=device)
    m1 = rank_dist.rank_dist_with_tiebreaking_one_by_one(k)
    # m3 = rank_dist.rank_dist_with_tiebreaking_batch(k)
    print(torch.exp(m1)[0])
    # print(torch.exp(m3)[0])


def run_small_test3():
    device = 'cpu'
    score_probabilities = torch.from_numpy(np.array([[0.4, 0.6, 0.0],
                     [0.2, 0.8, 0.0],
                     [0.1, 0.4, 0.5]]))
    score_values = torch.from_numpy(np.array([[2, 3, 10],
                     [1, 4, 7],
                     [0.5, 2.5, 5]]))
    rank_dist = RankDist(score_probabilities, score_values, device)
    result = rank_dist.rank_dist(3)
    print(result)
    print(torch.exp(result))


def tiebreaking_sanity_test():
    top_k_size = 3
    universe_size = 100
    n_scores = 5
    score_probabilities, _ = generate_universe(universe_size, n_scores)
    rank_dist = RankDist(score_probabilities, device='cuda')
    m1 = rank_dist.rank_dist_with_tiebreaking_one_by_one(top_k_size)
    #print(torch.exp(m1)[0])
    print(m1[-1])
    m3 = rank_dist.rank_dist_with_tiebreaking_batch(top_k_size, 20)
    #print(torch.exp(m3)[0])
    print(m3[-1])
    print((m3-m1).sum())


def tiebreaking_performance_test():
    top_k_size = 10
    universe_size = 10000
    n_scores = 5
    score_probabilities, _ = generate_universe(universe_size, n_scores)
    rank_dist = RankDist(score_probabilities, device='cuda')

    for batch_size in [75, 100, 125]:
        start = time.time()
        m3 = rank_dist.rank_dist_with_tiebreaking_batch(top_k_size, batch_size=batch_size)
        diff = time.time() - start
        print('batch_size', batch_size, 'time', diff)


def rank_dist_runtime_test(scores_num=10, uni_options=[1000,5000,10000], k_options=[10,50,100],
                           min_uni_size=1000, max_uni_size=10000, uni_size_jump=1000,
                           min_k=10, max_k=20, k_jump=10, num_repeats=10,
                           rank_dist_type='tie_breaking'):
    combinations = []
    if k_options is not None:
        for top_k_size in k_options:
            for uni_size in range(min_uni_size, max_uni_size + uni_size_jump, uni_size_jump):
                combinations.append((uni_size, top_k_size))
    if uni_options is not None:
        for uni_size in uni_options:
            for top_k_size in range(min_k, max_k + k_jump, k_jump):
                if (uni_size, top_k_size) not in combinations:
                    combinations.append((uni_size, top_k_size))
    if k_options is None and uni_options is None:
        top_k_size = min_k
        for uni_size in range(min_uni_size, max_uni_size + uni_size_jump, uni_size_jump):
            combinations.append((uni_size, top_k_size))
        uni_size = min_uni_size
        for top_k_size in range(min_k, max_k + k_jump, k_jump):
            combinations.append((uni_size, top_k_size))

    results = {}
    for uni_size, top_k_size in combinations:
        print("*"*20)
        print(f'Starting testing with uni_size={uni_size}, top_k_size={top_k_size}')
        total_time = 0
        for i in range(num_repeats):
            score_probabilities, score_values = generate_universe(uni_size, 10)
            start = time.time()
            if rank_dist_type == 'tie_breaking':
                rank_dist = RankDist(score_probabilities=score_probabilities, device=device)
                batch_size = math.floor(math.sqrt(uni_size))
                result = rank_dist.rank_dist_with_tiebreaking_batch(k=top_k_size, batch_size=batch_size)
            elif rank_dist_type == 'no_tie_breaking':
                rank_dist = RankDist(score_probabilities=score_probabilities,
                                     score_values=score_values, device=device)
                result = rank_dist.rank_dist(top_k_size)
            else:
                raise ValueError(f'rank_dist_type={rank_dist_type} is invalid')
            iter_time = time.time() - start
            print(f'Iteration {i+1} took {iter_time} seconds')
            total_time += iter_time
        avg_runtime = round(total_time / num_repeats, 4)
        results[(uni_size, top_k_size)] = avg_runtime
        print(f'uni_size={uni_size}, top_k_size={top_k_size}. Took {avg_runtime} seconds on average')

    # Write results to CSV
    with open(f'./rankdist_{rank_dist_type}_scores_num_{scores_num}_{num_repeats}_repeats_runtime',
              'w', newline='') as csvfile:
        fieldnames = ['uni_size', 'top_k_size', 'avg_runtime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (uni_size, top_k_size), avg_runtime in results.items():
            writer.writerow({'uni_size': uni_size, 'top_k_size': top_k_size,
                             'avg_runtime': avg_runtime})


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    # run_small_test3()

    # print("Running RankDist runtime test, no_tie_breaking version (deterministic tie-breaking)")
    # rank_dist_runtime_test(scores_num=10, uni_options=[1000, 5000, 10000], k_options=[10,50,100],
    #                        min_uni_size=1000, max_uni_size=10000, uni_size_jump=1000,
    #                        min_k=10, max_k=100, k_jump=10, num_repeats=10,
    #                        rank_dist_type='no_tie_breaking')

