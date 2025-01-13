from typing import List
import numpy as np
import torch
from rank_dist import RankDist
from consecutive_pair_rank import ConsecutivePairRank
from pairwise_rank import PairwiseRank
from utils import *
import time
from itertools import product
import logging


class PrecisionCalculator:
    def __init__(self, universe_score_probabilities: torch.Tensor, universe_score_values: torch.Tensor, device='cpu'):
        self.score_probabilities = universe_score_probabilities.float().to(device)
        self.score_probabilities = torch.nn.functional.normalize(self.score_probabilities, dim=1, p=1)
        self.score_values = universe_score_values.float().to(device)
        self.universe_len = self.score_probabilities.shape[0]

        self.candidate_indices = []
        self.candidate_mask = torch.zeros(self.universe_len, dtype=torch.bool).to(device)
        self.non_candidate_mask = torch.zeros(self.universe_len, dtype=torch.bool).to(device)
        self.k = 0
        self.calculate_mode = 'torch'
        self.device = device

        self.pairwise_rank_calculator = PairwiseRank([], 0, device)  # will be reused for update mode

        self.all_universes = torch.Tensor([])  # will be used for enumeration method
        self.universe_probabilities = torch.Tensor([])  # will be used for enumeration method

    def calculate_expected_precision(self, candidate_indices: List[int], calculate_mode='torch', update_mode=False):
        if update_mode:
            if self.k == 0 or self.candidate_indices == []:
                raise ValueError('update mode can be used only after initial run')
            removed_indices = [x for x in self.candidate_indices if x not in candidate_indices]
            if len(removed_indices) == 0:
                raise ValueError('Candidates in update mode should be a subset of original candidates')

        self.candidate_indices = candidate_indices
        self.k = len(self.candidate_indices)
        self.candidate_mask[:] = False
        self.candidate_mask[self.candidate_indices] = True
        self.non_candidate_mask = ~self.candidate_mask
        self.calculate_mode = calculate_mode

        if calculate_mode == 'enumeration':
            return self._calculate_precision_with_enumeration()

        start = time.time()
        hit_log_probability = np.log(np.zeros(self.k+1))
        log_p_at_k = self._calculate_precision_at_k()
        hit_log_probability[self.k] = log_p_at_k
        logging.info(f'Log P@K={log_p_at_k:.5f}. P@K={np.exp(log_p_at_k)}. Calculated in {time.time()-start:.1f} seconds')

        start2 = time.time()
        consecutive_calculator = ConsecutivePairRank(self.score_probabilities[self.candidate_mask],
                                                     self.score_values[self.candidate_mask], self.device)
        consecutive_probabilities_dict = consecutive_calculator.calculate_consecutive_rank()
        # Dict mapping from [m, n, s_m, s_n] to vector P(R(s_m =^A i), R(s_n =^A i+1)
        logging.info(f'Calculated consecutive probabilities for candidate set in {time.time()-start2:.1f} seconds')

        start3 = time.time()
        all_possible_tuples = list(set([(k[2], k[3]) for k in consecutive_probabilities_dict.keys()]))
        tuple_index_dict = {tup: i for i, tup in enumerate(all_possible_tuples)}
        if update_mode:
            self.pairwise_rank_calculator.update_result(self.score_probabilities[removed_indices], self.score_values[removed_indices])
            logging.info(f'Updated pairwise rank probabilities for {len(all_possible_tuples)} tuples in {time.time() - start3:.1f} seconds')
        else:
            self.pairwise_rank_calculator = PairwiseRank(all_possible_tuples, self.k, self.device)
            self.pairwise_rank_calculator.calculate_pairwise_rank(self.score_probabilities[self.non_candidate_mask],
                                                                  self.score_values[self.non_candidate_mask])
            # Array (n_tuples, rank_s_m, rank_s_n). For each tuple (s_m,s_n) we have P(R(s_m)=i, R(s_n)=j) in universe U-A+(s_m, s_n)
            logging.info(f'Calculated pairwise rank probabilities for {len(all_possible_tuples)} tuples in {time.time() - start3:.1f} seconds')

        start4 = time.time()
        for (_, __, s_m, s_n), probability_vector in consecutive_probabilities_dict.items():
            index_of_tuple = tuple_index_dict[(s_m, s_n)]
            pairwise_rank_probability_matrix = self.pairwise_rank_calculator.result[index_of_tuple].cpu().numpy()
            for i, log_probability_of_tuple_at_i in enumerate(probability_vector):
                if log_probability_of_tuple_at_i == -float('inf'):
                    continue
                # now we need to extract P(R(s_m) <= k-i+1, R(s_n) > k-i+1) (+1 dissapears due to index shift from 0 to 1)
                borderline_log_probability = np.logaddexp.reduce(pairwise_rank_probability_matrix[:self.k-i, self.k-i:], axis=None)
                precision_from_tuple = log_probability_of_tuple_at_i + borderline_log_probability
                hit_log_probability[i+1] = np.logaddexp(hit_log_probability[i+1], precision_from_tuple)  # i+1 cause real ranks begin at 1 and not at 0
        logging.info(f'Calculated precisions in {time.time() - start4:.1f} seconds')

        multiplication_coefficient = np.array([i/self.k for i in range(self.k+1)])
        log_expected_precision = np.logaddexp.reduce(hit_log_probability[1:] + np.log(multiplication_coefficient[1:]))
        logging.info(f'Done in {time.time() - start:.1f} seconds. Log of expected precision: {log_expected_precision:.5f}. EP: {np.exp(log_expected_precision)}')
        logging.debug(f'Log precisions: {hit_log_probability[1:]}. Precisions: {np.exp(hit_log_probability[1:])}')
        logging.info('###############################################################################################')
        return hit_log_probability, log_expected_precision

    def calculate_mean_expected_precision(self, candidate_indices: List[int], calculate_mode='torch'):
        if calculate_mode == 'enumeration':
            raise ValueError('Enumeration unsupported at the moment')
        log_expected_precisions = torch.log(torch.zeros(len(candidate_indices)+1))
        hit_log_probabilities = [[] for i in range(len(candidate_indices)+1)]
        logging.info(f'\t\t\t\t############ Beginning Mean EP calculation. Starting with the full list #########')
        full_hit_log_probabilities, full_expected_precision = self.calculate_expected_precision(candidate_indices, calculate_mode)

        log_expected_precisions[self.k] = full_expected_precision
        hit_log_probabilities[self.k] = full_hit_log_probabilities
        for candidate_length in range(self.k-1, 0, -1):
            logging.info(f'\t\t\t\t############ Computing precision for {candidate_length} candidates #########')
            hit_log_probability, log_expected_precision = self.calculate_expected_precision(candidate_indices[:candidate_length], calculate_mode, True)
            log_expected_precisions[candidate_length] = log_expected_precision
            hit_log_probabilities[candidate_length] = hit_log_probability
        log_mean_EP = np.logaddexp.reduce(log_expected_precisions[1:] + np.log(1/len(candidate_indices)))
        logging.info(f'\t\t\t\t ############################################## ')
        logging.info(f'Calculated expected precisions. Log of mean EP={log_mean_EP:.5f}. Mean EP={np.exp(log_mean_EP)}')
        return log_mean_EP, log_expected_precisions[1:], hit_log_probabilities[1:]

    def _calculate_precision_at_k(self):
        distribution_calculator = RankDist(self.score_probabilities[self.candidate_indices],
                                           self.score_values[self.candidate_indices], self.device)
        rank_distribution_in_candidate_set = distribution_calculator.rank_dist(self.k)  # Tensor (index, k, score)

        possible_scores = self.score_values[self.candidate_indices].unique()
        log_probability_of_being_first = calculate_log_probability_of_being_first(self.score_probabilities[self.non_candidate_mask],
                                                                              self.score_values[self.non_candidate_mask],
                                                                              possible_scores, self.device)

        log_precision = torch.log(torch.zeros(1)).to(self.device)
        for i, candidate_index in enumerate(self.candidate_indices):
            for score_index, candidate_score in enumerate(self.score_values[candidate_index]):
                log_borderline_probability = rank_distribution_in_candidate_set[i, self.k-1, score_index] + \
                                             log_probability_of_being_first[candidate_score.item()]
                log_precision = torch.logaddexp(log_precision, log_borderline_probability)
        return log_precision.item()

    def _calculate_precision_with_enumeration(self):
        n_values = self.score_values.shape[1]
        self.all_universes = torch.zeros((n_values**self.universe_len, self.universe_len)).to(self.device)
        self.universe_probabilities = torch.ones(n_values**self.universe_len).to(self.device)
        possible_indices = list(range(n_values))
        precisions = torch.zeros(self.k+1)
        candidate_set = set(self.candidate_indices)

        start = time.time()
        for universe_index, score_indices in enumerate(product(possible_indices, repeat=self.universe_len)):
            for item_index, score_index in enumerate(score_indices):
                self.all_universes[universe_index, item_index] = self.score_values[item_index, score_index]
                self.universe_probabilities[universe_index] *= self.score_probabilities[item_index, score_index]
        print(f'Enumerated universes in {time.time() - start:.1f} seconds')

        # TODO: consider implementing tie-breaking
        start2 = time.time()
        universe_topk_indices = torch.argsort(self.all_universes, dim=1, descending=True)
        for topk_indices, universe_probability in zip(universe_topk_indices, self.universe_probabilities):
            overlap = len(candidate_set.intersection(set(topk_indices[:self.k].tolist())))
            precisions[overlap] += universe_probability
        print(f'Calculated precisions in {time.time() - start2:.1f} seconds')

        expected_precision = sum([i/self.k*precision for i, precision in enumerate(precisions)])
        print(f'Done in {time.time() - start:.1f} seconds. Expected precision: {expected_precision:.5f}')
        print(f'Precisions: {precisions[1:]}')
        return precisions, expected_precision


def run_small_test():
    device = 'cpu'
    score_probabilities = torch.from_numpy(np.array([[0.4, 0.5, 0.1],
                                    [0.1, 0.5, 0.4],
                                    [0.3, 0.3, 0.4],
                                    [0.2, 0.3, 0.5]]))
    score_values = torch.from_numpy(np.array([[2, 5, 7],
                             [0, 6, 9],
                             [1, 3, 4],
                             [-1, 8, 10]]))
    calculator = PrecisionCalculator(score_probabilities, score_values, device)
    calculator.calculate_expected_precision([1, 2])
    calculator.calculate_expected_precision([1])
    calculator.calculate_mean_expected_precision([1, 2])
    #calculator.calculate_expected_precision([1, 2], 'enumeration')


def performance_test():
    calculate_mode = 'torch'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = 30000
    n_scores = 5
    n_candidates = 20
    for n_scores in [10]:
        print(f'{n_scores} scores. {size} universe size. {n_candidates} candidates')

        score_probabilities, score_values = generate_universe(size, n_scores)
        calculator = PrecisionCalculator(score_probabilities, score_values, device)
        calculator.calculate_expected_precision(list(range(n_candidates)), calculate_mode)
        #calculator.calculate_mean_expected_precision(list(range(n_candidates)), calculate_mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # run_small_test()
    performance_test()
