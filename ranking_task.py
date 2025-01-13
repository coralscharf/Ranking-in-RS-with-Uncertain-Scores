import math
from expected_precision_calculator import *
from constants import *


class UserRankingTask:
    def __init__(self, user_id, train_val_item_ids, user_test_data, rec_items_by_preds,
                 item_score_distributions, score_values, top_k_size_options,
                 unc_thresholds_by_percentiles=None, item_means=None, item_stds=None):
        self.user_id = user_id
        self.train_val_item_ids = train_val_item_ids
        self.user_test_data = user_test_data
        self.scores_by_test_items = {
            int(item_with_score[0]): item_with_score[1]
            for item_with_score in user_test_data
        }
        self.test_item_ids = torch.tensor(user_test_data[user_test_data[:, 1] >= 4, 0]).int()
        self.test_item_ids_as_list = self.test_item_ids.tolist()
        user_test_data_sorted = user_test_data[user_test_data[:, 1].argsort()]
        self.test_item_ids_grouped = np.split(user_test_data_sorted[:, 0],
                                      np.unique(user_test_data_sorted[:, 1],
                                                return_index=True)[1][1:])
        self.test_item_ids_grouped.reverse()

        self.score_values = score_values

        self.rec_items_by_preds = rec_items_by_preds
        self.item_score_distributions = item_score_distributions
        self.item_ids = torch.tensor(range(len(self.item_score_distributions))).to(device)
        self.item_means = item_means
        self.item_stds = item_stds
        self.item_score_values = None
        self.item_idxs_to_item_ids = None
        self.item_ids_to_item_idxs = None

        self.unc_thresholds_by_percentiles = unc_thresholds_by_percentiles
        self.top_k_size_options = top_k_size_options
        self.top_k_max_size = max(top_k_size_options)

        self.rank_dist_log_result = None
        self.top_k_results = dict()

    def compute_item_expected_scores(self):
        if self.item_means is None:
            return torch.sum(self.item_score_distributions * self.score_values, dim=1)
        else:
            return torch.tensor(self.item_means)

    def compute_item_prr_scores(self):
        if self.item_means is None:
            return torch.sum(self.item_score_distributions[:, self.score_values >= PRR_THRESHOLD], dim=1)
        else:
            return torch.tensor(norm.sf(PRR_THRESHOLD, self.item_means, self.item_stds))

    def get_item_ids_after_ubf(self, percentile, result_size):
        if percentile == 1.0:
            return self.rec_items_by_preds[:result_size].index.tolist()
        else:
            unc_threshold = self.unc_thresholds_by_percentiles[
                np.where(PERCENTILES_FOR_UBF == percentile)[0].item()]
            item_uncertainties_rec_by_preds = self.rec_items_by_preds.uncertainties.to_numpy()
            return self.rec_items_by_preds[item_uncertainties_rec_by_preds <
                                           unc_threshold][:result_size].index.tolist()

    def generate_candidate_items_set(self, add_prob_to_score, candidate_set_size, candidate_set_selection_method):
        mask = torch.ones(len(self.item_ids), dtype=torch.bool)
        mask[self.train_val_item_ids] = False
        self.item_score_distributions = self.item_score_distributions[mask]
        self.item_ids = self.item_ids[mask]
        if self.item_means is not None:
            self.item_means = self.item_means[mask]
            self.item_stds = self.item_stds[mask]

        if candidate_set_size is not None or candidate_set_selection_method is not None:
            if candidate_set_selection_method.startswith("reranking"):
                reranking_method = candidate_set_selection_method.split("reranking-")[1]
                if reranking_method.startswith("uncertainty_based_filtering"):
                    percentile = float(reranking_method.split("-")[1])
                    target_item_ids = torch.tensor(
                        self.get_item_ids_after_ubf(percentile, candidate_set_size)
                    ).to(device)
                    target_item_ids_indices = torch.nonzero(self.item_ids.unsqueeze(1) ==
                                                            target_item_ids.unsqueeze(0))[:, 0]
                else:
                    if reranking_method == "probability_of_relevance_ranking":
                        items_rr_scores = self.compute_item_prr_scores()
                    elif reranking_method == "expected_score":
                        items_rr_scores = self.compute_item_expected_scores()
                    else:
                        raise ValueError(f'Invalid reranking method')
                    target_item_ids_indices = torch.argsort(items_rr_scores, descending=True)[:candidate_set_size]

            elif candidate_set_selection_method == "leave-test-out":
                if candidate_set_size != 0:
                    mask = torch.ones(len(self.item_ids), dtype=torch.bool)
                    test_items_indices_in_item_ids = torch.nonzero(self.item_ids.unsqueeze(1) ==
                                                                   self.test_item_ids.unsqueeze(0))[:, 0]
                    mask[test_items_indices_in_item_ids] = False
                    not_test_item_ids = self.item_ids[mask]
                    not_test_indices_samples = torch.multinomial(not_test_item_ids.float(),
                                                                 num_samples=candidate_set_size - len(self.test_item_ids),
                                                                 replacement=False)
                    target_item_ids = torch.cat((not_test_item_ids[not_test_indices_samples], self.test_item_ids)).int()
                else:
                    target_item_ids = self.test_item_ids

                target_item_ids_indices = torch.nonzero(self.item_ids.unsqueeze(1) ==
                                                        target_item_ids.unsqueeze(0))[:, 0]
            else:
                raise ValueError(f'{candidate_set_selection_method} for candidate set selection method is invalid')

            self.item_score_distributions = self.item_score_distributions[target_item_ids_indices]
            self.item_ids = self.item_ids[target_item_ids_indices]
            if self.item_means is not None:
                self.item_means = self.item_means[target_item_ids_indices.detach().cpu().numpy()]
                self.item_stds = self.item_stds[target_item_ids_indices.detach().cpu().numpy()]

            self.rec_items_by_preds = self.rec_items_by_preds[
                self.rec_items_by_preds.index.isin(set(self.item_ids.tolist()))
            ]

        self.item_idxs_to_item_ids = {item_idx: item_id.item() for item_idx, item_id in enumerate(self.item_ids)}
        self.item_ids_to_item_idxs = {item_id.item(): item_idx for item_idx, item_id in enumerate(self.item_ids)}

        self.item_score_values = self.score_values.repeat(len(self.item_score_distributions), 1).to(device)
        if add_prob_to_score:
            self.item_score_values = self.item_score_values + self.item_score_distributions

    def run_rank_dist(self, scores_type, rank_dist_results_path):
        rank_dist = RankDist(self.item_score_distributions, self.item_score_values, device=device)
        if scores_type == "equal_scores":
            batch_size = math.floor(math.sqrt(len(self.item_score_distributions)))
            self.rank_dist_log_result = rank_dist.rank_dist_with_tiebreaking_batch(k=self.top_k_max_size,
                                                                                   batch_size=batch_size).cpu().detach().numpy()
        else:
            self.rank_dist_log_result = rank_dist.rank_dist(k=self.top_k_max_size).cpu().detach().numpy()

        if rank_dist_results_path is not None:
            np.save(f'{rank_dist_results_path}rank_dist_result_array_user_{self.user_id}.npy', self.rank_dist_log_result)

        self.rank_dist_log_result = np.logaddexp.reduce(self.rank_dist_log_result, axis=2)

    def generate_top_k_answers(self):
        for method in TOP_K_APPROACHES:
            if method == "expected_score":
                item_scores = self.compute_item_expected_scores()
                self.top_k_results[method] = torch.argsort(item_scores, descending=True)[:self.top_k_max_size].tolist()
            elif method == "global_top_k":
                self.top_k_results[method] = dict()
                for top_k_size in self.top_k_size_options:
                    rank_dist_result_at_k = self.rank_dist_log_result[:, :top_k_size]
                    tuples_prob_until_k = np.logaddexp.reduce(rank_dist_result_at_k, axis=1)
                    self.top_k_results[method][top_k_size] = np.argsort(tuples_prob_until_k)[::-1][:top_k_size].tolist()
            elif method == "probability_of_relevance_ranking":
                item_scores = self.compute_item_prr_scores()
                self.top_k_results[method] = torch.argsort(item_scores, descending=True)[:self.top_k_max_size].tolist()
            elif method.startswith("uncertainty_based_filtering"):
                percentile = float(method.split("-")[1])
                top_k_item_ids = self.get_item_ids_after_ubf(percentile, self.top_k_max_size)
                self.top_k_results[method] = [self.item_ids_to_item_idxs[item_id]
                                              for item_id in top_k_item_ids]
            elif method == "U-k-ranks":
                self.top_k_results[method] = np.argmax(self.rank_dist_log_result, axis=0)
            # elif method == "PT-k":
            #     if "threshold" not in kwargs:
            #         raise ValueError(f'Threshold was not specified for the PT-k approach')
            #     threshold = kwargs["threshold"]
            #     top_k_probabilities_summation = np.sum(rank_dist_result_at_k, axis=1)
            #     returned_tuples = (top_k_probabilities_summation >= threshold).nonzero()[0]  # Without order importance
            #     top_k_result_tuples_idx = np.argsort(top_k_probabilities_summation[returned_tuples])[::-1]
            #     return returned_tuples[top_k_result_tuples_idx]
            # elif method == "expected_rank":
            #     ranks = np.arange(top_k_size) + 1
            #     tuples_expected_rank = np.sum(rank_dist_result_at_k * ranks, axis=1)
            #     return np.argsort(tuples_expected_rank)[::-1][:top_k_size]
            else:
                raise ValueError(f'The method {method} for generating top-K answer does not exist')

    def get_top_k_result_by_method(self, method, top_k_size):
        if type(self.top_k_results[method]) is not dict:
            top_k_res_item_idxs = self.top_k_results[method][:top_k_size]
        else:
            top_k_res_item_idxs = self.top_k_results[method][top_k_size]

        top_k_res_item_ids = [self.item_idxs_to_item_ids[item_idx]
                              for item_idx in top_k_res_item_idxs]
        return top_k_res_item_ids, top_k_res_item_idxs

    def compute_exact_metrics(self, top_k_result):
        metric_results = dict()
        metric_results['precision_r_all'] = compute_precision(top_k_result,
                                                              self.test_item_ids_as_list,
                                                              self.test_item_ids_grouped,
                                                              reference="all_test_items")
        metric_results['precision_r_k_ranked'] = compute_precision(top_k_result,
                                                                   self.test_item_ids_as_list,
                                                                   self.test_item_ids_grouped,
                                                                   reference="k_ranked_test_items")

        metric_results['dcg_l'] = compute_dcg_for_user(self.scores_by_test_items,
                                                       self.test_item_ids_grouped,
                                                       top_k_result,
                                                       dcg_version='liberal')
        metric_results['dcg_c'] = compute_dcg_for_user(self.scores_by_test_items,
                                                       self.test_item_ids_grouped,
                                                       top_k_result,
                                                       dcg_version='conservative')

        metric_results['recall'] = compute_recall(top_k_result, self.test_item_ids_as_list)
        return metric_results

