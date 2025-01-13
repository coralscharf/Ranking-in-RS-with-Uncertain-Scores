from enum import Enum
import pandas as pd
from scipy.stats import norm, truncnorm
import pickle
import csv
from probability_computation.uncertain_SnapExplicit.uncertain.explicit import *
from constants import *


class Distribution(Enum):
    UNIFORM = 'uniform'


def generate_universe(universe_size=10, n_scores=3, distribution=Distribution.UNIFORM,
                      random_state=0, *args, **kwargs):
    if distribution == Distribution.UNIFORM:
        return _generate_uniform(universe_size, n_scores, random_state)


def _generate_uniform(universe_size, n_scores, random_state):
    old_seed = torch.seed()
    torch.manual_seed(random_state)
    score_probabilities = torch.rand(universe_size * n_scores, dtype=torch.double).reshape(universe_size, -1)
    score_probabilities = score_probabilities / (score_probabilities.sum(axis=1).reshape(-1, 1))
    score_values = torch.rand(universe_size * n_scores, dtype=torch.double).reshape(universe_size, -1) * 100
    torch.manual_seed(old_seed)
    return score_probabilities, score_values


def load_first_stage_model_and_data(fs_models_and_data_path, first_stage_model_name):

    # Load Data
    with open(f'{fs_models_and_data_path}data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'Data prepared: {data.n_user} users, {data.n_item} items.')
    print(f'{len(data.train)} train, {len(data.val)} validation and {len(data.test)} test interactions.')

    score_labels = pd.factorize(data.train[:, 2], sort=True)[1]

    trained_models_path = f"{fs_models_and_data_path}/checkpoints/{first_stage_model_name.lower()}/"
    files = {file: float(file.split('val_loss=')[1][:-5]) for file in os.listdir(trained_models_path)}

    if first_stage_model_name == "BeMF":
        first_stage_model = BeMF(data.n_user, data.n_item, score_labels=score_labels, embedding_dim=20)
        first_stage_model = first_stage_model.load_from_checkpoint(os.path.join(trained_models_path,
                                                                                min(files, key=files.get)))
    elif first_stage_model_name == "OrdRec":
        first_stage_model = OrdRec(data.n_user, data.n_item, score_labels=score_labels, embedding_dim=0)
        first_stage_model = first_stage_model.load_from_checkpoint(os.path.join(trained_models_path,
                                                                                min(files, key=files.get)))
    elif first_stage_model_name == "CPMF":
        first_stage_model = CPMF(data.n_user, data.n_item, embedding_dim=0, lr=0, weight_decay=0)
        first_stage_model = first_stage_model.load_from_checkpoint(os.path.join(trained_models_path,
                                                                                min(files, key=files.get)))
    else:
        raise ValueError(f'{first_stage_model_name} model for first stage is invalid')
    first_stage_model.to(device)
    return data, first_stage_model


def create_items_discrete_distribution_from_normal_distribution(item_means,
                                                                item_stds,
                                                                score_values,
                                                                truncate_distribution=True):
    if truncate_distribution:
        min_score, max_score = min(score_values), max(score_values)
        items_a_values = (min_score - item_means[:, np.newaxis]) / item_stds[:, np.newaxis]
        items_b_values = (max_score - item_means[:, np.newaxis]) / item_stds[:, np.newaxis]
        scores_cdf_values = truncnorm.cdf(score_values, items_a_values, items_b_values,
                                          loc=item_means[:, np.newaxis],
                                          scale=item_stds[:, np.newaxis])
        score_probabilities = np.diff(scores_cdf_values)
    else:
        scores_cdf_values = norm.cdf(score_values,
                                     loc=item_means[:, np.newaxis],
                                     scale=item_stds[:, np.newaxis])
        score_probabilities = np.diff(scores_cdf_values)
        score_probabilities = np.concatenate((scores_cdf_values[:, 0][:, np.newaxis],
                                              score_probabilities), axis=1)
        score_probabilities = np.concatenate((score_probabilities,
                                              (1 - scores_cdf_values[:, -1])[:, np.newaxis]), axis=1)
    return torch.tensor(score_probabilities)


def calculate_log_probability_of_being_first(universe_score_probabilities: torch.Tensor,
                                             universe_score_values: torch.Tensor,
                                             scores, device='cpu'):
    """
    Calculates probability of each score from a given list being the first in a universe
    :param universe_score_probabilities: array of universe score probabilities (shape |U|*n_scores)
    :param universe_score_values: array of universe score values (shape |U|*n_scores)
    :param scores: list of candidate scores
    :return: dict {score:log probability}
    """
    log_probabilities = torch.zeros(len(scores)).to(device)
    for score_probabilities, score_values in zip(universe_score_probabilities, universe_score_values):
        lower_mask = score_values.reshape(-1, 1) < scores
        equal_mask = score_values.reshape(-1, 1) == scores
        lower_probabilities = score_probabilities[None, :].mm(lower_mask.float())
        lower_probabilities += score_probabilities[None, :].mm(equal_mask.float())/2 # random tie-breaking

        log_probabilities += torch.log(lower_probabilities[0])

    log_probabilities_dict = {score.item(): probability.item() for score, probability in zip(scores, log_probabilities)}
    return log_probabilities_dict


def check_if_item_is_in_cons_list(top_k_result, ground_truth_items_grouped):
    top_k_items_in_list = [0 for _ in top_k_result]
    items_to_pass = len(top_k_result)
    for gt_items_by_score in ground_truth_items_grouped:
        if items_to_pass <= 0:
            break
        set_gt_items_by_score = set(gt_items_by_score.astype(int))
        common_items = set(top_k_result).intersection(set_gt_items_by_score)
        # Mark the items in the common_items which are in the top_k_result, at most items_to_pass items.
        # Priority is given to the items which are ranked higher
        for item_idx, item in enumerate(top_k_result):
            if items_to_pass <= 0:
                break
            if item in common_items:
                top_k_items_in_list[item_idx] = 1
                items_to_pass -= 1
    return top_k_items_in_list


def compute_dcg_for_user(user_scores_by_test_items, user_test_item_ids_grouped,
                         top_k_result, dcg_version='liberal',
                         ideal_dcg=False, user_test_scores_ranked=None):
    test_items_scores = user_scores_by_test_items  # [user_id]
    test_scores_ranked = user_test_scores_ranked  # [user_id]
    if ideal_dcg:
        # user_scores = sorted(list(test_items_scores.values()), reverse=True)[:len(top_k_result)]
        # user_scores = heapq.nlargest(len(top_k_result), test_items_scores.values())
        user_scores = test_scores_ranked[:len(top_k_result)]
    else:
        if dcg_version == 'conservative':
            top_k_items_in_list = check_if_item_is_in_cons_list(top_k_result,
                                                                user_test_item_ids_grouped)
            user_scores = [test_items_scores[item]
                           if (is_valid == 1 and item in test_items_scores) else 0
                           for item, is_valid in zip(top_k_result, top_k_items_in_list)]
        else:  # dcg_version == 'liberal'
            user_scores = [test_items_scores[item] if item in test_items_scores else 0
                           for item in top_k_result]

    return np.sum([(2 ** score - 1) / np.log2(rank + 2)
                   for rank, score in enumerate(user_scores)])


def compute_precision(top_k_result, ground_truth_items, ground_truth_items_grouped,
                      reference="all_test_items"):
    ground_truth_items = set(ground_truth_items)
    top_k_result = set(top_k_result)
    k = len(top_k_result)
    if reference == "all_test_items":
        total_common_items = len(top_k_result.intersection(ground_truth_items))
    elif reference == "k_ranked_test_items":
        items_to_pass = k
        total_common_items = 0
        for gt_items_by_score in ground_truth_items_grouped:
            if items_to_pass <= 0:
                break
            set_gt_items_by_score = set(gt_items_by_score.astype(int))
            common_items = len(top_k_result.intersection(set_gt_items_by_score))
            if common_items > items_to_pass:
                common_items = items_to_pass
            total_common_items += common_items
            items_to_pass -= len(gt_items_by_score)
    else:
        raise ValueError(f'The option {reference} for reference items in computing precision is invalid')
    if k == 0:
        return 0
    return round(total_common_items / k, 6)


def compute_recall(top_k_result, ground_truth_items):
    ground_truth_items = set(ground_truth_items)
    top_k_result = set(top_k_result)
    total_common_items = len(top_k_result.intersection(ground_truth_items))
    if len(ground_truth_items) == 0:
        return 0
    return round(total_common_items / len(ground_truth_items), 4)


def write_results_to_csv(results_path, header_results_file, results_rows):
    with open(f'{results_path}results.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header_results_file)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(results_rows)

