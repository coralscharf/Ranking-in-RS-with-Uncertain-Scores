import argparse
from ranking_task import *


def run_ranking_pipeline(seed=0, dataset="ml-25m", first_stage_model_name="BeMF",
                         candidate_set_size=None, candidate_set_selection_method=None,
                         add_prob_to_score=False, truncate_continuous_distribution=True,
                         top_k_min_size=5, top_k_max_size=20, evaluation_jumps=5,
                         calculate_expected_precision=False, write_to_csv=False,
                         save_rank_dist_results=True):
    torch.manual_seed(seed)

    top_k_size_options = list(range(top_k_min_size, top_k_max_size + 1, evaluation_jumps))
    scores_type = "scores_with_prob" if add_prob_to_score else "equal_scores"
    fs_models_and_data_path = f'{FIRST_STAGE_MODELS_AND_DATA_MAIN_PATH}{dataset}/seed={seed}/'

    data, first_stage_model = load_first_stage_model_and_data(fs_models_and_data_path, first_stage_model_name)

    unc_thresholds_by_percentiles = None
    if "uncertainty_based_filtering" in candidate_set_selection_method or \
            any("uncertainty_based_filtering" in approach for approach in TOP_K_APPROACHES):
        rand_preds = first_stage_model.predict(data.rand['users'].to(device),
                                               data.rand['items'].to(device))
        unc_thresholds_by_percentiles = np.nanquantile(rand_preds[1], PERCENTILES_FOR_UBF)

    score_distribution_type = MODELS_SCORE_DISTRIBUTION_TYPE[first_stage_model_name]
    dataset_raw_score_values = np.array(POSSIBLE_SCORES[dataset])
    if score_distribution_type == "discrete":
        score_values = torch.tensor(POSSIBLE_SCORES[dataset])
    elif score_distribution_type == "normal":
        if truncate_continuous_distribution:
            score_values = torch.tensor(POSSIBLE_SCORES[dataset])[:-1]
        else:
            score_values = torch.tensor([0.0] + POSSIBLE_SCORES[dataset])
    else:
        raise ValueError(f'{score_distribution_type} for score distribution type is invalid')
    score_values = score_values.to(device)

    rank_dist_results_path = None
    if write_to_csv:
        items_ranked_selection_method = "full-ranking" if candidate_set_size is None \
            else f"{candidate_set_size}-{candidate_set_selection_method}"

        is_trunc_dist = (f"trunc_normal_dist/" if score_distribution_type == "normal" and
                                               truncate_continuous_distribution else "")
        results_path = ''.join(('./results/', dataset, '/', first_stage_model_name, '/', scores_type, '/',
                                is_trunc_dist, items_ranked_selection_method, f'/seed={seed}', '/'))
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        header_results_file = ["user_id", "approach", "top_k_size", "top_k_result"] + \
                              [metric for metric in METRICS]
        if calculate_expected_precision:
            header_results_file.append("expected_precision")
        results_rows = list()

        if save_rank_dist_results:
            rank_dist_results_path = ''.join((results_path, '/rank_dist_results/'))
            if not os.path.exists(rank_dist_results_path):
                os.makedirs(rank_dist_results_path)

    test_users = data.test_users
    for user_idx, user_id in enumerate(test_users):
        print(f"User IDX: {user_idx}, User ID: {user_id}")
        start_time = time.time()
        user_id_tensor = torch.tensor(user_id).to(device)

        train_val_item_ids = list(data.train_val.item[data.train_val.user == user_id])
        user_test_data = data.test[data.test[:, 0] == user_id][:, 1::]

        rec_items_by_preds = first_stage_model.recommend(user_id_tensor,
                                                         remove_items=np.array(train_val_item_ids),
                                                         n=data.n_item - len(train_val_item_ids))

        if score_distribution_type == "discrete":
            item_score_distributions = (
                first_stage_model.predict_items_distribution_for_user(user_id_tensor).to(device)
            )
            user_ranking_task = UserRankingTask(user_id, train_val_item_ids, user_test_data,
                                                rec_items_by_preds, item_score_distributions,
                                                score_values, top_k_size_options,
                                                unc_thresholds_by_percentiles)
        elif score_distribution_type == "normal":
            item_means, item_stds = first_stage_model.predict_items_mean_and_var_for_user(user_id_tensor)
            item_score_distributions = (
                create_items_discrete_distribution_from_normal_distribution(
                    item_means, item_stds, dataset_raw_score_values, truncate_continuous_distribution
                )
            ).to(device)
            user_ranking_task = UserRankingTask(user_id, train_val_item_ids, user_test_data,
                                                rec_items_by_preds, item_score_distributions,
                                                score_values, top_k_size_options,
                                                unc_thresholds_by_percentiles,
                                                item_means, item_stds)
        else:
            raise ValueError(f'{score_distribution_type} for score distribution type is invalid')

        user_ranking_task.generate_candidate_items_set(add_prob_to_score, candidate_set_size,
                                                       candidate_set_selection_method)
        user_ranking_task.run_rank_dist(scores_type, rank_dist_results_path)
        user_ranking_task.generate_top_k_answers()

        for top_k_size in top_k_size_options:
            for method in TOP_K_APPROACHES:
                top_k_res_item_ids, top_k_res_item_idxs = user_ranking_task.get_top_k_result_by_method(method,
                                                                                                       top_k_size)
                metric_results = user_ranking_task.compute_exact_metrics(top_k_res_item_ids)

                if calculate_expected_precision:
                    start_eval_time = time.time()
                    precision_calculator = PrecisionCalculator(user_ranking_task.item_score_distributions,
                                                               user_ranking_task.item_score_values,
                                                               device)
                    log_expected_precision = \
                        precision_calculator.calculate_expected_precision(top_k_res_item_idxs.tolist())[1]
                    expected_precision = np.exp(log_expected_precision)
                    metric_results['expected_precision'] = expected_precision
                    print(f"Eval time took {time.time() - start_eval_time} sec")

                if write_to_csv:
                    results_row = {"user_id": user_id, "approach": method, "top_k_size": top_k_size,
                                   "top_k_result": top_k_res_item_ids}
                    results_row.update(metric_results)
                    results_rows.append(results_row)

                print(f"Top-{top_k_size} result by method: {method} is {top_k_res_item_ids}")
                print("\t".join(f"{metric}: {result}" for metric, result in metric_results.items()))
                print("-" * 10)

        print(f"User {user_id} execution finished in {round(time.time()-start_time, 4)}")
        print("*"*25)

        if write_to_csv and user_idx % 100 == 0:
            write_results_to_csv(results_path, header_results_file, results_rows)
            results_rows.clear()
            print(f"Writing the results to CSV, finishing execution of user_idx={user_idx}")

    if write_to_csv and results_rows:
        write_results_to_csv(results_path, header_results_file, results_rows)
    print(f"Finished!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, default=0)
    parser.add_argument('--dataset_name', type=str, required=True, default="ml-25m")  # "ml-25m" or "netflix"
    parser.add_argument('--fs_model_name', type=str, required=True, default="CPMF")  # "OrdRec" / "BeMF" / "CPMF"
    parser.add_argument('--candidate_set_size', type=int, required=True, default=500)
    parser.add_argument('--candidate_set_selection_method', type=str, required=True,
                        default="reranking-probability_of_relevance_ranking")
    parser.add_argument('--save_rank_dist_results', action='store_true', default=True)

    args = parser.parse_args()

    seed = args.seed
    dataset = args.dataset_name
    first_stage_model_name = args.fs_model_name
    candidate_set_size = args.candidate_set_size
    candidate_set_selection_method = args.candidate_set_selection_method
    save_rank_dist_results = args.save_rank_dist_results

    run_ranking_pipeline(seed=seed, dataset=dataset,
                         first_stage_model_name=first_stage_model_name,
                         candidate_set_size=candidate_set_size,
                         candidate_set_selection_method=candidate_set_selection_method,
                         add_prob_to_score=False, truncate_continuous_distribution=True,
                         top_k_min_size=1, top_k_max_size=20, evaluation_jumps=1,
                         calculate_expected_precision=False, write_to_csv=True,
                         save_rank_dist_results=save_rank_dist_results
                         )


if __name__ == '__main__':
    main()
