# A Rank-Based Approach to Recommender System's Top-K Queries with Uncertain Scores

This is the code repository for our paper "A Rank-Based Approach to Recommender System's Top-K Queries with Uncertain Scores"
 that was accepted to [SIGMOD25](https://2025.sigmod.org/index.shtml).


## Running Experiments

### Probability Computation
The first step is training a model that will be used to generate the score distribution for each user-item pair. 
We use the code from [GitHub](https://github.com/vcoscrato/uncertain), which is available under the directory `probability_compuattion`.

To train a model, run the script `generate_fs_model_uncertain_ranking_RS.py` located in the 
 path `./probability_computation/uncertain_SnapExplicit/tests/` using the command below:
```
python generate_fs_model_uncertain_ranking_RS.py --dataset_name dataset_name \
                                                 --train \
                                                 --model_name model_name \
                                                 --seed seed
```

### Ranking with Uncertain Scores
The second step is to rank the items based on the uncertain scores.
The script `run_pipline.py` is used to run a ranking task in recommender systems. 
This code generates the uncertain scores using the trained model, 
ranks the items based on the uncertain scores using multiple approaches, 
and evaluates the ranking performance.

To run the code, use the following command:
```
python run_pipeline.py --seed seed \
                       --dataset_name dataset_name \
                       --fs_model_name fs_model_name \
                       --candidate_set_size candidate_set_size \
                       --candidate_set_selection_method candidate_set_selection_method                       
```

Required arguments:
* `--seed:` The seed used for reproducibility.
* `--dataset_name:` The name of the dataset to be used. 
* `--fs_model_name:` The name of the model responsible for generating score distribution for each user-item pair. 
* `--candidate_set_size:` The size of the candidate set. 
* `--candidate_set_selection_method:` The method used for generating the candidate set. In the paper, we used the `reranking-probability_of_relevance_ranking` method.
