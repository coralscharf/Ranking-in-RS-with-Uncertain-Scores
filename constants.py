import os, sys
import numpy as np
import torch

sys.path.append('./probability_computation/uncertain_SnapExplicit/')

POSSIBLE_SCORES = {"ml1m": [1.0, 2.0, 3.0, 4.0, 5.0],
                   "ml-25m": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                   "netflix": [1.0, 2.0, 3.0, 4.0, 5.0],}
MODELS_SCORE_DISTRIBUTION_TYPE = {"BeMF": "discrete",
                                  "OrdRec": "discrete",
                                  "CPMF": "normal"}
FIRST_STAGE_MODELS_AND_DATA_MAIN_PATH = "./probability_computation/uncertain_SnapExplicit/tests/"

PRR_THRESHOLD = 4.0
PERCENTILES_FOR_UBF = np.around(np.linspace(1.0, 0.2, 5), decimals=1)

TOP_K_APPROACHES = ["expected_score", "global_top_k",
                    "probability_of_relevance_ranking"]  # "expected_rank", "PT-k", "U-k-ranks"
for percentile in PERCENTILES_FOR_UBF:
    TOP_K_APPROACHES.append(f"uncertainty_based_filtering-{str(percentile)}")

METRICS = ['recall', 'precision_r_all', 'precision_r_k_ranked', 'dcg_c', 'dcg_l']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cur_path = os.path.dirname(__file__)

