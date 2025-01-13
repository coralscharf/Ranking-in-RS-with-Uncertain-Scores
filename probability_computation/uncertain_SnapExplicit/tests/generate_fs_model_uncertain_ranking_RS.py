
import os
import io
import pickle
import warnings
from copy import deepcopy
from zipfile import ZipFile
from datetime import datetime
warnings.filterwarnings("ignore")
from IPython.display import clear_output

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm


from uncertain.utils.data import Data
from uncertain.utils.training import train
from uncertain.utils.evaluation import test
from uncertain.explicit import Bias, MF, CPMF, OrdRec, BeMF
from uncertain.extras import Ensemble, Resample, UncertainWrapper, UserHeuristic, ItemHeuristic

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

import argparse


def extract_items_distribution_for_users(model, data):
    user_col_data, item_col_data = [], []
    score_distribution_data = torch.tensor([])
    for user_idx, user in enumerate(tqdm(data.test_users)):
        items_distribution = model.predict_items_distribution_for_user(torch.tensor(user))
        items_num = items_distribution.shape[0]
        user_col_data.extend([user] * items_num)
        item_col_data.extend(list(range(items_num)))
        score_distribution_data = torch.cat((score_distribution_data, items_distribution), axis=0)
        # score_distribution
    df = pd.DataFrame({'user_id': user_col_data,
                       'item_id': item_col_data,
                       'score_distribution': score_distribution_data.tolist()})

def train_BeMF(data, main_path, dataset_name):
    data.to_ordinal()
    dim_options = [50, 100, 200] if dataset_name == "ml-25m" else [50]
    for dim in dim_options:
        for wd in [0.001, 0.0001]:
            for lr in [0.001, 0.005]:
                print(f"Training BeMF model with the following parameters; dim: {dim}, wd: {wd}, lr: {lr}")
                model = BeMF(data.n_user, data.n_item, data.score_labels, embedding_dim=dim,
                             lr=lr, weight_decay=wd)
                train(model, data, path=f'{main_path}checkpoints/bemf', name=f'dim={dim}-wd={wd}-lr={lr}')

def train_CPMF(data, main_path):
    for dim in [50, 100, 200]:
        for lr in [5e-6, 1e-6]:
            print(f"Training CPMF model with the following parameters; dim: {dim}, lr: {lr}")
            model = CPMF(data.n_user, data.n_item, embedding_dim=dim, lr=0.0002, weight_decay=0, lr_var=lr)
            train(model, data, path=f'{main_path}checkpoints/cpmf', name=f'dim={dim}-lr={lr}')


def train_OrdRec(data, main_path):
    data.to_ordinal()
    for wd in [0.0001]:
        for lr in [0.00005]:
            print(f"Training OrdRec model with the following parameters; wd: {wd}, lr: {lr}")
            model = OrdRec(data.n_user, data.n_item, data.score_labels, embedding_dim=50, lr=0.0001, weight_decay=wd,
                           lr_step=lr)
            train(model, data, path=f'{main_path}checkpoints/ordrec', name=f'wd={wd}-lr={lr}')


def process_data(main_path, dataset_name, seed):
    print("Starting Process Data!")
    if os.path.isfile(f'{main_path}data.pkl'):
        with open(f'{main_path}data.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        if dataset_name == "ml-25m":
            data = pd.read_csv('./data/ml-25m/ratings.csv')  # ratings_small.csv'
            data.columns = ['user', 'item', 'score', 'timestamps']
            data = Data(data, implicit=False, users_on_test=10000, seed=seed)  # users_on_test=100
            with open(f'{main_path}data.pkl', 'wb') as f:
                pickle.dump(data, f, protocol=4)
        elif dataset_name == "netflix":
            zip_file = ZipFile('./data/netflix/data.zip')
            data = {'user': [], 'item': [], 'score': [], 'timestamps': []}
            for file in zip_file.infolist():
                if file.filename.startswith('combined'):
                    print('processing {0}'.format(file.filename))
                    with io.TextIOWrapper(zip_file.open(file.filename), encoding='utf-8') as f:
                        movie = -1
                        for line in f:
                            if line.endswith(':\n'):
                                movie = int(line[:-2]) - 1
                                continue
                            assert movie >= 0
                            splitted = line.split(',')
                            data['user'].append(int(splitted[0]))
                            data['item'].append(movie)
                            data['score'].append(float(splitted[1]))
                            data['timestamps'].append(
                                datetime.strptime(splitted[2].replace('\n', ''), '%Y-%m-%d').strftime('%Y%m%d%H%M%S'))
            _, data['user'] = np.unique(data['user'], return_inverse=True)
            data['score'] = np.array(data['score'], dtype=np.int32)
            data['item'] = np.array(data['item'], dtype=np.int32)
            data['timestamps'] = np.array(data['timestamps'], dtype=np.int32)
            data = Data(pd.DataFrame(data), implicit=False, users_on_test=10000)
            with open(f'{main_path}data.pkl', 'wb') as f:
                pickle.dump(data, f)
        elif dataset_name.startswith('amazon_'):
            amazon_with_category_idx = dataset_name.find("_", dataset_name.find("_") + 1)
            amazon_with_category = dataset_name[:amazon_with_category_idx]
            data = pd.read_csv(f'./data/{amazon_with_category}/{dataset_name}.csv')
            data.columns = ['user', 'item', 'score', 'timestamps']
            data = Data(data, implicit=False, users_on_test=10000, seed=seed, distances=False)
            with open(f'{main_path}data.pkl', 'wb') as f:
                pickle.dump(data, f, protocol=4)
        else:
            raise ValueError(f'The dataset of {dataset_name} is invalid')

    print(f'Data prepared: {data.n_user} users, {data.n_item} items.')
    print(f'{len(data.train)} train, {len(data.val)} validation and {len(data.test)} test interactions.')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, default="ml-25m")
    parser.add_argument('--train', action="store_true", help="Flag that indicated whether to train the model")
    parser.add_argument('--test', action="store_true", help="Flag that indicated whether to test the best model trained")
    parser.add_argument('--model_name', type=str, required=True, default="BeMF")
    parser.add_argument('--seed', type=int, required=True, default=0)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    seed = args.seed
    print(f"Dataset: {dataset_name}, Model name: {model_name}, seed={seed}")

    main_path = f'./tests/{dataset_name}/seed={seed}/'
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    data = process_data(main_path, dataset_name, seed)

    if args.train:
        if model_name == "BeMF":
            # The dataset_name is forwarded to this function since the
            # hyperparameters options differ between the datasets
            train_BeMF(data, main_path, dataset_name)
        elif model_name == "CPMF":
            train_CPMF(data, main_path)
        elif model_name == "OrdRec":
            train_OrdRec(data, main_path)

# score_labels = pd.factorize(data.train[:, 2], sort=True)[1]
# files = {file: float(file.split('val_loss=')[1][:-5]) for file in os.listdir('checkpoints/bemf')}
# model = BeMF(data.n_user, data.n_item, score_labels=score_labels, embedding_dim=20)
# model = model.load_from_checkpoint(os.path.join('checkpoints/bemf', min(files, key=files.get)))
# test(model, data, name='BeMF', max_k=10)
#
# extract_items_distribution_for_users(model, data)
