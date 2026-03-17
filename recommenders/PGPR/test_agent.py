from __future__ import absolute_import, division, print_function

import os
import gzip
import argparse
from math import log
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
from functools import reduce
from kg_env import BatchKGEnvironment
from train_agent import ActorCritic
from pgpr_utils import *
from extract_predicted_paths import *


def batch_beam_search(env, model, uids, device, topk=[25, 50, 1], seed=123):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(len(topk)):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        probs_max, _ = torch.max(probs, 0)
        probs_min, _ = torch.min(probs, 0)
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()
        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)

                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[env.dataset_name][path[-1][1]][relation] # Changing according to the dataset

                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, args):
    # 1) Predict paths
    torch.cuda.empty_cache()
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                             state_history=args.state_history)
    pretrain_sd = torch.load(policy_file, weights_only=False)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 25
    start_idx = 0
    all_paths, all_probs = [], []

    with tqdm(total=len(test_uids), desc="Predicting paths") as pbar:
        while start_idx < len(test_uids):
            end_idx = min(start_idx + batch_size, len(test_uids))
            batch_uids = test_uids[start_idx:end_idx]
            paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk, seed=args.seed)
            all_paths.extend(paths)
            all_probs.extend(probs)
            start_idx = end_idx
            pbar.update(batch_size)

    results = {'paths': all_paths, 'probs': all_probs}

    # 2) Evaluate paths
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[args.dataset]
    product = main_entity
    watched_embeds = embeds[main_relation][0]
    movie_embeds = embeds[main_entity]
    scores = np.dot(user_embeds + watched_embeds, movie_embeds.T)

    train_set_df = pd.read_csv('../../process/train.txt.gz', compression='gzip', sep='\t', header=None, names=['uid', 'pid', 'score', 'timestamp'])
    product_df = pd.read_csv('../../process/products.txt.gz', compression='gzip', sep='\t', header=0, names=['new_id', 'raw_dataset_id'])
    users_df = pd.read_csv('../../process/users.txt.gz', compression='gzip', sep='\t', header=0, names=['new_id', 'raw_dataset_id'])
    matrix_score = np.zeros((len(users_df), len(product_df)))

    for index, row in train_set_df.iterrows():
        matrix_score[int(row['uid']), int(row['pid'])] = row['score']

    # 3) Get all valid paths for each user, compute path score and path probability.
    pred_paths = {uid: {} for uid in test_labels}
    for path, probs in tqdm(zip(results['paths'], results['probs']), total=len(results['paths']), desc="Processing paths"):
        if path[0][1] == "user" and path[1][1] == product and path[2][1] == "user" and path[-1][1] == product and path[0][2] in pred_paths:
            if path[-1][2] not in pred_paths[path[0][2]]:
                pred_paths[path[0][2]][path[-1][2]] = []
            path_score = scores[path[0][2]][path[-1][2]] + matrix_score[int(path[0][2]), int(path[1][2])] + matrix_score[int(path[2][2]), int(path[1][2])] + matrix_score[int(path[2][2]), int(path[-1][2])]
            path_prob = reduce(lambda x, y: x * y, probs)
            pred_paths[path[0][2]][path[-1][2]].append((path_score, path_prob, path))

    # 4) Pick best path for each user-product pair, also remove pid if it is in train set.
    all_pred_paths = {}
    for uid in pred_paths:
        if uid in train_labels:
            train_pids = set(train_labels[uid])
        else:
            print("Invalid train_pids")
        all_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid not in train_pids:
                sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
                for element in sorted_path:
                    all_pred_paths[uid].append(element)

    torch.cuda.empty_cache()
    save_best_pred_paths("../../results/", all_pred_paths)


def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    predict_paths(policy_file, args)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=30, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=list, nargs='*', default=[10, 10, 10], help='number of samples')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = os.path.join(TMP_DIR[args.dataset], args.name)
    test(args)
