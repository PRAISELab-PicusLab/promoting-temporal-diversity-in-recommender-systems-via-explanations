from __future__ import absolute_import, division, print_function

import csv
import sys
import time

import pandas as pd
from collections import defaultdict

import numpy as np
import pickle
import logging
import logging.handlers
import math
from tqdm import tqdm
import torch
from torch.nn import functional as F

from my_knowledge_graph import *
from data_utils import KGMask
from symbolic_model import SymbolicNetwork, create_symbolic_model
from cafe_utils import *


logger = None


def set_logger(logname):
    global logger
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def infer_paths(args):
    kg = load_kg(args.dataset)
    model = create_symbolic_model(args, kg, train=False)

    train_labels = load_labels(args.dataset, 'train')
    train_uids = list(train_labels.keys())
    kg_mask = KGMask(kg)

    predicts = {}
    for uid in tqdm(train_uids, total=len(train_uids), desc="Paths Inference"):
        predicts[uid] = {}
        for mpid in range(len(kg.metapaths)):
            metapath = kg.metapaths[mpid]
            paths = model.infer_with_path(metapath, uid, kg_mask,
                                          excluded_pids=train_labels[uid],
                                          topk_paths=200)
            predicts[uid][mpid] = paths
    with open(args.infer_path_data, 'wb') as f:
        pickle.dump(predicts, f)




class MetaProgramExecutor(object):
    """This implements the profile-guided reasoning algorithm."""

    def __init__(self, symbolic_model, kg_mask, args):
        self.symbolic_model = symbolic_model
        self.kg_mask = kg_mask
        self.device = args.device

    def _get_module(self, relation):
        return getattr(self.symbolic_model, relation)

    def execute(self, program, uid, excluded_pids=None):
        """Execute the program to generate node representations and real nodes.
        Args:
            program: an instance of MetaProgram.
            uid: user ID (integer).
            excluded_pids: list of product IDs (list).
        """
        uid_tensor = torch.LongTensor([uid]).to(self.device)
        user_vec = self.symbolic_model.embedding(USER, uid_tensor)  # tensor [1, d]
        root = program.root  # TreeNode
        root.data['vec'] = user_vec  # tensor [1, d]
        root.data['paths'] = [([uid], [], [], [])]  # (path, value, mp)

        excluded_pids = [] if excluded_pids is None else excluded_pids.copy()

        # Run BFS to traverse tree.
        queue = root.get_children()
        while queue:  # queue is not empty
            node = queue.pop(0)
            child_nodes = np.random.permutation(node.get_children())
            queue.extend(child_nodes)

            # Compute estimated vector of the node.
            x = (node.parent.data['vec'], user_vec)
            node.data['vec'] = self._get_module(node.relation)(x)  # tensor [1, d]

            # Compute scores (log prob) for the node.
            entity_vecs = self.symbolic_model.embedding(node.entity)  # tensor [vocab, d]
            scores = torch.matmul(node.data['vec'], entity_vecs.t())  # tensor [1, vocab]
            scores = F.log_softmax(scores[0], dim=0)  # tensor [vocab, ]

            node.data['paths'] = []
            visited_ids = []
            for path, value, ep, mp in node.parent.data['paths']:
                # Find valid node ids for current path.
                valid_ids = self.kg_mask.get_ids(node.parent.entity, path[-1], node.relation)
                valid_ids = set(valid_ids).difference(visited_ids)
                if not node.has_children() and excluded_pids:
                    valid_ids = valid_ids.difference(excluded_pids)
                if not valid_ids:  # empty list
                    continue
                valid_ids = list(valid_ids)

                # Compute top k nodes.
                valid_ids = torch.LongTensor(valid_ids).to(self.device)
                valid_scores = scores.index_select(0, valid_ids)
                k = min(node.sample_size, len(valid_ids))
                topk_scores, topk_idxs = valid_scores.topk(k)
                topk_ids = valid_ids.index_select(0, topk_idxs)

                # Add nodes and scores to paths.
                topk_ids = topk_ids.detach().cpu().numpy()
                topk_scores = topk_scores.detach().cpu().numpy()
                for j in range(k):
                    new_path = path + [topk_ids[j]]
                    new_value = value + [topk_scores[j]]
                    new_mp = mp + [node.relation]
                    new_ep = ep + [node.entity]
                    node.data['paths'].append((new_path, new_value, new_ep, new_mp))

                    # Remember to add the node to visited list!!!
                    visited_ids.append(topk_ids[j])
                    if not node.has_children():
                        excluded_pids.append(topk_ids[j])

    def collect_results(self, program):
        entities, results = [], []
        #entities.append(program.root.entity)
        queue = program.root.get_children()
        while len(queue) > 0:
            node = queue.pop(0)
            #entities.append(node.entity)
            queue.extend(node.get_children())
            if not node.has_children():
                results.extend(node.data['paths'])
        return results


class TreeNode(object):
    def __init__(self, level, entity, relation):
        super(TreeNode, self).__init__()
        self.level = level
        self.entity = entity  # Entity type
        self.relation = relation  # Relation pointing to this tail entity
        self.parent = None
        self.children = {}  # key = (relation, entity), value = TreeNode
        self.sample_size = 0  # number of nodes to sample
        self.data = {}  # extra information to save

    def has_parent(self):
        return self.parent is not None

    def has_children(self):
        return len(self.children) > 0

    def get_children(self):
        return list(self.children.values())

    def __str__(self):
        parent = None if not self.has_parent() else self.parent.entity
        msg = '({},{},{})'.format(parent, self.relation, self.entity)
        return msg


class NeuralProgramLayout(object):
    """This refers to the layout tree in the paper."""

    def __init__(self, metapaths):
        super(NeuralProgramLayout, self).__init__()
        # self.metapaths = metapaths
        # print(metapaths)
        self.mp2id = {}
        for mpid, mp in enumerate(metapaths):
            simple_mp = tuple([v[0] for v in mp[1:]])
            self.mp2id[simple_mp] = mpid
        # self.root = None
        # self.initialize()
        
        self.root = TreeNode(0, USER, None)
        for mp in metapaths:
            node = self.root
            for i in range(1, len(mp)):
                # child = TreeNode(1, mp[i][1], mp[i][0])
                if mp[i] not in node.children:
                    node.children[mp[i]] = TreeNode(i, mp[i][1], mp[i][0])
                    node.children[mp[i]].parent = node
                node = node.children[mp[i]]

    def update_by_path_count(self, path_count):
        """Update sample size of each node by expected number of paths.
        Args:
            path_count: dict with key=mpid, value=int
        """
        def _postorder_update(node, parent_rels):
            if not node.has_children():
                mpid = self.mp2id[tuple(parent_rels)]
                node.sample_size = int(path_count[mpid])
                return

            min_pos_sample_size, max_sample_size = 99, 0
            for child in node.get_children():
                _postorder_update(child, parent_rels + [child.relation])
                max_sample_size = max(max_sample_size, child.sample_size)
                if child.sample_size > 0:
                    #min_pos_sample_size = min(max_sample_size, child.sample_size)
                    min_pos_sample_size = min(min_pos_sample_size, child.sample_size)

            # Update current node sampling size.
            # a) if current node is root, set to 1.
            if not node.has_parent():
                node.sample_size = 1
            # b) if current node is not root, and all children sample sizes are 0, set to 0.
            elif max_sample_size == 0:
                node.sample_size = 0
            # c) if current node is not root, take the minimum and update children.
            else:
                node.sample_size = min_pos_sample_size
                for child in node.get_children():
                    child.sample_size = int(child.sample_size / node.sample_size)

        _postorder_update(self.root, [])

    def print_postorder(self, hide_branch=True):
        def _postorder(node, msgs):
            msg = (node.entity, node.relation, node.sample_size)
            new_msgs = msgs + [msg]

            if not node.has_children():
                if hide_branch and msg[2] == 0:
                    return
                str_msgs = ['({},{},{})'.format(msg[0], msg[1], msg[2]) for msg in new_msgs]
                print('  '.join(str_msgs))
                return

            for child in node.children:
                _postorder(child, new_msgs)

        _postorder(self.root, [])


def create_heuristic_program(metapaths, raw_paths_with_scores, prior_count, sample_size):
    pcount = prior_count.astype(int)
    #pcount[pcount > 5] = 5

    mp_scores = np.ones(len(metapaths)) * -99
    for mpid in raw_paths_with_scores:
        paths = raw_paths_with_scores[mpid]
        if len(paths) <= 0:
            continue
        scores = np.array([p2[-1] for p1, p2 in paths])
        scores[scores < -5.0] = -5.0
        mp_scores[mpid] = np.mean(scores)

    top_idxs = np.argsort(mp_scores)[::-1]

    norm_count = np.zeros(len(metapaths))
    rest = sample_size
    for mpid in top_idxs:
        if pcount[mpid] <= rest:
            norm_count[mpid] = pcount[mpid]
        else:
            norm_count[mpid] = rest
        rest -= norm_count[mpid]

    program_layout = NeuralProgramLayout(metapaths)
    program_layout.update_by_path_count(norm_count)
    return program_layout


def save_pred_paths(dataset, pred_paths):
    extracted_path_dir = f"../../results/all_pred_paths.csv"

    df = pd.DataFrame(pred_paths, columns=['uid', 'rec item', 'path_score', 'path_prob', 'path'])
    df = df.drop_duplicates()

    path_scores = df['path_score'].values
    normalized_scores = (path_scores - path_scores.min()) / (path_scores.max() - path_scores.min())
    df['path_score'] = normalized_scores
    df.to_csv(extracted_path_dir, index=False)



def run_program(args):
    kg = load_kg(args.dataset)
    kg_mask = KGMask(kg)

    train_labels = load_labels(args.dataset,  'train')
    test_labels = load_labels(args.dataset, 'test')
    path_counts = load_path_count(args.dataset)  # Training path freq
    with open(args.infer_path_data, 'rb') as f:
        raw_paths = pickle.load(f)  # Test path with scores
    symbolic_model = create_symbolic_model(args, kg, train=False)
    program_exe = MetaProgramExecutor(symbolic_model, kg_mask, args)

    train_set_df = pd.read_csv('../../process/train.txt.gz', compression='gzip', sep='\t', header=None, names=['uid', 'pid', 'score', 'timestamp'])
    product_df = pd.read_csv(f'../../process/products.txt.gz', compression='gzip', sep='\t', header=0, names=['new_id', 'raw_dataset_id'])
    users_df = pd.read_csv(f'../../process/users.txt.gz', compression='gzip', sep='\t', header=0, names=['new_id', 'raw_dataset_id'])
    matrix_score = np.zeros((len(users_df), len(product_df)))

    for index, row in train_set_df.iterrows():
        matrix_score[int(row['uid']), int(row['pid'])] = row['score']

    pred_paths_istances = []

    for uid in tqdm(test_labels, total=len(test_labels), desc="  Extract Paths"):
        program = create_heuristic_program(kg.metapaths, raw_paths[uid], path_counts[uid], args.sample_size)
        train_labels[uid] = [int(label) for label in train_labels[uid]]
        program_exe.execute(program, uid, train_labels[uid])
        paths = program_exe.collect_results(program)

        for value in raw_paths[uid].values():
            paths.extend(value)   
        for raw in paths:
            raw_path = raw[0]
            raw_values = raw[1]
            if int(raw_path[1]) in train_labels[uid] and int(raw_path[3]) not in train_labels[uid]:
                # Minimal U I U I format: U <uid> I <item1> U <user2> I <item2>
                path = f"U {raw_path[0]} I {raw_path[1]} U {raw_path[2]} I {raw_path[3]}"
                path_score = raw_values[2] + matrix_score[int(raw_path[0]), int(raw_path[1])] + matrix_score[int(raw_path[2]), int(raw_path[1])] + matrix_score[int(raw_path[2]), int(raw_path[3])]
                pred_paths_istances.append([raw_path[0], raw_path[3], path_score, np.mean(raw_values), path])
    save_pred_paths(args.dataset, pred_paths_istances)


def main():
    args = parse_args()

    infer_paths(args)
    run_program(args)


if __name__ == '__main__':
    main()
