from mapper import *
from tqdm import tqdm
from collections import defaultdict, Counter

import os
import re
import ast
import csv
import math
import gzip
import json
import time
import torch
import pickle
import random
import shutil
import argparse
import datetime
import warnings
import subprocess

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def create_directory(directory_path):
    shutil.rmtree(directory_path, ignore_errors=True)
    os.makedirs(directory_path)


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def max_path_score(lst):
    return max(lst) if lst else None


def linear_decreasing_distribution(n):
    probabilities = [1 / (i + 1) for i in range(n)]
    total_prob = sum(probabilities)
    normalized_probabilities = [p / total_prob for p in probabilities]
    return normalized_probabilities


def print_elapsed_time(iteration, start_time, section):
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    if section in ["TRAIN TRANS-E EMBEDDING", "TRAIN AGENT", "TRAIN NEURAL SYMBOL", "EXTRACT PATHS"]:
        file_path = "../../results/utils/times.txt"
    else:
        file_path = "results/utils/times.txt"

    with open(file_path, 'a') as file:
        if minutes < 1:
            file.write(f"ITERATION {iteration} - {section}: {seconds} s\n")
        else:
            file.write(f"ITERATION {iteration} - {section}: {minutes} m {seconds} s\n")

        if section == "ALL PROCESS":
            file.write("\n")


def prepare_dataset(selected_dataset, selected_model, corrective_action, corrective_weight, trained_model, selected_users, users_ratings):
    dataset_folder_path = f'dataset/{selected_dataset}/dataset'

    create_directory('process/csv')
    create_directory('process/preprocessed')
    create_directory('process/train_test_set')
    create_directory('results/recommendations')
    create_directory('results/analysis')
    create_directory('results/utils')

    if trained_model == 0:
        create_directory(f'process/trained_model')

    if selected_dataset == 'ML1M':
        users_df = pd.read_csv(f'{dataset_folder_path}/users.dat', sep="::",
                            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
                            header=None, engine='python')
        items_df = pd.read_csv(f'{dataset_folder_path}/movies.dat', sep="::",
                            names=["ItemID", "Name", "Genres"],
                            header=None, encoding='latin-1', engine='python')
        ratings_df = pd.read_csv(f'{dataset_folder_path}/ratings.dat', sep="::",
                                names=["UserID", "ItemID", "Rating", "Timestamp"],
                                header=None, engine='python')
    elif selected_dataset == 'LFM1M':
        users_df = pd.read_csv(f"{dataset_folder_path}/users.txt", sep="\t",
                            names=["UserID", "Country", "Age", "Gender", "Play-count", "Timestamp"],
                            engine='python')
        items_df = pd.read_csv(f"{dataset_folder_path}/tracks.txt", sep="\t",
                            names=["ItemID", "Name", "ArtistID"],
                            engine='python')
        ratings_df = pd.read_csv(f"{dataset_folder_path}/ratings.txt", sep="\t",
                                names=["UserID", "ItemID", "Rating", "Timestamp"],
                                header=0, engine='python')
    elif selected_dataset == 'CELLPHONES':
        users_df = pd.read_csv(f"{dataset_folder_path}/users.csv", engine='python')
        items_df = pd.read_csv(f"{dataset_folder_path}/items.csv", engine='python')
        ratings_df = pd.read_csv(f"{dataset_folder_path}/ratings.csv", engine='python')

    # Sort dataframes
    users_df = users_df.sort_values(by=['UserID']).reset_index(drop=True)
    items_df = items_df.sort_values(by=['ItemID']).reset_index(drop=True)
    ratings_df = ratings_df.sort_values(by=['UserID', 'Timestamp']).reset_index(drop=True)

    # Print dataset statistics
    print(f"\nSELECTED DATASET: {selected_dataset.upper()}")
    print("-------------------------------------------------------------------------")
    print(f"ORIGINAL USERS: {len(users_df)}")
    print(f"ORIGINAL ITEMS: {len(items_df)}")
    print(f"ORIGINAL RATINGS: {len(ratings_df)}")

    user_id_counts = ratings_df['UserID'].value_counts().sort_values()
    filtered_user_ids = user_id_counts[user_id_counts > users_ratings - 1].head(selected_users)
    users = sorted(filtered_user_ids.index.tolist())

    selected_users = users_df[users_df["UserID"].isin(users)]
    selected_ratings = ratings_df[ratings_df['UserID'].isin(selected_users['UserID'])]
    selected_items_list = selected_ratings['ItemID'].unique()
    selected_items = items_df[items_df['ItemID'].isin(selected_items_list)]

    users_df = selected_users
    items_df = selected_items
    ratings_df = selected_ratings

    # Print filtered statistics
    print(f"\nFILTERED USERS: {len(users_df)}")
    print(f"FILTERED ITEMS: {len(items_df)}")
    print(f"FILTERED RATINGS: {len(ratings_df)}")

    # Save filtered dataset
    users_df.to_csv('process/csv/users.csv', sep=",", header=True, index=False)
    items_df.to_csv('process/csv/items.csv', sep=",", header=True, index=False)
    ratings_df.to_csv('process/csv/ratings.csv', sep=",", header=True, index=False)

    # Write experiment info to times.txt
    with open('results/utils/times.txt', 'a') as file:
        file.write(f"{len(users_df)} USERS - {len(items_df)} ITEMS - {len(ratings_df)} RATINGS - ")
        file.write(f"{selected_model} - {selected_dataset} - {corrective_action} - {corrective_weight}\n")
        file.write(f"-------------------------------------------------------------------------\n")

def prepare_dataloader(dataset, iteration):
    if iteration == 1:
        write_time_based_train_test_split(dataset, 0.7)

        shutil.copy('process/preprocessed/model/train.txt.gz', 'process/train_test_set')
        shutil.copy('process/preprocessed/model/train.txt.gz', 'results/utils')
        shutil.copy('process/preprocessed/model/test.txt.gz', 'process/train_test_set')
        shutil.copy('process/preprocessed/model/test.txt.gz', 'results/utils')
        shutil.copy('process/preprocessed/products.txt', 'results/utils')
        shutil.copy('process/preprocessed/users.txt', 'results/utils')
    else:
        # Reuse the same split in subsequent iterations
        shutil.copy('process/train_test_set/train.txt.gz', 'process/preprocessed/model')
        shutil.copy('process/train_test_set/test.txt.gz', 'process/preprocessed/model')

    mapper(dataset)

def copy_simulation_results(selected_dataset, selected_recommender, selected_corrective_action, selected_corrective_weight):
    if selected_corrective_weight == 0.0:
        dest_folder = os.path.join("simulation_results", selected_dataset, selected_recommender, "Baseline")
    else:
        dest_folder = os.path.join("simulation_results", selected_dataset, selected_recommender, f"{selected_corrective_action}_{selected_corrective_weight}")

    parent_dir = os.path.dirname(dest_folder)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    shutil.copytree("results", dest_folder)
