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


def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)


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
