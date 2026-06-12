"""
Compute Recall@K, NDCG@K, Diversity@K, InterLD@K, RBO@K for every simulation.

Structure read:
  <base>/<dataset>/[<dataset>/]<recommender>/<user_model>/<config>/
      analysis/iteration_N/all_recommendation.csv
      utils/test.txt.gz

Output CSV:
  <base>/metrics_top{K}.csv
  Columns: dataset, recommender, user_model, config, iteration,
           avg_recall, avg_ndcg, avg_diversity, avg_interld, avg_rbo
"""

import os
import gzip
import argparse
import pandas as pd
from utils import (calculate_recall, calculate_diversity,
                   calculate_ndcg, calculate_interld, calculate_rbo)

TOP_K = 5


def load_test_set(utils_dir):
    path = os.path.join(utils_dir, "test.txt.gz")
    with gzip.open(path, "rt") as f:
        rows = [line.strip().split("\t") for line in f if line.strip()]
    df = pd.DataFrame(rows, columns=["uid", "pid", "rating", "timestamp"])
    df["uid"] = df["uid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    return df


def main(top_k: int, base: str = "Simulations"):
    records = []

    for dataset in sorted(os.listdir(base)):
        dataset_path = os.path.join(base, dataset)
        if not os.path.isdir(dataset_path):
            continue
        inner = os.path.join(dataset_path, dataset)
        if os.path.isdir(inner):
            dataset_path = inner

        for recommender in sorted(os.listdir(dataset_path)):
            rec_path = os.path.join(dataset_path, recommender)
            if not os.path.isdir(rec_path):
                continue
            for user_model in sorted(os.listdir(rec_path)):
                um_path = os.path.join(rec_path, user_model)
                if not os.path.isdir(um_path):
                    continue
                for config in sorted(os.listdir(um_path)):
                    config_path = os.path.join(um_path, config)
                    if not os.path.isdir(config_path):
                        continue

                    analysis_path = os.path.join(config_path, "analysis")
                    utils_path    = os.path.join(config_path, "utils")
                    if not os.path.isdir(analysis_path) or not os.path.isdir(utils_path):
                        continue

                    print(f"  {dataset} | {recommender} | {user_model} | {config}")

                    test_set   = load_test_set(utils_path)
                    iterations = sorted(
                        [d for d in os.listdir(analysis_path) if d.startswith("iteration_")],
                        key=lambda x: int(x.split("_")[1])
                    )

                    prev_df = None
                    for iter_dir in iterations:
                        iteration = int(iter_dir.split("_")[1])
                        rec_csv   = os.path.join(analysis_path, iter_dir, "all_recommendation.csv")
                        if not os.path.exists(rec_csv):
                            continue

                        curr_df = pd.read_csv(rec_csv)

                        _, avg_recall    = calculate_recall(test_set, curr_df, top_k)
                        _, avg_ndcg      = calculate_ndcg(test_set, curr_df, top_k)
                        _, avg_interld   = calculate_interld(curr_df, top_k)

                        if prev_df is not None:
                            _, avg_diversity = calculate_diversity(prev_df, curr_df, top_k)
                            _, avg_rbo       = calculate_rbo(prev_df, curr_df, top_k)
                        else:
                            avg_diversity = None
                            avg_rbo       = None

                        records.append({
                            "dataset":       dataset,
                            "recommender":   recommender,
                            "user_model":    user_model,
                            "config":        config,
                            "iteration":     iteration,
                            "avg_recall":    avg_recall,
                            "avg_ndcg":      avg_ndcg,
                            "avg_diversity": avg_diversity,
                            "avg_interld":   avg_interld,
                            "avg_rbo":       avg_rbo,
                        })

                        prev_df = curr_df

    metrics_df = pd.DataFrame(records)
    os.makedirs(base, exist_ok=True)
    out_path = os.path.join(base, f"metrics_top{top_k}.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--base",  type=str, default="Simulations")
    args = parser.parse_args()
    main(args.top_k, args.base)
