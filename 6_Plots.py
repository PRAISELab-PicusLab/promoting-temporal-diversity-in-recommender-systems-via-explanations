"""
Plot Recall@K and Diversity@K across iterations, one line per user_model.
One figure per (dataset, recommender, config) combination.
All Recall figures → plots/recall.pdf
All Diversity figures → plots/diversity.pdf

Usage:
    python plot_metrics.py
    python plot_metrics.py --metrics simulation_results/metrics.csv
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np


def make_figure(group_df, dataset, recommender, config, metric, label):
    user_models = sorted(group_df['user_model'].unique())
    colors = cm.tab10(np.linspace(0, 1, len(user_models)))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{dataset}  |  {recommender}  |  {config}", fontsize=12, fontweight='bold')

    for color, um in zip(colors, user_models):
        um_df = group_df[group_df['user_model'] == um].sort_values('iteration')
        valid  = um_df.dropna(subset=[metric])
        ax.plot(valid['iteration'], valid[metric],
                marker='o', label=um, color=color, linewidth=1.8)

    ax.set_title(label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(label)
    ax.legend(title='User Model', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


def main(metrics_path: str):
    df = pd.read_csv(metrics_path)

    out_dir = os.path.join(os.path.dirname(metrics_path), "plots")
    os.makedirs(out_dir, exist_ok=True)

    recall_pdf    = pdf_backend.PdfPages(os.path.join(out_dir, "recall.pdf"))
    diversity_pdf = pdf_backend.PdfPages(os.path.join(out_dir, "diversity.pdf"))

    groups = df.groupby(['dataset', 'recommender', 'config'])
    for (dataset, recommender, config), group_df in groups:
        print(f"{dataset} | {recommender} | {config}")

        fig_recall = make_figure(group_df, dataset, recommender, config,
                                 'avg_recall', 'Recall@K (%)')
        recall_pdf.savefig(fig_recall, bbox_inches='tight')
        plt.close(fig_recall)

        fig_div = make_figure(group_df, dataset, recommender, config,
                              'avg_diversity', 'Diversity@K (%)')
        diversity_pdf.savefig(fig_div, bbox_inches='tight')
        plt.close(fig_div)

    recall_pdf.close()
    diversity_pdf.close()
    print(f"\nSaved → {os.path.join(out_dir, 'recall.pdf')}")
    print(f"Saved → {os.path.join(out_dir, 'diversity.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='simulation_results/metrics.csv')
    args = parser.parse_args()
    main(args.metrics)
