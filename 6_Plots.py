import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np


def make_figure(group_df, dataset, recommender, config, metric, label):
    user_models = [um for um in sorted(group_df['user_model'].unique()) if um != 'UNI' and um != 'CBM'][:6]
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

SELECTED_ITERATIONS = [2, 4, 6, 9, 11, 14, 16, 19, 21, 24, 26, 30]

def main(metrics_path: str, selected_recommenders=None, selected_user_models=None):
    df = pd.read_csv(metrics_path)
    df = df[df['iteration'].isin(SELECTED_ITERATIONS)]

    if selected_recommenders:
        df = df[df['recommender'].isin(selected_recommenders)]
    if selected_user_models:
        df = df[df['user_model'].isin(selected_user_models)]

    if df.empty:
        print("No data matches the selected filters.")
        return

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
    parser.add_argument('--recommenders', type=str, nargs='+', default=None,
                        help='Recommender models to include, e.g. --recommenders NeuMF ItemKNN')
    parser.add_argument('--user_models', type=str, nargs='+', default=None,
                        help='User models to include, e.g. --user_models UNI LIN TOP')
    args = parser.parse_args()
    main(args.metrics, args.recommenders, args.user_models)
