from utils import *

def recommender(dataset, recommender, iteration, corrective_iteration, corrective_weight):
    if recommender in ['PGPR', 'CAFE']:
        command = f"python -m recommenders.ERS --dataset {dataset} --recommender {recommender} --iteration {iteration} --corrective_iteration {corrective_iteration} --corrective_weight {corrective_weight}"
    elif recommender in  ["NeuMF", "LightGCN"]:
        command = f"python -m recommenders.RecBole --recommender {recommender} --iteration {iteration} --corrective_weight {corrective_weight}"
    subprocess.run(command, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset: ML1M | LFM1M | CELLPHONES')
    parser.add_argument('--recommender', type=str, default='CAFE', help='Model name: PGPR | CAFE')
    parser.add_argument('--iteration', type=int, default=1, help='Current iteration index')
    parser.add_argument('--corrective_iteration', type=int, default=5, help='Apply retraining every k-1 iters')
    parser.add_argument('--corrective_weight', type=float, default=0.5, help='Trade-off weight for explanations (0 to skip retraining)')

    args = parser.parse_args()
    recommender(args.dataset, args.recommender, args.iteration, args.corrective_iteration, args.corrective_weight)

if __name__ == "__main__":
    main()
