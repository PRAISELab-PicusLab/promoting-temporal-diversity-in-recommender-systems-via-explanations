from utils import *

def ers(dataset, recommender, iteration, corrective_iteration):
    print(f"\nITERATION {iteration} - DATALOADER: PREPROCESSING")
    print("-------------------------------------------------------------------------")
    ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PGPR_DIR  = os.path.join(ROOT_DIR, "recommenders", "PGPR")
    CAFE_DIR  = os.path.join(ROOT_DIR, "recommenders", "CAFE")
    os.chdir(PGPR_DIR)

    command = f"python preprocess.py --dataset {dataset}"
    subprocess.run(command, shell=True)

    print(f"\nITERATION {iteration} - TRAIN TRANS-E EMBEDDING")
    print("-------------------------------------------------------------------------")
    first_epochs_embedding = 30

    start_time = time.time()
    if iteration == 1:
        # Train embeddings from scratch with higher learning rate
        command = f"python train_transe_model.py --dataset {dataset} --iteration {iteration} --epochs {first_epochs_embedding} --lr 0.5"
    else:
        # Continue / refine embeddings for later iterations with smaller LR
        command = f"python train_transe_model.py --dataset {dataset} --iteration {iteration} --epochs {first_epochs_embedding} --lr 0.005"
    subprocess.run(command, shell=True)
    print_elapsed_time(iteration, start_time, "TRAIN TRANS-E EMBEDDING")

    if recommender == 'PGPR':
        epochs_training = 30

        # Train agent at iteration 1 and at every corrective boundary
        if iteration == 1 or (iteration > 1 and ((iteration - 1) % corrective_iteration == 0)):
            print(f"\nITERATION {iteration} - TRAIN AGENT")
            print("-------------------------------------------------------------------------")
            command = f"python train_agent.py --dataset {dataset} --epochs {epochs_training} --seed 123"
            start_time = time.time()
            subprocess.run(command, shell=True)
            print_elapsed_time(iteration, start_time, "TRAIN AGENT")

        # Extract paths with the trained agent
        print(f"\nITERATION {iteration} - EXTRACT PATHS: {recommender}")
        print("-------------------------------------------------------------------------")
        command = f"python test_agent.py --epochs {epochs_training} --dataset {dataset}"
        start_time = time.time()
        subprocess.run(command, shell=True)
        print_elapsed_time(iteration, start_time, "EXTRACT PATHS")

    elif recommender == "CAFE":
        # Switch working directory to CAFE implementation
        epochs_training = 1
        os.chdir(CAFE_DIR)

        # CAFE-specific preprocessing
        print(f"\nITERATION {iteration} - CAFE: PREPROCESSING")
        print("-------------------------------------------------------------------------")
        command = f"python preprocess.py --dataset {dataset} --iteration {iteration}"
        subprocess.run(command, shell=True)

        # Train the neural symbolic model
        print(f"\nITERATION {iteration} - TRAIN NEURAL SYMBOL")
        print("-------------------------------------------------------------------------")
        start_time = time.time()
        command = f"python train_neural_symbol.py --dataset {dataset} --iteration {iteration} --epochs {epochs_training} --name neural_symbolic_model"
        subprocess.run(command, shell=True)
        print_elapsed_time(iteration, start_time, "TRAIN NEURAL SYMBOL")

        # Extract paths
        print(f"\nITERATION {iteration} - EXTRACT PATHS: {recommender}")
        print("-------------------------------------------------------------------------")
        command = f"python execute_neural_symbol.py --dataset {dataset} --do_execute True"
        start_time = time.time()
        subprocess.run(command, shell=True)
        print_elapsed_time(iteration, start_time, "EXTRACT PATHS")

    print(f"\nITERATION {iteration} - EXTRACTION RECOMMENDATION")
    print("-------------------------------------------------------------------------")
    # Return to project root
    os.chdir(ROOT_DIR)

    # Load all predicted paths and aggregate by (uid, item)
    all_pred_paths_df = pd.read_csv('results/all_pred_paths.csv')
    start_time = time.time()
    aggregate_all_pred_paths_df = (
        all_pred_paths_df
        .groupby(['uid', 'rec item'])
        .agg({'path_score': list, 'path': list})
        .reset_index()
    )
    aggregate_all_pred_paths_df['max_path_score'] = aggregate_all_pred_paths_df['path_score'].apply(max_path_score)
    aggregate_all_pred_paths_df = aggregate_all_pred_paths_df.drop(['path_score'], axis=1)
    aggregate_all_pred_paths_df = aggregate_all_pred_paths_df.sort_values(by=['uid', 'max_path_score'], ascending=[True, False])
    aggregate_all_pred_paths_df['max_path_score'] = aggregate_all_pred_paths_df['max_path_score'].round(3)

    # Standardize column names and save per-iteration recommendations
    aggregate_all_pred_paths_df.rename(
        columns={'uid': 'uid', 'rec item': 'item', 'max_path_score': 'score', 'path': 'paths'},
        inplace=True
    )
    aggregate_all_pred_paths_df.to_csv(
        f'results/recommendations/iteration_{iteration}.csv',
        columns=['uid', 'item', 'score', 'paths'],
        sep=',', index=False
    )
    print_elapsed_time(iteration, start_time, "EXTRACTION RECOMMENDATION")
    print("DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset: ML1M | LFM1M | CELLPHONES')
    parser.add_argument('--recommender', type=str, default='CAFE', help='Model name: PGPR | CAFE')
    parser.add_argument('--iteration', type=int, default=1, help='Current iteration index')
    parser.add_argument('--corrective_iteration', type=int, default=5, help='Apply retraining every k-1 iters')
    args = parser.parse_args()

    ers(args.dataset, args.recommender, args.iteration, args.corrective_iteration)


if __name__ == "__main__":
    main()
