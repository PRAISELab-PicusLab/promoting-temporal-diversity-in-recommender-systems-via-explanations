from utils import *
from mapper import *


def models(model, dataset, iteration, corrective_iteration, trained_model):
    # -------------------------------
    # 1) TRAIN/TEST SPLIT (time-based)
    # -------------------------------
    print(f"\nITERATION {iteration} - DATALOADER: TRAIN TEST SPLIT")
    print("-------------------------------------------------------------------------")
    if iteration == 1:
        # Create a time-based split on first iteration and copy artifacts for traceability
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
    print("DONE")

    # -------------------------------
    # 2) MAPPER: prepare inputs for model dataloaders
    # -------------------------------
    print(f"\nITERATION {iteration} - DATALOADER: PREPARE FOR DATALOADER")
    print("-------------------------------------------------------------------------")
    mapper(dataset)
    print("DONE")

    # -------------------------------
    # 3) MODEL PREPROCESSING (PGPR path by default)
    # -------------------------------
    print(f"\nITERATION {iteration} - DATALOADER: PREPROCESSING")
    print("-------------------------------------------------------------------------")
    directory_path = f"models/PGPR"
    os.chdir(directory_path)

    command = f"python preprocess.py --dataset {dataset}"
    subprocess.run(command, shell=True)

    # -------------------------------
    # 4) TRANS-E EMBEDDINGS (train or reuse)
    # -------------------------------
    print(f"\nITERATION {iteration} - TRAIN TRANS-E EMBEDDING")
    print("-------------------------------------------------------------------------")
    first_epochs_embedding = 30
    trained_model_folder_path = '../../process/trained_model'
    tmp_folder_path = '../../process/preprocessed/model/tmp'

    start_time = time.time()
    if iteration == 1 and trained_model == 0:
        # Train embeddings from scratch (first run)
        command = f"python train_transe_model.py --dataset {dataset} --iteration {iteration} --epochs {first_epochs_embedding} --lr 0.5"
        subprocess.run(command, shell=True)

        # Save trained checkpoint so it can be reused
        create_directory(trained_model_folder_path)
        shutil.copy(f"{tmp_folder_path}/train_transe_model/transe_model.ckpt", trained_model_folder_path)
        shutil.copy(f"{tmp_folder_path}/transe_embed.pkl", trained_model_folder_path)

    elif iteration == 1 and trained_model == 1:
        # Reuse an externally trained checkpoint (no training)
        create_directory(f"{tmp_folder_path}/train_transe_model")
        shutil.copy(f"{trained_model_folder_path}/transe_model.ckpt", f"{tmp_folder_path}/train_transe_model")
        shutil.copy(f"{trained_model_folder_path}/transe_embed.pkl", tmp_folder_path)
        print("DONE")

    elif iteration > 1:
        # Continue / refine embeddings for later iterations with smaller LR
        command = f"python train_transe_model.py --dataset {dataset} --iteration {iteration} --epochs {first_epochs_embedding} --lr 0.005"
        subprocess.run(command, shell=True)

    print_elapsed_time(iteration, start_time, "TRAIN TRANS-E EMBEDDING")

    # -------------------------------
    # 5) MODEL-SPECIFIC TRAINING & PATH EXTRACTION
    # -------------------------------
    if model == 'PGPR':
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
        print(f"\nITERATION {iteration} - EXTRACT PATHS: {model}")
        print("-------------------------------------------------------------------------")
        command = f"python test_agent.py --epochs {epochs_training} --dataset {dataset}"
        start_time = time.time()
        subprocess.run(command, shell=True)
        print_elapsed_time(iteration, start_time, "EXTRACT PATHS")

    elif model == "CAFE":
        # Switch working directory to CAFE implementation
        epochs_training = 1
        directory_path = f"../../models/CAFE"
        os.chdir(directory_path)

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
        print(f"\nITERATION {iteration} - EXTRACT PATHS: {model}")
        print("-------------------------------------------------------------------------")
        command = f"python execute_neural_symbol.py --dataset {dataset} --do_execute True"
        start_time = time.time()
        subprocess.run(command, shell=True)
        print_elapsed_time(iteration, start_time, "EXTRACT PATHS")

    # -------------------------------
    # 6) AGGREGATE PATHS -> RECOMMENDATIONS
    # -------------------------------
    print(f"\nITERATION {iteration} - EXTRACTION RECOMMENDATION")
    print("-------------------------------------------------------------------------")
    # Return to project root
    directory_path = f"../../"
    os.chdir(directory_path)

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
    parser.add_argument('--model', type=str, default='CAFE', help='Model name: PGPR | CAFE')
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset: ML1M | LFM1M | CELLPHONES')
    parser.add_argument('--iteration', type=int, default=1, help='Current iteration index')
    parser.add_argument('--corrective_iteration', type=int, default=5, help='Apply retraining every k-1 iters')
    parser.add_argument('--trained_model', type=int, default=0, help='Use pre-trained TransE (1) or train (0)')
    args = parser.parse_args()
    models(args.model, args.dataset, args.iteration, args.corrective_iteration, args.trained_model)


if __name__ == "__main__":
    main()
