from utils import *

# Available models, datasets, corrective actions, and weight values
models = ["PGPR", "CAFE"]
datasets = ["ML1M", "LFM1M", "CELLPHONES"]
actions = ['MW', 'RR']
possible_weight = ['0', '0.125', '0.25', '0.375', '0.5', '0.75', '1']

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
selected_model = "CAFE"
selected_dataset = "ML1M"
corrective_action = "RR"
corrective_weight = 0.25

selected_users = 100
users_ratings = 50
current_iteration = 1
final_iteration = 2
corrective_iteration = 2
trained_model = 0

if current_iteration == 1:
    # -----------------------------
    # CREATE DIRECTORY STRUCTURE
    # -----------------------------
    dataset_folder_path = f'dataset/{selected_dataset}/dataset'

    create_directory('process/csv')
    create_directory('process/preprocessed')
    create_directory('process/train_test_set')
    create_directory('results/recommendations')
    create_directory('results/analysis')
    create_directory('results/utils')

    if trained_model == 0:
        create_directory(f'process/trained_model')

    # -----------------------------
    # LOAD DATASET
    # -----------------------------
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

    # -----------------------------
    # FILTER DATASET
    # -----------------------------
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

# -----------------------------
# MAIN SIMULATION LOOP
# -----------------------------
for iteration in range(current_iteration, final_iteration+1):
    start_iteration = time.time()
    print("\n-------------------------------------------------------------------------")
    print(f"ITERATION {iteration}")
    print("-------------------------------------------------------------------------")

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    print(f"\nPREPROCESSING")
    print("*************************************************************************")
    command = f"python 1_Preprocessing.py --dataset {selected_dataset} --iteration {iteration}"
    subprocess.run(command, shell=True)

    # -----------------------------
    # MODEL TRAINING / INFERENCE
    # -----------------------------
    print(f"\nMODEL")
    print("*************************************************************************")
    command = f"python 2_Models.py --model {selected_model} --dataset {selected_dataset} --iteration {iteration} --corrective_iteration {corrective_iteration} --trained_model {trained_model}"
    subprocess.run(command, shell=True)

    # Remove intermediate results
    os.remove('results/all_pred_paths.csv')

    # -----------------------------
    # APPLY CORRECTIVE ACTION
    # -----------------------------
    if (iteration % corrective_iteration == 0 or corrective_action == 'RR') and iteration > 1 and corrective_weight > 0:
        print(f"\nCORRECTIVE ACTION")
        print("*************************************************************************")
        command = f"python 3_Correction.py --iteration {iteration} --corrective_action {corrective_action} --corrective_weight {corrective_weight}"
        start_time = time.time()
        subprocess.run(command, shell=True)
        if corrective_action == 'RR':
            print_elapsed_time(iteration, start_time, 'CORRECTIVE ACTION: RE-RANKING')
        else:
            print_elapsed_time(iteration, start_time, 'CORRECTIVE ACTION: MODIFY WEIGHTS')
    else:
        # Save recommendations without correction
        iteration_folder_path = f"results/analysis/iteration_{iteration}"
        create_directory(iteration_folder_path)
        recommendations_df = pd.read_csv(f"results/recommendations/iteration_{iteration}.csv")
        recommendations_df = recommendations_df.drop(columns=['score', 'paths'])
        recommendations_df = recommendations_df.groupby('uid')['item'].apply(list).reset_index()
        recommendations_df.rename(columns={"item": "items"}, inplace=True)
        recommendations_df.to_csv(f"{iteration_folder_path}/all_recommendation.csv", index=False)

    # -----------------------------
    # SAVE FINAL RECOMMENDATIONS
    # -----------------------------
    print(f"\nRECOMMENDATION")
    print("*************************************************************************")
    start_time = time.time()
    command = f"python 4_Recommendation.py --iteration {iteration}"
    subprocess.run(command, shell=True)
    print_elapsed_time(iteration, start_time, 'SAVING RECOMMENDATIONS')
    print_elapsed_time(iteration, start_iteration, 'ALL PROCESS')
