from utils import *


def recommendation(iteration, seed):
    np.random.seed(seed)

    # Paths and input data
    iteration_folder_path = f"results/analysis/iteration_{iteration}"
    train_test_folder_path = 'process/train_test_set'
    recommendation_df = pd.read_csv(f"{iteration_folder_path}/all_recommendation.csv")

    # -----------------------------
    # 4.1 SAVE FILTERED RECOMMENDATIONS (exclude items that are in test set)
    # -----------------------------
    test_set = pd.read_csv(
        'results/utils/test.txt.gz',
        compression='gzip', sep='\t', header=None,
        names=['uid', 'pid', 'score', 'timestamp']
    )
    unique_uids = recommendation_df['uid'].unique()
    recommended_items_df = pd.DataFrame(columns=['uid', 'items'])

    for uid in tqdm(unique_uids, desc="Recommendation"):
        # Items in the user's test set (to be filtered out)
        uid_test_set = test_set[test_set['uid'] == uid]
        test_set_items = list(uid_test_set['pid'])

        # Items recommended for this user (stringified list in the CSV)
        uid_recommendation_df = recommendation_df[recommendation_df['uid'] == uid]
        recommended_items = uid_recommendation_df['items'].iloc[0].strip('[]').split(', ')
        recommended_items = [int(item) for item in recommended_items]

        # Remove test-set items
        filtered_recommended_items = [item for item in recommended_items if item not in test_set_items]

        # Append row
        recommended_items_df.loc[len(recommended_items_df)] = {'uid': uid, 'items': filtered_recommended_items}

    # Persist the filtered recommendation lists per user
    recommended_items_df.to_csv(f'{iteration_folder_path}/real_recommendation.csv', index=False)

    # -----------------------------
    # 4.2 SAVE USER CHOICES (simulate one pick per user)
    # -----------------------------
    users_df = pd.read_csv('process/preprocessed/users.txt', delimiter='\t')
    products_df = pd.read_csv('process/preprocessed/products.txt', delimiter='\t')
    train_set = pd.read_csv(
        f'{train_test_folder_path}/train.txt.gz',
        compression='gzip', sep='\t', header=None,
        names=['uid', 'pid', 'score', 'timestamp']
    )

    selected_items_df = pd.DataFrame(columns=['uid', 'item'])

    for uid in tqdm(unique_uids, desc="Saving Choices"):
        # Get the filtered recommended items for the user
        uid_recommendation = recommended_items_df[recommended_items_df['uid'] == uid]['items'].iloc[0]

        if len(uid_recommendation) > 0:
            # Linearly decreasing probability over positions (top ranks more likely)
            probabilities = linear_decreasing_distribution(len(uid_recommendation))
            selected_index = np.random.choice(len(uid_recommendation), p=probabilities)
            selected_item = uid_recommendation[selected_index]

            # Map back to raw dataset IDs for ratings.csv (traceability)
            selected_user_row = users_df[users_df['new_id'] == uid]
            selected_product_row = products_df[products_df['new_id'] == selected_item]

            selected_user = selected_user_row["raw_dataset_id"].values[0]
            selected_product = selected_product_row["raw_dataset_id"].values[0]
            current_timestamp = datetime.datetime.now().timestamp()

            # Append to train set (uid, pid in the *new id* space)
            row = {'uid': uid, 'pid': selected_item, 'score': 0, 'timestamp': current_timestamp}
            train_set = pd.concat([train_set, pd.DataFrame([row])], ignore_index=True)

            # Append to ratings.csv (raw ID space)
            with open('process/csv/ratings.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([selected_user, selected_product, 0, int(current_timestamp)])

            # Log selected item (new ID space)
            selected_items_df.loc[len(selected_items_df)] = {'uid': uid, 'item': selected_item}

    # Ensure correct dtypes and persist updated train split
    train_set['uid'] = train_set['uid'].astype(int)
    train_set['pid'] = train_set['pid'].astype(int)
    train_set['timestamp'] = train_set['timestamp'].astype(int)
    train_set.to_csv(
        f'{train_test_folder_path}/train.txt.gz',
        compression='gzip', sep='\t', header=False, index=False
    )

    # Persist chosen items for this iteration
    selected_items_df.to_csv(f'{iteration_folder_path}/all_choices.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=3, help='Current iteration.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for choice sampling.')

    args = parser.parse_args()
    recommendation(args.iteration, args.seed)

if __name__ == "__main__":
    main()
