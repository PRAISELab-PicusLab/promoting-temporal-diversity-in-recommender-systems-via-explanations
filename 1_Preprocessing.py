from utils import *
warnings.filterwarnings("ignore")

def preprocessing(dataset, iteration):
    # 1) Load the filtered CSVs produced earlier in the pipeline
    users_df = pd.read_csv('process/csv/users.csv', sep=",", header=0, engine='python')
    items_df = pd.read_csv('process/csv/items.csv', sep=",", header=0, encoding='latin-1', engine='python')
    ratings_df = pd.read_csv('process/csv/ratings.csv', sep=",", header=0, engine='python')

    # 2) Remove dataset-specific columns we don’t need downstream
    if dataset == 'ML1M':
        users_df = users_df.drop(["Gender", "Age", "Occupation", "Zip-code"], axis=1)
    elif dataset == 'LFM1M':
        users_df = users_df.drop(["Country", "Age", "Gender", "Play-count", "Timestamp"], axis=1)
    elif dataset == 'CELLPHONES':
        users_df = users_df.drop(["Name"], axis=1)

    # 3) Assign new contiguous user IDs [0..num_users-1] and save mapping
    users_df.insert(0, 'new_id', range(users_df.shape[0]))
    users_df.to_csv(
        'process/preprocessed/users.txt',
        header=["new_id", "raw_dataset_id"],  # rename headers for downstream tools
        index=False, sep='\t', mode='w+'
    )

    # Build mapping: raw user ID -> new user ID
    user_id2new_id = dict(zip(users_df["UserID"], users_df['new_id']))

    # 4) Drop unneeded item columns depending on dataset
    if dataset == 'ML1M':
        items_df = items_df.drop(["Name", "Genres"], axis=1)
    elif dataset == 'LFM1M':
        items_df = items_df.drop(["Name", "ArtistID"], axis=1)
    # (CELLPHONES keeps all columns already minimal)

    # 5) Assign new contiguous item IDs and save mapping
    items_df.insert(0, 'new_id', range(items_df.shape[0]))
    items_df.to_csv(
        'process/preprocessed/products.txt',
        header=["new_id", "raw_dataset_id"],
        index=False, sep='\t', mode='w+'
    )

    # Build mapping: raw item ID -> new item ID
    item_id2new_id = dict(zip(items_df["ItemID"], items_df['new_id']))

    # 6) Remap ratings to the new (uid, pid) ID space and save
    ratings_df["UserID"] = ratings_df['UserID'].map(user_id2new_id)
    ratings_df["ItemID"] = ratings_df['ItemID'].map(item_id2new_id)
    ratings_df.to_csv(
        'process/preprocessed/ratings.txt',
        header=["uid", "pid", "rating", "timestamp"],
        index=False, sep='\t', mode='w+'
    )

    prepare_dataloader(dataset, iteration)

    # 7) Console summary for traceability
    print(f"\nITERATION {iteration} - DATASET: {dataset}")
    print("-------------------------------------------------------------------------")
    print(f"USERS: {len(users_df)}")
    print(f"ITEMS: {len(items_df)}")
    print(f"RATINGS: {len(ratings_df)}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for a given iteration.")
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset name: ML1M | LFM1M | CELLPHONES')
    parser.add_argument('--iteration', type=int, default=1, help='Current iteration index (int)')
    args = parser.parse_args()
    preprocessing(args.dataset, args.iteration)


if __name__ == "__main__":
    main()
