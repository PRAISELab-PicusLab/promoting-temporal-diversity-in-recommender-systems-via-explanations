from utils import *


def correction(iteration: int, corrective_action: str, corrective_weight: float) -> None:
    create_directory(f"results/analysis/iteration_{iteration}")

    # Load recommendations from current and previous iteration
    recommendation_previous_iteration_df = pd.read_csv(
        f"results/recommendations/iteration_{iteration - 1}.csv"
    )
    recommendation_iteration_df = pd.read_csv(
        f"results/recommendations/iteration_{iteration}.csv"
    )

    # Merge on (uid, item) to compare current vs previous explanations/scores
    merged_df = pd.merge(
        recommendation_iteration_df,
        recommendation_previous_iteration_df,
        on=['uid', 'item'],
        how='left',
        suffixes=('', '_previous')
    )
    merged_df = merged_df.drop(columns=['score_previous'])
    unique_uids = merged_df['uid'].unique()

    print(f"\nITERATION {iteration} - CORRECTIVE ACTION")
    print("-------------------------------------------------------------------------")

    if corrective_action == 'RR':
        # -----------------------------
        # Re-Ranking (RR)
        # -----------------------------
        explanations_diversity_threshold = 0.5

        # Work on a copy; drop path columns after we compute penalties
        modified_recommendation_iteration_df = merged_df
        modified_recommendation_iteration_df = modified_recommendation_iteration_df.drop(
            columns=['paths', 'paths_previous']
        )

        # Process each user independently (penalize low-diversity items)
        for uid in tqdm(unique_uids, desc="Re-Ranking (RR)"):
            uid_merged_df = merged_df[merged_df['uid'] == uid]
            relevance_scores = list(uid_merged_df['score'])

            reranking_elements = []  # indices to penalize
            diversity_scores = []    # diversity per (uid, item)

            for index, row in uid_merged_df.iterrows():
                if not pd.isna(row['paths_previous']):
                    # Explanation diversity via set overlap across iterations
                    paths_iteration = ast.literal_eval(row['paths'])
                    paths_previous_iteration = ast.literal_eval(row['paths_previous'])
                    common_paths = set(paths_previous_iteration) & set(paths_iteration)
                    unique_paths = set(paths_previous_iteration) ^ set(paths_iteration)

                    denom = len(common_paths) + len(unique_paths)
                    diversity = 1 - (len(common_paths) / denom) if denom > 0 else 1

                    # Low-diversity items get penalized
                    if diversity < explanations_diversity_threshold:
                        reranking_elements.append(index)

                    diversity_scores.append(diversity)

            # Aggregate user-level diversity signal
            diversity_score = (
                sum(diversity_scores) / len(diversity_scores) if len(diversity_scores) > 0 else 0
            )

            # Penalty magnitude mixes relevance and diversity, scaled by list length
            score = corrective_weight / len(uid_merged_df) * (
                sum(relevance_scores) + corrective_weight * diversity_score
            )

            # Apply penalty to low-diversity items
            for index in reranking_elements:
                modified_recommendation_iteration_df.loc[index, 'score'] -= score

        # Re-rank, collapse to list-of-items per user, and save
        modified_recommendation_iteration_df = (
            modified_recommendation_iteration_df
            .sort_values(by=['uid', 'score'], ascending=[True, False])
            .drop(columns=['score'])
            .groupby('uid')['item'].apply(list).reset_index()
            .rename(columns={"item": "items"})
        )
        modified_recommendation_iteration_df.to_csv(
            f"results/analysis/iteration_{iteration}/all_recommendation.csv", index=False
        )

    elif corrective_action == 'MW':
        # -----------------------------
        # Adjusting Item-related Weights (MW)
        # -----------------------------
        penalty = 0.005  # base factor applied along common paths

        # Load train split & entity maps; build dense user-item score matrix
        train_set_df = pd.read_csv(
            'process/train.txt.gz',
            compression='gzip', sep='\t',
            header=None, names=['uid', 'pid', 'score', 'timestamp']
        )
        users_df = pd.read_csv('process/users.txt', delimiter='\t')
        products_df = pd.read_csv('process/products.txt', delimiter='\t')
        matrix_score = np.zeros((len(users_df), len(products_df)))

        # Seed current scores
        for _, row in train_set_df.iterrows():
            matrix_score[int(row['uid']), int(row['pid'])] = row['score']

        # For each (uid, item), penalize edges along common explanation paths
        for _, row in tqdm(
            merged_df.iterrows(),
            desc="Adjusting Item-related Weights (MW)",
            total=len(merged_df)
        ):
            if not pd.isna(row['paths_previous']):
                paths_iteration = ast.literal_eval(row['paths'])
                paths_previous_iteration = ast.literal_eval(row['paths_previous'])
                common_paths = set(paths_previous_iteration) & set(paths_iteration)

                for path in common_paths:
                    # Format: U <uid> I <item1> U <user2> I <item2>
                    path_words = path.split()
                    uid_p, item1, user2, item2 = int(path_words[1]), int(path_words[3]), int(path_words[5]), int(path_words[7])
                    matrix_score[uid_p, item1] -= float(corrective_weight * penalty)
                    matrix_score[user2, item1] -= float(corrective_weight * penalty)
                    matrix_score[user2, item2] -= float(corrective_weight * penalty)

        # Write updated train scores (rounded) back to disk
        train_set_df['score'] = train_set_df.apply(
            lambda r: matrix_score[int(r['uid']), int(r['pid'])], axis=1
        )
        train_set_df['score'] = train_set_df['score'].round(3)
        train_set_df.to_csv(
            'process/train.txt.gz',
            index=False, compression='gzip', sep='\t', header=None
        )

        # Keep ratings.csv aligned with the corrected train scores (same interactions)
        users_map_df = pd.read_csv('process/users.txt', delimiter='\t')
        products_map_df = pd.read_csv('process/products.txt', delimiter='\t')
        ratings_df = pd.read_csv('process/ratings.csv')

        users_map_df = users_map_df.rename(columns={'new_id': 'uid', 'raw_dataset_id': 'UserID'})
        products_map_df = products_map_df.rename(columns={'new_id': 'pid', 'raw_dataset_id': 'ItemID'})

        train_sync_df = train_set_df[['uid', 'pid', 'timestamp', 'score']].copy()
        train_sync_df['uid'] = train_sync_df['uid'].astype(int)
        train_sync_df['pid'] = train_sync_df['pid'].astype(int)
        train_sync_df['timestamp'] = pd.to_numeric(train_sync_df['timestamp'], errors='coerce').round().astype('Int64')

        train_sync_df = train_sync_df.merge(users_map_df[['uid', 'UserID']], on='uid', how='left')
        train_sync_df = train_sync_df.merge(products_map_df[['pid', 'ItemID']], on='pid', how='left')
        train_sync_df = train_sync_df[['UserID', 'ItemID', 'timestamp', 'score']].rename(
            columns={'timestamp': 'Timestamp', 'score': 'Rating'}
        )

        ratings_df['Timestamp'] = pd.to_numeric(ratings_df['Timestamp'], errors='coerce').round().astype('Int64')
        train_sync_df = train_sync_df.dropna(subset=['UserID', 'ItemID', 'Timestamp'])

        ratings_df = ratings_df.merge(
            train_sync_df,
            on=['UserID', 'ItemID', 'Timestamp'],
            how='left',
            suffixes=('', '_mw')
        )
        ratings_df['Rating'] = ratings_df['Rating_mw'].combine_first(ratings_df['Rating'])
        ratings_df = ratings_df.drop(columns=['Rating_mw'])
        ratings_df.to_csv('process/ratings.csv', index=False)

        # Save list of items per user for current iteration (scores/paths not needed)
        recommendation_iteration_df = recommendation_iteration_df.drop(columns=['score', 'paths'])
        recommendation_iteration_df = (
            recommendation_iteration_df
            .groupby('uid')['item'].apply(list).reset_index()
            .rename(columns={"item": "items"})
        )
        recommendation_iteration_df.to_csv(
            f"results/analysis/iteration_{iteration}/all_recommendation.csv", index=False
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=2, help='Current iteration')
    parser.add_argument('--corrective_action', type=str, default='MW', help='RR | MW')
    parser.add_argument('--corrective_weight', type=float, default=0.5, help='Trade-off (accuracy vs diversity)')
    
    args = parser.parse_args()
    correction(args.iteration, args.corrective_action, args.corrective_weight)

if __name__ == "__main__":
    main()
