from libraries import *

def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def create_directory(directory_path):
    shutil.rmtree(directory_path, ignore_errors=True)
    os.makedirs(directory_path)

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def write_time_based_train_test_split(train_size):
    uid2pids_timestamp_tuple = defaultdict(list)
    with open('process/ratings.txt', 'r') as ratings_file: #uid	pid	rating	timestamp
        reader = csv.reader(ratings_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            k, pid, rating, timestamp = row
            uid2pids_timestamp_tuple[k].append([pid, rating, int(timestamp)]) #DR
    ratings_file.close()

    train_file = gzip.open('process/train.txt.gz', 'wt')
    writer_train = csv.writer(train_file, delimiter="\t")
    test_file = gzip.open('process/test.txt.gz', 'wt')
    writer_test = csv.writer(test_file, delimiter="\t")

    for k in uid2pids_timestamp_tuple.keys():
        curr = uid2pids_timestamp_tuple[k]
        n = len(curr)
        last_idx_train = int(n * train_size)
        pids_train = curr[:last_idx_train]
        for pid, rating, timestamp in pids_train:
            writer_train.writerow([k, pid, float(rating), timestamp]) 
        last_idx_valid = last_idx_train
        pids_test = curr[last_idx_valid:]
        for pid, rating, timestamp in pids_test:
            writer_test.writerow([k, pid, float(rating), timestamp])
    train_file.close()
    test_file.close()

def mapper(dataset_name):
    entity_type_id2plain_name = defaultdict(dict)
    org_datasetid2movie_title = {}
    with open('process/items.csv', 'r', encoding="latin-1") as org_movies_file:
        reader = csv.reader(org_movies_file)
        next(reader, None)
        for row in reader:
            if dataset_name == 'CELLPHONES':
                org_datasetid2movie_title[row[0]] = row[0]
            else:
                org_datasetid2movie_title[row[0]] = row[1]
    org_movies_file.close()

    dataset_id2new_id = {}
    with open('process/products.txt', 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            new_id, dataset_id = row[0], row[1]
            entity_type_id2plain_name["product"][new_id] = org_datasetid2movie_title[dataset_id]
            dataset_id2new_id[dataset_id] = new_id
    item_to_kg_file.close()

    mapping_data = []
    with gzip.open('process/mappings.txt.gz', 'wt', encoding='utf-8') as mapping_file:
        writer = csv.writer(mapping_file, delimiter="\t")
        for entity_type, entities in entity_type_id2plain_name.items():
            for entity in entities:
                name = entity_type_id2plain_name[entity_type][entity]
                name = name.split("/")[-1] if type(name) == str else str(name)
                writer.writerow([f"{entity_type}_{entity}", name])
                mapping_data.append([f"{entity_type}_{entity}", name])
    mapping_file.close()
    mapping_df = pd.DataFrame(mapping_data, columns=['product', 'name'])

    with gzip.open('process/products.txt.gz', 'wt') as product_fileo:
        writer = csv.writer(product_fileo, delimiter="\t")
        with open('process/products.txt', 'r') as product_file:
            reader = csv.reader(product_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        product_file.close()
    product_fileo.close()

    with gzip.open('process/users.txt.gz', 'wt') as users_fileo:
        writer = csv.writer(users_fileo, delimiter="\t")
        with open('process/users.txt', 'r') as users_file:
            reader = csv.reader(users_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        users_file.close()
    users_fileo.close()

    entity_type_eid2global_id = defaultdict(dict)
    ratings_pid2global_id = {}
    ratings_uid2global_id = {}
    with gzip.open('process/kg_entities.txt.gz', 'wt',  encoding='utf-8') as entity_file:
        writer = csv.writer(entity_file, delimiter="\t")
        writer.writerow(['entity_global_id', 'entity_local_id', 'entity_value'])
        global_id = 0

        with open('process/users.txt', 'r') as user_file:
            reader = csv.reader(user_file, delimiter="\t")
            next(reader, None)
            for local_id, row in enumerate(reader):
                ratings_id, old_id = row[0], row[1]
                ratings_uid2global_id[ratings_id] = global_id
                writer.writerow([global_id, f"user_{local_id}", old_id])
                global_id += 1
        user_file.close()

        ratings_pid2local_id = {}
        with open('process/products.txt', 'r', encoding='utf-8') as product_file:
            reader = csv.reader(product_file, delimiter="\t")
            next(reader, None)
            for local_id, row in enumerate(reader):
                ratings_id, old_id = row[0], row[1]
                ratings_pid2global_id[ratings_id] = global_id
                ratings_pid2local_id[ratings_id] = local_id
                entity_type_eid2global_id['product'][ratings_id] = global_id
                row_found = mapping_df.loc[mapping_df['product'] == f"product_{local_id}"]
                name_value = row_found.iloc[0]['name']
                writer.writerow([global_id, f"product_{local_id}", str(name_value)])
                global_id += 1
        product_file.close()

    with gzip.open('process/kg_relations.txt.gz', 'wt') as relations_fileo:
        writer = csv.writer(relations_fileo, delimiter="\t")
        if dataset_name == "ML1M":
            writer.writerow([0, "watched"])
            writer.writerow([1, "rev_watched"])
        elif dataset_name == "LFM1M":
            writer.writerow([0, "listened"])
            writer.writerow([1, "rev_listened"])
        else:
            writer.writerow([0, "purchase"])
            writer.writerow([1, "rev_purchase"])
    relations_fileo.close()

    with gzip.open('process/kg_triples.txt.gz', 'wt') as kg_final_file:
        writer = csv.writer(kg_final_file, delimiter="\t")
        with gzip.open('process/train.txt.gz', 'rt') as f:
            for row in f:
                fields = row.strip().split('\t')
                if len(fields) == 4:
                    uid, pid = fields[0], fields[1]
                    writer.writerow([ratings_uid2global_id[uid], 0, ratings_pid2global_id[pid]])
                    writer.writerow([ratings_pid2global_id[pid], 1, ratings_uid2global_id[uid]])
    kg_final_file.close()

    with gzip.open('process/kg_rules.txt.gz', 'wt') as kg_rules_file:
        writer = csv.writer(kg_rules_file, delimiter="\t")
        writer.writerow([0, 1, 0])
    kg_rules_file.close()

def max_path_score(lst):
    return max(lst) if lst else None

# ── User Choice Models ──────────────────────────────────────────────────────

def uniform_distribution(n):
    """Each position has equal probability."""
    return [1.0 / n] * n


def linear_decreasing_distribution(n):
    """Probability inversely proportional to rank: P(i) ∝ 1/(i+1)."""
    probabilities = [1.0 / (i + 1) for i in range(n)]
    total = sum(probabilities)
    return [p / total for p in probabilities]


def top1_distribution(n):
    """Deterministic: always pick the first item."""
    probs = [0.0] * n
    probs[0] = 1.0
    return probs


def exponential_distribution(n, lam=0.5):
    """Exponential decay: P(i) ∝ exp(-λ·i)."""
    probabilities = [math.exp(-lam * i) for i in range(n)]
    total = sum(probabilities)
    return [p / total for p in probabilities]


def cascade_distribution(n, p_click=0.3):
    """Cascade model: user scans top-down and clicks with prob p_click at each position."""
    probabilities = [p_click * ((1 - p_click) ** i) for i in range(n)]
    total = sum(probabilities)
    return [p / total for p in probabilities]


def softmax_distribution(n, tau=1.0):
    """Softmax over reversed ranks (top item has highest score). τ controls sharpness."""
    scores = [n - i for i in range(n)]
    max_s = max(scores)
    exp_scores = [math.exp((s - max_s) / tau) for s in scores]
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


USER_MODELS = {
    'UNI': uniform_distribution,
    'LIN': linear_decreasing_distribution,
    'TOP': top1_distribution,
    'EXP': exponential_distribution,
    'CBM': cascade_distribution,
    'PBM': softmax_distribution,
}


def get_user_model_distribution(name, n):
    """Return a probability vector of length n for the given user model name."""
    if name not in USER_MODELS:
        raise ValueError(f"Unknown user model '{name}'. Choose from: {list(USER_MODELS.keys())}")
    return USER_MODELS[name](n)

def print_elapsed_time(iteration, start_time, section):
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    if section in ["TRAIN TRANS-E EMBEDDING", "TRAIN AGENT", "TRAIN NEURAL SYMBOL", "EXTRACT PATHS"]:
        file_path = "../../results/utils/times.txt"
    else:
        file_path = "results/utils/times.txt"

    with open(file_path, 'a') as file:
        if minutes < 1:
            file.write(f"ITERATION {iteration} - {section}: {seconds} s\n")
        else:
            file.write(f"ITERATION {iteration} - {section}: {minutes} m {seconds} s\n")

        if section == "ALL PROCESS":
            file.write("\n")

def prepare_dataset(selected_dataset, selected_model, corrective_action, corrective_weight, selected_users, users_ratings):
    dataset_folder_path = f'dataset/{selected_dataset}/dataset'

    create_directory('process')
    create_directory('results/recommendations')
    create_directory('results/analysis')
    create_directory('results/utils')

    if selected_dataset == 'ML1M':
        users_df = pd.read_csv(f'{dataset_folder_path}/users.dat', sep="::", names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], header=None, engine='python')
        items_df = pd.read_csv(f'{dataset_folder_path}/movies.dat', sep="::", names=["ItemID", "Name", "Genres"], header=None, encoding='latin-1', engine='python')
        ratings_df = pd.read_csv(f'{dataset_folder_path}/ratings.dat', sep="::", names=["UserID", "ItemID", "Rating", "Timestamp"], header=None, engine='python')
    elif selected_dataset == 'LFM1M':
        users_df = pd.read_csv(f"{dataset_folder_path}/users.txt", sep="\t", names=["UserID", "Country", "Age", "Gender", "Play-count", "Timestamp"], engine='python')
        items_df = pd.read_csv(f"{dataset_folder_path}/tracks.txt", sep="\t", names=["ItemID", "Name", "ArtistID"], engine='python')
        ratings_df = pd.read_csv(f"{dataset_folder_path}/ratings.txt", sep="\t", names=["UserID", "ItemID", "Rating", "Timestamp"], header=0, engine='python')
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
    users_df.to_csv('process/users.csv', sep=",", header=True, index=False)
    items_df.to_csv('process/items.csv', sep=",", header=True, index=False)
    ratings_df.to_csv('process/ratings.csv', sep=",", header=True, index=False)

    # Write experiment info to times.txt
    with open('results/utils/times.txt', 'a') as file:
        file.write(f"{len(users_df)} USERS - {len(items_df)} ITEMS - {len(ratings_df)} RATINGS - ")
        file.write(f"{selected_model} - {selected_dataset} - {corrective_action} - {corrective_weight}\n")
        file.write(f"-------------------------------------------------------------------------\n")

def prepare_dataloader(dataset, iteration):
    if iteration == 1:
        write_time_based_train_test_split(0.7)
        shutil.copy('process/train.txt.gz', 'results/utils')
        shutil.copy('process/test.txt.gz', 'results/utils')
        shutil.copy('process/products.txt', 'results/utils')
        shutil.copy('process/users.txt', 'results/utils')

    mapper(dataset)


def calculate_recall(test_set, recommendation_df, top_k):
    unique_uids = recommendation_df['uid'].unique()
    recall_values = []
    for uid in unique_uids:
        uid_test_set = test_set[test_set['uid'] == uid]
        uid_items = ast.literal_eval(recommendation_df[recommendation_df['uid'] == uid]['items'].iloc[0])
        relevant_items = set(uid_test_set['pid'])
        recommended_items = set(uid_items[:top_k])
        recall = len(recommended_items & relevant_items) / len(relevant_items) if relevant_items else 0.0
        recall_values.append(round(recall * 100, 2))
    avg = round(sum(recall_values) / len(recall_values), 2) if recall_values else 0.0
    return recall_values, avg


def calculate_diversity(previous_df, current_df, top_k):
    unique_uids = previous_df['uid'].unique()
    diversity_values = []
    for uid in unique_uids:
        prev_row = previous_df[previous_df['uid'] == uid]
        curr_row = current_df[current_df['uid'] == uid]
        if prev_row.empty or curr_row.empty:
            continue
        prev_items = set(ast.literal_eval(prev_row['items'].iloc[0])[:top_k])
        curr_items = set(ast.literal_eval(curr_row['items'].iloc[0])[:top_k])
        union = len(prev_items | curr_items)
        jaccard_dist = 1 - (len(prev_items & curr_items) / union if union > 0 else 0.0)
        diversity_values.append(round(jaccard_dist * 100, 2))
    avg = round(sum(diversity_values) / len(diversity_values), 2) if diversity_values else 0.0
    return diversity_values, avg


def copy_simulation_results(selected_dataset, selected_recommender, selected_corrective_action, selected_corrective_weight, user_model):
    if selected_corrective_weight == 0.0:
        dest_folder = os.path.join("simulation_results", selected_dataset, selected_recommender, user_model, "Baseline")
    else:
        dest_folder = os.path.join("simulation_results", selected_dataset, selected_recommender, user_model, f"{selected_corrective_action}_{selected_corrective_weight}")

    parent_dir = os.path.dirname(dest_folder)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    shutil.copytree("results", dest_folder)
