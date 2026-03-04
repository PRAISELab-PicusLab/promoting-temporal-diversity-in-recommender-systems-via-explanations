from utils import *
warnings.filterwarnings("ignore", category=FutureWarning)

def write_inter_file(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        df.to_csv(f, index=False, header=False, sep="\t")

def write_item_file(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("item_id:token\tname:token\n")
        df.to_csv(f, index=False, header=False, sep="\t")

def prepare_recbole_files(train_path="process/preprocessed/model/train.txt.gz", items_path="process/csv/items.csv", out_dir="process/data", items_has_header=True,):
    df = pd.read_csv(train_path, compression="gzip" if train_path.endswith(".gz") else None, sep="\t", names=["user_id", "item_id", "rating", "timestamp"], low_memory=False)
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["rating", "timestamp"])

    header = 0 if items_has_header else None
    df_item = pd.read_csv(items_path, names=["item_id", "name", "genres"], header=header, low_memory=False)

    df_item = df_item.dropna(subset=["item_id", "name"])
    df_item["item_id"] = df_item["item_id"].astype(str).str.strip()
    df_item["name"] = df_item["name"].astype(str).str.replace("\t", " ").str.replace("\n", " ")
    df_item = df_item[["item_id", "name"]]
    df_item = df_item[df_item["item_id"].isin(df["item_id"])]

    os.makedirs(out_dir, exist_ok=True)
    write_inter_file(df, os.path.join(out_dir, "data.inter"))
    write_item_file(df_item, os.path.join(out_dir, "data.item"))

def build_config(recommender: str, epochs: int = 30):
    return Config(
        model=recommender,
        dataset="data",
        config_dict={
            "data_path": "process",
            "field_separator": "\t",
            "load_col": {
                "inter": ["user_id", "item_id", "rating", "timestamp"],
                "item": ["item_id", "name"],
            },
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating",
            "TIME_FIELD": "timestamp",
            "epochs": epochs,

            "user_inter_num_interval": "[0,inf)",
            "item_inter_num_interval": "[0,inf)",
        },
    )

def train_and_recommend(recommender: str, topk: int = 100, epochs: int = 30):
    config = build_config(recommender=recommender, epochs=epochs)
    init_seed(config["seed"], config["reproducibility"])

    dataset_obj = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    model_class = get_model(recommender)
    model = model_class(config, train_data.dataset).to(config["device"])

    trainer = Trainer(config, model)
    trainer.fit(train_data)

    if os.path.isdir("log_tensorboard"):
        shutil.rmtree("log_tensorboard", ignore_errors=True)

    ds = train_data.dataset
    uid_field = ds.uid_field
    iid_field = ds.iid_field

    device = model.device if hasattr(model, "device") else torch.device("cpu")
    model.to(device)
    model.eval()

    internal_uids = list(range(ds.user_num))[1:]  # skip padding 0
    raw_user_ids = ds.id2token(uid_field, internal_uids)

    inter_users = ds.inter_feat[uid_field].cpu().numpy()
    inter_items = ds.inter_feat[iid_field].cpu().numpy()

    recs = []
    for internal_uid, raw_uid in tqdm(
        list(zip(internal_uids, raw_user_ids)),
        total=len(internal_uids),
        desc="Generating recommendations",
    ):
        seen_mask = (inter_users == internal_uid)
        seen_items = set(inter_items[seen_mask])

        user_tensor = torch.tensor([internal_uid], dtype=torch.long, device=device)
        interaction = Interaction({uid_field: user_tensor}).to(device)

        with torch.no_grad():
            try:
                scores = model.full_sort_predict(interaction).view(-1)
            except NotImplementedError:
                n_items = ds.item_num
                item_tensor = torch.arange(n_items, dtype=torch.long, device=device)
                user_rep = user_tensor.repeat(n_items)
                inter_all = Interaction({uid_field: user_rep, iid_field: item_tensor}).to(device)
                scores = model.predict(inter_all).view(-1)

            scores_np = scores.detach().cpu().numpy()

        sorted_idx = np.argsort(-scores_np)

        top_item_ids = []
        top_scores = []

        for item_internal in sorted_idx:
            if item_internal == 0:
                continue
            if item_internal in seen_items:
                continue
            top_item_ids.append(int(item_internal))
            top_scores.append(float(scores_np[item_internal]))
            if len(top_item_ids) >= topk:
                break

        raw_item_ids = ds.id2token(iid_field, top_item_ids)

        for item_id, s in zip(raw_item_ids, top_scores):
            recs.append({"user_id": raw_uid, "item_id": item_id, "score": s})

    recommendations_df = pd.DataFrame(recs, columns=["user_id", "item_id", "score"])
    return recommendations_df


def recbole(dataset_name: str, recommender: str):
    prepare_recbole_files(train_path="process/preprocessed/model/train.txt.gz", items_path="process/csv/items.csv",out_dir="process/data", items_has_header=True,)
    recommendations_df = train_and_recommend(recommender=recommender, topk=100, epochs=30)

    out_path = "process/recommendations.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    recommendations_df.to_csv(out_path, index=False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ML1M", help="Dataset name (informativo)")
    parser.add_argument("--recommender", type=str, default="NeuMF", help="RecBole model name, es. NeuMF, ItemKNN")
    args = parser.parse_args()

    recbole(args.dataset, args.recommender)


if __name__ == "__main__":
    main()