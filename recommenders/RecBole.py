from utils import *
warnings.filterwarnings("ignore", category=FutureWarning)

def recbole(recommender: str, iteration: int, corrective_weight: float = 0.5, topk: int = 100, epochs: int = 30):
    train_path = "process/train.txt.gz"
    items_path = "process/items.csv"
    out_dir    = "process/data"

    df = pd.read_csv(train_path, compression="gzip", sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"], low_memory=False)
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    df["user_id"]   = df["user_id"].astype(str).str.strip()
    df["item_id"]   = df["item_id"].astype(str).str.strip()
    df["rating"]    = pd.to_numeric(df["rating"],    errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["rating", "timestamp"])

    df_item = pd.read_csv(items_path, names=["item_id", "name", "genres"], header=0, low_memory=False)
    df_item = df_item.dropna(subset=["item_id", "name"])
    df_item["item_id"] = df_item["item_id"].astype(str).str.strip()
    df_item["name"]    = df_item["name"].astype(str).str.replace("\t", " ").str.replace("\n", " ")
    df_item = df_item[["item_id", "name"]]
    df_item = df_item[df_item["item_id"].isin(df["item_id"])]

    os.makedirs(out_dir, exist_ok=True)
    inter_path = os.path.join(out_dir, "data.inter")
    with open(inter_path, "w", encoding="utf-8", newline="") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        df.to_csv(f, index=False, header=False, sep="\t")
    item_path = os.path.join(out_dir, "data.item")
    with open(item_path, "w", encoding="utf-8", newline="") as f:
        f.write("item_id:token\tname:token\n")
        df_item.to_csv(f, index=False, header=False, sep="\t")

    import sys
    sys.argv = sys.argv[:1]
    config = Config(
        model=recommender,
        dataset="data",
        config_dict={
            "data_path": "process",
            "field_separator": "\t",
            "load_col": {
                "inter": ["user_id", "item_id", "rating", "timestamp"],
                "item":  ["item_id", "name"],
            },
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD":  "rating",
            "TIME_FIELD":    "timestamp",
            "epochs": epochs,
            "user_inter_num_interval": "[0,inf)",
            "item_inter_num_interval": "[0,inf)",
        },
    )

    init_seed(config["seed"], config["reproducibility"])
    dataset_obj = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    model_class = get_model(recommender)
    model = model_class(config, train_data.dataset).to(config["device"])
    Trainer(config, model).fit(train_data)

    if os.path.isdir("log_tensorboard"):
        shutil.rmtree("log_tensorboard", ignore_errors=True)

    ds        = train_data.dataset
    uid_field = ds.uid_field
    iid_field = ds.iid_field
    device    = model.device if hasattr(model, "device") else torch.device("cpu")
    model.to(device)
    model.eval()

    internal_uids = list(range(ds.user_num))[1:]
    raw_user_ids  = ds.id2token(uid_field, internal_uids)
    inter_users   = ds.inter_feat[uid_field].cpu().numpy()
    inter_items   = ds.inter_feat[iid_field].cpu().numpy()

    recs = []
    for internal_uid, raw_uid in tqdm(list(zip(internal_uids, raw_user_ids)), total=len(internal_uids), desc="Generating recommendations"):
        seen_items  = set(inter_items[inter_users == internal_uid])
        user_tensor = torch.tensor([internal_uid], dtype=torch.long, device=device)
        interaction = Interaction({uid_field: user_tensor}).to(device)

        with torch.no_grad():
            try:
                scores = model.full_sort_predict(interaction).view(-1)
            except NotImplementedError:
                n_items  = ds.item_num
                item_tensor = torch.arange(n_items, dtype=torch.long, device=device)
                inter_all   = Interaction({uid_field: user_tensor.repeat(n_items), iid_field: item_tensor}).to(device)
                scores = model.predict(inter_all).view(-1)
            scores_np = scores.detach().cpu().numpy()

        top_ids, top_scores = [], []
        for idx in np.argsort(-scores_np):
            if idx == 0 or idx in seen_items:
                continue
            top_ids.append(int(idx))
            top_scores.append(float(scores_np[idx]))
            if len(top_ids) >= topk:
                break

        for item_id, s in zip(ds.id2token(iid_field, top_ids), top_scores):
            recs.append({"user_id": raw_uid, "item_id": item_id, "score": s})

    rec_path = "process/recommendations.csv"
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    pd.DataFrame(recs, columns=["user_id", "item_id", "score"]).to_csv(rec_path, index=False)

    compute_explanations(corrective_weight, rec_path=rec_path, train_path=train_path)

    # ── 4. Save in ERS-compatible format ─────────────────────────────────────
    final_df = pd.read_csv(rec_path)
    rename_map = {
        "UserID": "uid",
        "user_id": "uid",
        "ItemID": "item",
        "item_id": "item",
        "Score": "score",
        "Paths": "paths",
    }
    final_df = final_df.rename(columns=rename_map)

    if "paths" not in final_df.columns:
        final_df["paths"] = [[] for _ in range(len(final_df))]

    required_cols = ["uid", "item", "score", "paths"]
    missing_cols = [c for c in required_cols if c not in final_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required recommendation columns: {missing_cols}. "
            f"Available columns: {list(final_df.columns)}"
        )

    os.makedirs("results/recommendations", exist_ok=True)
    final_df.to_csv(f"results/recommendations/iteration_{iteration}.csv", columns=["uid", "item", "score", "paths"], sep=",", index=False)

    if os.path.isdir("saved"):
        shutil.rmtree("saved", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommender",       type=str,   default="NeuMF", help="RecBole model name")
    parser.add_argument("--iteration",         type=int,   default=1,       help="Current iteration index")
    parser.add_argument("--corrective_weight", type=float, default=0.5,     help="Trade-off weight for explanations")
    args = parser.parse_args()

    recbole(args.recommender, args.iteration, args.corrective_weight)

if __name__ == "__main__":
    main()