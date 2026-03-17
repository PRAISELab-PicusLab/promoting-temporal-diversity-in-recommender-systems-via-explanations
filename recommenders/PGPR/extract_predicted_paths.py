import csv
from tqdm import tqdm


def save_best_pred_paths(folder_path, best_pred_paths):
    # Get min and max score to performe normalization between 0 and 1
    score_list = []
    for uid, pid_list in best_pred_paths.items():
        for path in pid_list:
            score_list.append(path[0])
    min_score = min(score_list)
    max_score = max(score_list)

    with open(folder_path + "/all_pred_paths.csv", 'w+', newline='') as best_pred_paths_file:
        header = ["uid", "rec item", "path_score", "path_prob", "path"]
        writer = csv.writer(best_pred_paths_file)
        writer.writerow(header)
        ENTITY_ABBREV = {
            'user': 'U', 'product': 'I', 'song': 'I',
            'actor': 'A', 'artist': 'A', 'featured_artist': 'A',
            'director': 'D', 'composer': 'C', 'category': 'Ca',
            'production_company': 'P', 'producer': 'Pr', 'editor': 'Ed',
            'writter': 'W', 'cinematographer': 'Ci', 'country': 'Co',
            'related_product': 'RP', 'engineer': 'E', 'brand': 'B', 'wikipage': 'WP',
        }
        for uid, recs in tqdm(best_pred_paths.items(), desc="    Saving paths"):
            for rec in recs:
                path_score = str((rec[0] - min_score) / (max_score - min_score))
                path_prob = rec[1]
                recommended_item_id = rec[2][-1][-1]
                path_explaination = ' '.join(
                    f"{ENTITY_ABBREV.get(tup[1], tup[1][0].upper())} {tup[2]}"
                    for tup in rec[2]
                )
                writer.writerow([uid, recommended_item_id, path_score, path_prob, path_explaination])
    best_pred_paths_file.close()
