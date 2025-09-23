from utils import *


def propagate_item_removal_to_kg(items_df, items_to_kg_df, entities_df, kg_df, dataset):
    # Filter items differently depending on dataset format
    if dataset == 'ml1m':
        items_to_kg_df_after = items_to_kg_df[items_to_kg_df.dataset_id.isin(items_df.movie_id)]
        removed_items = items_to_kg_df[~items_to_kg_df.dataset_id.isin(items_to_kg_df_after.dataset_id)]
    elif dataset == 'lfm1m':
        items_to_kg_df_after = items_to_kg_df[items_to_kg_df["track-id"].isin(items_df["track-id"])]
        removed_items = items_to_kg_df[~items_to_kg_df["track-id"].isin(items_to_kg_df_after["track-id"])]

    # Remove entities corresponding to removed items
    if dataset == 'ml1m':
        removed_entities = entities_df[entities_df.entity_url.isin(removed_items.entity_url)]
        entities_df = entities_df[~entities_df.entity_url.isin(removed_items.entity_url)]
    elif dataset == 'lfm1m':
        removed_entities = entities_df[entities_df["raw_dataset_id"].isin(removed_items["entity_id"])]
        entities_df = entities_df[~entities_df["raw_dataset_id"].isin(removed_items["entity_id"])]

    # Remove KG triplets that involve removed entities
    if dataset == 'ml1m':
        kg_df = kg_df[~kg_df.entity_head.isin(removed_entities.entity_id)]
    elif dataset == 'lfm1m':
        kg_df = kg_df[~kg_df.entity_head.isin(removed_entities["raw_dataset_id"])]

    return items_to_kg_df_after, entities_df, kg_df


def discard_entity_with_lt_th(entities_list, th):
    return [k for k, v in Counter(entities_list).items() if v >= th]


def discard_k_letter_categories(entities_list, k):
    return [x for x in entities_list if len(x) > k]


def entity2plain_text(dataset, method):
    entity2plain_text_map = defaultdict(dict)

    if method == "cafe":
        with gzip.open(f"data/{dataset}/preprocessed/{method}/kg_entities.txt.gz", 'rt') as entities_file:
            reader = csv.reader(entities_file, delimiter="\t")
            next(reader, None)  # skip header
            for row in reader:
                row[1] = row[1].split("_")
                entity_type, local_id = '_'.join(row[1][:-1]), row[1][-1]
                entity2plain_text_map[entity_type][int(local_id)] = row[-1]

    elif method == "pgpr":
        with gzip.open(f"data/{dataset}/preprocessed/{method}/mappings.txt.gz", 'rt') as entities_file:
            reader = csv.reader(entities_file, delimiter="\t")
            # Remove empty rows
            reader = [sublist for sublist in reader if sublist]
            for row in reader:
                row[0] = row[0].split("_")
                entity_type, local_id = '_'.join(row[0][:-1]), row[0][-1]
                entity2plain_text_map[entity_type][int(local_id)] = row[-1]

    return entity2plain_text_map


def create_kg_from_metadata(dataset):
    input_data = f'data/{dataset}/preprocessed'
    input_kg = f'data/{dataset}/kg'

    # Load metadata JSON into DataFrame
    metaproduct_df = getDF(input_kg + '/meta_Cell_Phones_and_Accessories.json.gz')
    metaproduct_df = metaproduct_df.drop(
        ['tech1', 'description', 'fit', 'title', 'tech2', 'feature',
         'rank', 'details', 'similar_item', 'date', 'price',
         'imageURL', 'imageURLHighRes'],
        axis=1
    )

    # Keep only products present in dataset
    valid_products = set()
    with open(input_data + '/products.txt', 'r') as products_file:
        reader = csv.reader(products_file, delimiter="\t")
        for row in reader:
            _, dataset_asin = row
            valid_products.add(dataset_asin)
    metaproduct_df = metaproduct_df[metaproduct_df.asin.isin(valid_products)]

    # -----------------------------
    # i2kg_map.txt: item -> KG entity ID
    # -----------------------------
    products_id = metaproduct_df['asin'].unique()
    product_id2new_id = {}
    entities = {}
    with open(input_data + "/i2kg_map.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["entity_id", "entity_url"])
        for new_id, pid in enumerate(products_id):
            product_id2new_id[pid] = new_id
            entities[pid] = new_id
            writer.writerow([new_id, pid])

    # -----------------------------
    # r_map.txt: relation name -> ID
    # -----------------------------
    columns = list(metaproduct_df.columns)
    columns.remove('asin')
    columns.remove('main_cat')

    relation_name2id = {}
    with open(input_data + "/r_map.txt", "w+") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["relation_id", "relation_url"])
        new_rid = 0
        for relation in columns:
            if relation in ["also_buy", "also_view"]:
                # Split into related_product and product relations
                relation_related_product = relation + "_related_product"
                writer.writerow([new_rid, relation_related_product])
                relation_name2id[relation_related_product] = new_rid
                new_rid += 1

                relation_product = relation + "_product"
                writer.writerow([new_rid, relation_product])
                relation_name2id[relation_product] = new_rid
                new_rid += 1
            else:
                writer.writerow([new_rid, relation])
                relation_name2id[relation] = new_rid
                new_rid += 1

    # -----------------------------
    # Build KG triplets
    # -----------------------------
    entity_names = set()
    for col in columns:
        if col == 'also_view':
            entity_names.update(['related_product', 'also_view_product'])
        elif col == 'also_buy':
            entity_names.add('also_buy_product')
        else:
            entity_names.add(col)

    last_id = len(entities)
    triplets = []

    for entity_name in entity_names:
        for _, row in metaproduct_df.iterrows():
            pid = row['asin']

            if entity_name in ['also_buy_product', 'also_view_product']:
                # Relations to other products in the catalog
                relation = '_'.join(entity_name.split("_")[:2])
                related_products_in_catalog = [
                    rp for rp in row[relation] if rp in product_id2new_id
                ]
                for product in related_products_in_catalog:
                    triplets.append([entities[pid], entities[product], relation_name2id[entity_name]])

            elif entity_name == 'related_product':
                # Relations to products not in the catalog
                for relation in ['also_buy', 'also_view']:
                    related_products_not_in_catalog = [
                        rp for rp in row[relation] if rp not in product_id2new_id
                    ]
                    for related_product in related_products_not_in_catalog:
                        entities[related_product] = last_id
                        triplets.append([entities[pid], entities[related_product], relation_name2id[relation + "_related_product"]])
                        last_id += 1

            else:
                # Attributes like brand, category, etc.
                curr_attributes = row[entity_name]
                if curr_attributes == "":
                    continue
                if isinstance(curr_attributes, list):
                    valid_entities = [value for value in curr_attributes if value not in entities]
                    for entity in valid_entities:
                        entities[entity] = last_id
                        triplets.append([entities[pid], entities[entity], relation_name2id[entity_name]])
                        last_id += 1
                else:
                    if curr_attributes not in entities:
                        entities[curr_attributes] = last_id
                        triplets.append([entities[pid], entities[curr_attributes], relation_name2id[entity_name]])
                        last_id += 1

    # -----------------------------
    # Save entity map (e_map.txt)
    # -----------------------------
    with open(input_data + "/e_map.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["entity_id", "entity_url"])
        for entity_id, new_id in entities.items():
            writer.writerow([new_id, entity_id])

    # -----------------------------
    # Save KG triplets (kg_final.txt)
    # -----------------------------
    with open(input_data + "/kg_final.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["entity_head", "entity_tail", "relation"])
        for e_h, e_t, r in triplets:
            writer.writerow([e_h, e_t, r])
