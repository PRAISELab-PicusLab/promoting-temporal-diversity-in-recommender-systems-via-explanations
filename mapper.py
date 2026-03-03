from collections import defaultdict

import csv
import os
import gzip
import time
import argparse
import pandas as pd

ML1M = 'ML1M'
LFM1M = 'LFM1M'
CELL = 'CELLPHONES'

def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)

"""
ENTITIES
"""
#ML1M ENTITIES
MOVIE = 'movie'
ACTOR = 'actor'
DIRECTOR = 'director'
PRODUCTION_COMPANY = 'production_company'
EDITOR = 'editor'
WRITTER = 'writter'
CINEMATOGRAPHER = 'cinematographer'
COMPOSER = 'composer'
COUNTRY = 'country'
AWARD = 'award'

#LASTFM ENTITIES
SONG = 'song'
ARTIST = 'artist'
ENGINEER = 'engineer'
PRODUCER = 'producer'

#COMMON ENTITIES
USER = 'user'
CATEGORY = 'category'
PRODUCT = 'product'

RELATION_LIST = {
    ML1M: {
        0: "http://dbpedia.org/ontology/cinematography",
        1: "http://dbpedia.org/property/productionCompanies",
        2: "http://dbpedia.org/property/composer",
        3: "http://purl.org/dc/terms/subject",
        4: "http://dbpedia.org/ontology/openingFilm",
        5: "http://www.w3.org/2000/01/rdf-schema",
        6: "http://dbpedia.org/property/story",
        7: "http://dbpedia.org/ontology/series",
        8: "http://www.w3.org/1999/02/22-rdf-syntax-ns",
        9: "http://dbpedia.org/ontology/basedOn",
        10: "http://dbpedia.org/ontology/starring",
        11: "http://dbpedia.org/ontology/country",
        12: "http://dbpedia.org/ontology/wikiPageWikiLink",
        13: "http://purl.org/linguistics/gold/hypernym",
        14: "http://dbpedia.org/ontology/editing",
        15: "http://dbpedia.org/property/producers",
        16: "http://dbpedia.org/property/allWriting",
        17: "http://dbpedia.org/property/notableWork",
        18: "http://dbpedia.org/ontology/director",
        19: "http://dbpedia.org/ontology/award",
    },
    LFM1M: {
        0: "http://rdf.freebase.com/ns/common.topic.notable_types",
        1: "http://rdf.freebase.com/ns/music.recording.releases",
        2: "http://rdf.freebase.com/ns/music.recording.artist",
        3: "http://rdf.freebase.com/ns/music.recording.engineer",
        4: "http://rdf.freebase.com/ns/music.recording.producer",
        5: "http://rdf.freebase.com/ns/music.recording.canonical_version",
        6: "http://rdf.freebase.com/ns/music.recording.song",
        7: "http://rdf.freebase.com/ns/music.single.versions",
        8: "http://rdf.freebase.com/ns/music.recording.featured_artists",
    },
    CELL: {
        0: "belong_to",
        1: "also_buy_related_product",
        2: "also_buy_product",
        3: "produced_by_company",
        4: "also_view_related_product",
        5: "also_view_product",
    }
}

relation_name2entity_name = {
    ML1M: {
            "cinematographer_p_ci": 'cinematographer',
            "production_company_p_pr" :'production_company',
            "composer_p_co":'composer',
            "category_p_ca":'category',
            "actor_p_ac":'actor',
            "country_p_co":'country',
            "wikipage_p_wi":'wikipage',
            "editor_p_ed":'editor',
            "producer_p_pr":'producer',
            "writter_p_wr": 'writter',
            "director_p_di":'director',
        },
    LFM1M: {
        "category_p_ca": "category",
        "related_product_p_re": "related_product",
        "artist_p_ar": "artist",
        "engineer_p_en": "engineer",
        "producer_p_pr": "producer",
        "featured_artist_p_fe": "featured_artist",
    },
    CELL: {
        "category_p_ca": "category",
        "also_buy_related_product_p_re": "related_product",
        "also_buy_product_p_pr": "product",
        "brand_p_br": "brand",
        "also_view_related_product_p_re": "related_product",
        "also_view_product_p_pr": "product",
    }

}

relation_to_entity = {
    ML1M: {
        "http://dbpedia.org/ontology/cinematography": 'cinematographer',
        "http://dbpedia.org/property/productionCompanies": 'production_company',
        "http://dbpedia.org/property/composer": 'composer',
        "http://purl.org/dc/terms/subject": 'category',
        "http://dbpedia.org/ontology/starring": 'actor',
        "http://dbpedia.org/ontology/country": 'country',
        "http://dbpedia.org/ontology/wikiPageWikiLink": 'wikipage',
        "http://dbpedia.org/ontology/editing": 'editor',
        "http://dbpedia.org/property/producers": 'producer',
        "http://dbpedia.org/property/allWriting": 'writter',
        "http://dbpedia.org/ontology/director": 'director',
    },
    LFM1M: {
        "http://rdf.freebase.com/ns/common.topic.notable_types": "category",
        "http://rdf.freebase.com/ns/music.recording.releases": "related_product",
        "http://rdf.freebase.com/ns/music.recording.artist": "artist",
        "http://rdf.freebase.com/ns/music.recording.engineer": "engineer",
        "http://rdf.freebase.com/ns/music.recording.producer": "producer",
        "http://rdf.freebase.com/ns/music.recording.featured_artists": "featured_artist",
    },
    CELL: {
        "category": "category",
        "also_buy_related_product": "related_product",
        "also_buy_product": "product",
        "brand": "brand",
        "also_view_product": "product",
        "also_view_related_product": "related_product",
    }
}

relation_id2plain_name = {
    ML1M: {
        "0" : "cinematography_by",
        "1" : "produced_by_company",
        "2" : "composed_by",
        "3" : "belong_to",
        "10": "starred_by",
        "11": "produced_in",
        "12": "related_to",
        "14": "edited_by",
        "15": "produced_by_producer",
        "16": "wrote_by",
        "18": "directed_by",
    },
    LFM1M: {
        "0": "category",
        "1": "related_product",
        "2": "artist",
        "3": "engineer",
        "4": "producer",
        "5": "featured_artist",
    },
    CELL: {
        "0": "category",
        "1": "also_buy_related_product",
        "2": "related_product",
        "3": "brand",
        "4": "also_view_related_product",
        "5": "related_product"
    }
}


def write_time_based_train_test_split(dataset_name, train_size, ratings_pid2local_id = {}, ratings_uid2global_id = {}):
    input_folder = 'process/preprocessed/'
    output_folder = 'process/preprocessed/model/'
    ensure_dir(output_folder)

    uid2pids_timestamp_tuple = defaultdict(list)
    with open(input_folder + 'ratings.txt', 'r') as ratings_file: #uid	pid	rating	timestamp
        reader = csv.reader(ratings_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            k, pid, rating, timestamp = row
            uid2pids_timestamp_tuple[k].append([pid, rating, int(timestamp)]) #DR
    ratings_file.close()

    train_file = gzip.open(output_folder + 'train.txt.gz', 'wt')
    writer_train = csv.writer(train_file, delimiter="\t")
    test_file = gzip.open(output_folder + 'test.txt.gz', 'wt')
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
    process_folder_path = f'process'
    input_folder = f'{process_folder_path}/preprocessed/'
    output_folder = f'{process_folder_path}/preprocessed/model/'
    ensure_dir(output_folder)

    entity_type_id2plain_name = defaultdict(dict)
    org_datasetid2movie_title = {}
    with open(f'{process_folder_path}/csv/items.csv', 'r', encoding="latin-1") as org_movies_file:
        reader = csv.reader(org_movies_file)
        next(reader, None)
        for row in reader:
            if dataset_name == 'CELLPHONES':
                org_datasetid2movie_title[row[0]] = row[0]
            else:
                org_datasetid2movie_title[row[0]] = row[1]
    org_movies_file.close()

    dataset_id2new_id = {}
    with open(input_folder + "products.txt", 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            new_id, dataset_id = row[0], row[1]
            entity_type_id2plain_name["product"][new_id] = org_datasetid2movie_title[dataset_id]
            dataset_id2new_id[dataset_id] = new_id
    item_to_kg_file.close()

    mapping_data = []
    with gzip.open(output_folder + 'mappings.txt.gz', 'wt', encoding='utf-8') as mapping_file:
        writer = csv.writer(mapping_file, delimiter="\t")
        for entity_type, entities in entity_type_id2plain_name.items():
            for entity in entities:
                name = entity_type_id2plain_name[entity_type][entity]
                name = name.split("/")[-1] if type(name) == str else str(name)
                writer.writerow([f"{entity_type}_{entity}", name])
                mapping_data.append([f"{entity_type}_{entity}", name])
    mapping_file.close()
    mapping_df = pd.DataFrame(mapping_data, columns=['product', 'name'])

    with gzip.open(output_folder + 'products.txt.gz', 'wt') as product_fileo:
        writer = csv.writer(product_fileo, delimiter="\t")
        with open(input_folder + 'products.txt', 'r') as product_file:
            reader = csv.reader(product_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        product_file.close()
    product_fileo.close()

    with gzip.open(output_folder + 'users.txt.gz', 'wt') as users_fileo:
        writer = csv.writer(users_fileo, delimiter="\t")
        with open(input_folder + 'users.txt', 'r') as users_file:
            reader = csv.reader(users_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        users_file.close()
    users_fileo.close()

    entity_type_eid2global_id = defaultdict(dict)
    ratings_pid2global_id = {}
    ratings_uid2global_id = {}
    with gzip.open(output_folder + 'kg_entities.txt.gz', 'wt',  encoding='utf-8') as entity_file:
        writer = csv.writer(entity_file, delimiter="\t")
        writer.writerow(['entity_global_id', 'entity_local_id', 'entity_value'])
        global_id = 0

        with open(input_folder + 'users.txt', 'r') as user_file:
            reader = csv.reader(user_file, delimiter="\t")
            next(reader, None)
            for local_id, row in enumerate(reader):
                ratings_id, old_id = row[0], row[1]
                ratings_uid2global_id[ratings_id] = global_id
                writer.writerow([global_id, f"user_{local_id}", old_id])
                global_id += 1
        user_file.close()

        ratings_pid2local_id = {}
        with open(input_folder + 'products.txt', 'r', encoding='utf-8') as product_file:
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

    with gzip.open(output_folder + 'kg_relations.txt.gz', 'wt') as relations_fileo:
        writer = csv.writer(relations_fileo, delimiter="\t")
        if dataset_name == ML1M:
            writer.writerow([0, "watched"])
            writer.writerow([1, "rev_watched"])
        elif dataset_name == LFM1M:
            writer.writerow([0, "listened"])
            writer.writerow([1, "rev_listened"])
        else:
            writer.writerow([0, "purchase"])
            writer.writerow([1, "rev_purchase"])
    relations_fileo.close()

    with gzip.open(output_folder + 'kg_triples.txt.gz', 'wt') as kg_final_file:
        writer = csv.writer(kg_final_file, delimiter="\t")
        with gzip.open(output_folder + 'train.txt.gz', 'rt') as f:
            for row in f:
                fields = row.strip().split('\t')
                if len(fields) == 4:
                    uid, pid = fields[0], fields[1]
                    writer.writerow([ratings_uid2global_id[uid], 0, ratings_pid2global_id[pid]])
                    writer.writerow([ratings_pid2global_id[pid], 1, ratings_uid2global_id[uid]])
    kg_final_file.close()

    with gzip.open(output_folder + 'kg_rules.txt.gz', 'wt') as kg_rules_file:
        writer = csv.writer(kg_rules_file, delimiter="\t")
        writer.writerow([0, 1, 0])
    kg_rules_file.close()
