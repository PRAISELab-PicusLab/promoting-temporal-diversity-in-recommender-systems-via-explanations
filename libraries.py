from tqdm import tqdm
from collections import defaultdict, Counter

import os
import re
import ast
import csv
import math
import gzip
import json
import time
import torch
import pickle
import random
import shutil
import argparse
import datetime
import subprocess

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from recbole.config import Config
from recbole.trainer import Trainer
from recbole.utils import init_seed, get_model
from recbole.data.interaction import Interaction
from recbole.data import create_dataset, data_preparation

# ML1M = 'ML1M'
# LFM1M = 'LFM1M'
# CELL = 'CELLPHONES'


# """
# ENTITIES
# """
# #ML1M ENTITIES
# MOVIE = 'movie'
# ACTOR = 'actor'
# DIRECTOR = 'director'
# PRODUCTION_COMPANY = 'production_company'
# EDITOR = 'editor'
# WRITTER = 'writter'
# CINEMATOGRAPHER = 'cinematographer'
# COMPOSER = 'composer'
# COUNTRY = 'country'
# AWARD = 'award'

# #LASTFM ENTITIES
# SONG = 'song'
# ARTIST = 'artist'
# ENGINEER = 'engineer'
# PRODUCER = 'producer'

# #COMMON ENTITIES
# USER = 'user'
# CATEGORY = 'category'
# PRODUCT = 'product'

# RELATION_LIST = {
#     ML1M: {
#         0: "http://dbpedia.org/ontology/cinematography",
#         1: "http://dbpedia.org/property/productionCompanies",
#         2: "http://dbpedia.org/property/composer",
#         3: "http://purl.org/dc/terms/subject",
#         4: "http://dbpedia.org/ontology/openingFilm",
#         5: "http://www.w3.org/2000/01/rdf-schema",
#         6: "http://dbpedia.org/property/story",
#         7: "http://dbpedia.org/ontology/series",
#         8: "http://www.w3.org/1999/02/22-rdf-syntax-ns",
#         9: "http://dbpedia.org/ontology/basedOn",
#         10: "http://dbpedia.org/ontology/starring",
#         11: "http://dbpedia.org/ontology/country",
#         12: "http://dbpedia.org/ontology/wikiPageWikiLink",
#         13: "http://purl.org/linguistics/gold/hypernym",
#         14: "http://dbpedia.org/ontology/editing",
#         15: "http://dbpedia.org/property/producers",
#         16: "http://dbpedia.org/property/allWriting",
#         17: "http://dbpedia.org/property/notableWork",
#         18: "http://dbpedia.org/ontology/director",
#         19: "http://dbpedia.org/ontology/award",
#     },
#     LFM1M: {
#         0: "http://rdf.freebase.com/ns/common.topic.notable_types",
#         1: "http://rdf.freebase.com/ns/music.recording.releases",
#         2: "http://rdf.freebase.com/ns/music.recording.artist",
#         3: "http://rdf.freebase.com/ns/music.recording.engineer",
#         4: "http://rdf.freebase.com/ns/music.recording.producer",
#         5: "http://rdf.freebase.com/ns/music.recording.canonical_version",
#         6: "http://rdf.freebase.com/ns/music.recording.song",
#         7: "http://rdf.freebase.com/ns/music.single.versions",
#         8: "http://rdf.freebase.com/ns/music.recording.featured_artists",
#     },
#     CELL: {
#         0: "belong_to",
#         1: "also_buy_related_product",
#         2: "also_buy_product",
#         3: "produced_by_company",
#         4: "also_view_related_product",
#         5: "also_view_product",
#     }
# }

# relation_name2entity_name = {
#     ML1M: {
#             "cinematographer_p_ci": 'cinematographer',
#             "production_company_p_pr" :'production_company',
#             "composer_p_co":'composer',
#             "category_p_ca":'category',
#             "actor_p_ac":'actor',
#             "country_p_co":'country',
#             "wikipage_p_wi":'wikipage',
#             "editor_p_ed":'editor',
#             "producer_p_pr":'producer',
#             "writter_p_wr": 'writter',
#             "director_p_di":'director',
#         },
#     LFM1M: {
#         "category_p_ca": "category",
#         "related_product_p_re": "related_product",
#         "artist_p_ar": "artist",
#         "engineer_p_en": "engineer",
#         "producer_p_pr": "producer",
#         "featured_artist_p_fe": "featured_artist",
#     },
#     CELL: {
#         "category_p_ca": "category",
#         "also_buy_related_product_p_re": "related_product",
#         "also_buy_product_p_pr": "product",
#         "brand_p_br": "brand",
#         "also_view_related_product_p_re": "related_product",
#         "also_view_product_p_pr": "product",
#     }

# }

# relation_to_entity = {
#     ML1M: {
#         "http://dbpedia.org/ontology/cinematography": 'cinematographer',
#         "http://dbpedia.org/property/productionCompanies": 'production_company',
#         "http://dbpedia.org/property/composer": 'composer',
#         "http://purl.org/dc/terms/subject": 'category',
#         "http://dbpedia.org/ontology/starring": 'actor',
#         "http://dbpedia.org/ontology/country": 'country',
#         "http://dbpedia.org/ontology/wikiPageWikiLink": 'wikipage',
#         "http://dbpedia.org/ontology/editing": 'editor',
#         "http://dbpedia.org/property/producers": 'producer',
#         "http://dbpedia.org/property/allWriting": 'writter',
#         "http://dbpedia.org/ontology/director": 'director',
#     },
#     LFM1M: {
#         "http://rdf.freebase.com/ns/common.topic.notable_types": "category",
#         "http://rdf.freebase.com/ns/music.recording.releases": "related_product",
#         "http://rdf.freebase.com/ns/music.recording.artist": "artist",
#         "http://rdf.freebase.com/ns/music.recording.engineer": "engineer",
#         "http://rdf.freebase.com/ns/music.recording.producer": "producer",
#         "http://rdf.freebase.com/ns/music.recording.featured_artists": "featured_artist",
#     },
#     CELL: {
#         "category": "category",
#         "also_buy_related_product": "related_product",
#         "also_buy_product": "product",
#         "brand": "brand",
#         "also_view_product": "product",
#         "also_view_related_product": "related_product",
#     }
# }

# relation_id2plain_name = {
#     ML1M: {
#         "0" : "cinematography_by",
#         "1" : "produced_by_company",
#         "2" : "composed_by",
#         "3" : "belong_to",
#         "10": "starred_by",
#         "11": "produced_in",
#         "12": "related_to",
#         "14": "edited_by",
#         "15": "produced_by_producer",
#         "16": "wrote_by",
#         "18": "directed_by",
#     },
#     LFM1M: {
#         "0": "category",
#         "1": "related_product",
#         "2": "artist",
#         "3": "engineer",
#         "4": "producer",
#         "5": "featured_artist",
#     },
#     CELL: {
#         "0": "category",
#         "1": "also_buy_related_product",
#         "2": "related_product",
#         "3": "brand",
#         "4": "also_view_related_product",
#         "5": "related_product"
#     }
# }