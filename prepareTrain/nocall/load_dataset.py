from multiprocessing import Pool
import shutil
import pandas as pd
import math
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
import sys


sys.path.append(os.getcwd())

import config as cfg

SOURCE_DIR = cfg.NOCALL_DATA_DIR #+ "birdvox/"
DEST_DIR = cfg.DATASET_DIR
EMBEDDINGS_EXT = ".txt"
TRAIN_VAL_SPLIT = cfg.TRAIN_VAL_SPLIT


def split_df(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # random shuffle
    df = df.sample(frac=1.0, random_state=37)
    
    train_len = math.ceil(len(df) * TRAIN_VAL_SPLIT)

    train_df = df.head(train_len)
    val_df = df.tail(len(df) - train_len)

    return train_df, val_df


def load_embeddings(rec_id: str, dest_dir: str) -> int:
    birdnet_embeddings_path = "{}/embeddings/{}.npy".format(SOURCE_DIR, rec_id)
    perch_embeddings_path = "{}/perch-embeddings/{}.npy".format(SOURCE_DIR, rec_id)
    
    #birdnet_embeddings = np.load(birdnet_embeddings_path)
    perch_embeddings = np.load(perch_embeddings_path)
    '''
    embeddings_to_load = []
    duration = min(len(birdnet_embeddings), len(perch_embeddings))
    for i in range(duration):
        embedding = np.append(birdnet_embeddings[i], perch_embeddings[i])
        embeddings_to_load.append(embedding)
    
    embeddings_to_load = np.array(embeddings_to_load, dtype=np.float32)
    '''
    embeddings_to_load = perch_embeddings[:-3] if len(perch_embeddings) > 3 else perch_embeddings[:1]
    dest_path = "{}embeddings/{}.npy".format(dest_dir, rec_id)
    np.save(dest_path, embeddings_to_load)

    return len(embeddings_to_load)


def load_species_data_split(df: pd.DataFrame, dest_dir: str) -> pd.DataFrame:
    labels = pd.DataFrame(columns=['id', 'primary', 'secondary', 'embeddings'])
    for row in df.itertuples():
        rec_id = row.itemid

        n_embeddings = load_embeddings(rec_id, dest_dir)
        labels.loc[len(labels)] = [rec_id, "Noise", [], n_embeddings]

    return labels


def load_labels(split_labels: pd.DataFrame, dest_dir: str) -> None:
    labels_path = "{}labels.csv".format(dest_dir)
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path, index_col=0)
    else:
        labels = pd.DataFrame()
   
    labels = pd.concat([labels, split_labels])
    labels.to_csv(labels_path)


def load_dataset() -> None:
    df = pd.read_csv("{}/recs.csv".format(SOURCE_DIR), index_col=0)
    df = df[df['processed']]
    df = df[df['hasbird'] == 0]
    train_df, val_df = split_df(df)

    train_labels = load_species_data_split(train_df, DEST_DIR + "train/")
    val_labels = load_species_data_split(val_df, DEST_DIR + "val/")

    load_labels(train_labels, DEST_DIR + "train/")
    load_labels(val_labels, DEST_DIR + "val/")
