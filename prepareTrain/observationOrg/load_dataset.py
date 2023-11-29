from multiprocessing import Pool
import shutil
import math
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
import sys

from utils.utils import flat_sigmoid

sys.path.append("C:\\Users\\niels\\Documents\\Github\\Birds-NL-PAM\\")

import config as cfg
import utils


SOURCE_DIR = cfg.OBS_ORG_DATA_DIR
DEST_DIR = cfg.DATASET_DIR


MAX_N_SAMPLES_PER_REC = 3
N_SAMPLES_TRAIN = math.ceil(cfg.N_SAMPLES_PER_CLASS * cfg.TRAIN_VAL_SPLIT)
N_SAMPLES_VAL = cfg.N_SAMPLES_PER_CLASS - N_SAMPLES_TRAIN


def load_labels(df: pd.DataFrame, dest_dir: str) -> None:
    labels = pd.DataFrame(columns=['id', 'labels'])
    for row in df.itertuples():
        rec_id = os.path.splitext(row.file)[0]
        rec_labels = [str(row.speciesKey)]
        labels.loc[len(labels)] = [rec_id, rec_labels]

    labels.to_csv(dest_dir + "recs.csv")


def load_embeddings(species_name: str, rec_id: str, dest_dir: str, bn_indexes: list[int]) -> None:
    #source_path = "{}{}/embeddings/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    #dest_path = "{}embeddings/{}.npy".format(dest_dir, rec_id)
    #shutil.copy(source_path, dest_path)
    embeddings_path = "{}{}/embeddings/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    predictions_path = "{}{}/predictions/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    embeddings = np.load(embeddings_path)
    predictions = np.load(predictions_path)

    embeddings_to_load = []
    if len(bn_indexes) > 0:
        bn_index = bn_indexes[0]
        for i in range(len(embeddings)):
            if flat_sigmoid(predictions[i][bn_index]) >= 0.01:
                embeddings_to_load.append(embeddings[i])
    
    if len(embeddings_to_load) == 0:
        embeddings_to_load.append(embeddings[0])

    embeddings_to_load = np.array(embeddings_to_load, dtype=np.float32)
    dest_path = "{}embeddings/{}.npy".format(dest_dir, rec_id)
    np.save(dest_path, embeddings_to_load)


def load_predictions(species_name: str, rec_id: str, dest_dir: str) -> None:
    source_path = "{}{}/predictions/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    dest_path = "{}birdnet-predictions/{}.npy".format(dest_dir, rec_id)
    shutil.copy(source_path, dest_path)


def load_species_data_split(species_name: str, df: pd.DataFrame, split: str) -> pd.DataFrame:
    dest_dir = "{}{}/".format(DEST_DIR, split)
    labels = pd.DataFrame(columns=['id', 'labels'])

    bn_labels = pd.read_csv(cfg.BN_LABELS_PATH, sep='_', header=None)
    bn_indexes = bn_labels.index[bn_labels[0] == species_name].tolist()
    for row in df.itertuples():
        rec_id = os.path.splitext(row.file)[0]
        rec_labels = [species_name]

        load_embeddings(species_name, rec_id, dest_dir, bn_indexes)
        load_predictions(species_name, rec_id, dest_dir)

        labels.loc[len(labels)] = [rec_id, rec_labels]
   
    return labels



def split_df(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # random shuffle but keep recs from same creator together
    recs_per_creator = [df for _, df in df.groupby('creator')]
    random.shuffle(recs_per_creator)
    
    train_len = math.ceil(len(df) * cfg.TRAIN_VAL_SPLIT)

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    for recs in recs_per_creator:
        if len(train_df) + len(recs) <= train_len:
            train_df = pd.concat([train_df, recs])
        else:
            val_df = pd.concat([val_df, recs])
            
    return train_df, val_df


def load_species_data(species_name: str, train_samples_loaded, val_samples_loaded) -> (pd.DataFrame, pd.DataFrame):
    recs_path = "{}{}/recs.csv".format(SOURCE_DIR, species_name)
    if not os.path.exists(recs_path): return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(recs_path, index_col=0)
    df = df[df['processed']]
    if len(df) == 0: return pd.DataFrame(), pd.DataFrame()

    train_df, val_df = split_df(df)
    train_df = train_df.head(N_SAMPLES_TRAIN - train_samples_loaded)
    val_df = val_df.head(N_SAMPLES_VAL - val_samples_loaded)

    train_labels = load_species_data_split(species_name, train_df, "train")
    val_labels = load_species_data_split(species_name, val_df, "val")

    return train_labels, val_labels 


def load_labels(labels, dest_dir):
    if os.path.exists(dest_dir):
        old_labels = pd.read_csv(dest_dir, index_col=0)
        labels = pd.concat([old_labels, labels])
    
    labels.to_csv(dest_dir)


def load_dataset(species_names: list[str], samples_per_class: np.ndarray) -> None:
    with Pool(cfg.CPU_THREADS) as pool:
        train_results, val_results = zip(*pool.starmap(load_species_data, zip(species_names, samples_per_class[0], samples_per_class[1])))

    for i in range(samples_per_class.shape[-1]):
        samples_per_class[0][i] += len(train_results[i])
        samples_per_class[1][i] += len(val_results[i])

    train_labels = pd.concat(train_results)
    val_labels = pd.concat(val_results)

    load_labels(train_labels, DEST_DIR + "train/labels.csv")
    load_labels(val_labels, DEST_DIR + "val/labels.csv")

    return samples_per_class
