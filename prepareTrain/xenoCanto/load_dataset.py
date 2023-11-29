import ast
from itertools import repeat
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


sys.path.append(os.getcwd())

from utils.utils import flat_sigmoid
import config as cfg


SOURCE_DIR = cfg.XENO_CANTO_DATA_DIR
DEST_DIR = cfg.DATASET_DIR


N_SAMPLES_TRAIN = math.ceil(cfg.N_SAMPLES_PER_CLASS * cfg.TRAIN_VAL_SPLIT)
N_SAMPLES_VAL = cfg.N_SAMPLES_PER_CLASS - N_SAMPLES_TRAIN


def get_secondary_labels(row, classes):
    field = getattr(row, "also", None)
    if not field: return []

    secondary_labels = ast.literal_eval(field)
    secondary_labels = list(filter(lambda label: label in classes, secondary_labels))

    if "Coloeus monedula" in secondary_labels:
        secondary_labels[secondary_labels.index("Coloeus monedula")] = "Corvus monedula"

    return secondary_labels


def get_bn_score(predictions: np.ndarray, segment: int, bn_indexes: list):
    if len(bn_indexes) > 0:
        class_index = bn_indexes[0]
        return np.max(flat_sigmoid(predictions[segment:min(segment+3, len(predictions)),class_index]))
        #flat_sigmoid(predictions[segment,class_index])
    
    return 0


def get_perch_score(predictions: np.ndarray, segment: int, perch_indexes: list):
    if len(perch_indexes) > 0:
        class_index = perch_indexes[0]
        return predictions[segment,class_index]
    
    return 0


def load_embeddings(species_name: str, rec_id: str, dest_dir: str, bn_indexes: list[int], perch_indexes) -> int:
    birdnet_embeddings_path = "{}{}/embeddings/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    perch_embeddings_path = "{}{}/perch-embeddings/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    perch_predictions_path = "{}{}/perch-predictions/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    birdnet_predictions_path = "{}{}/predictions/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    
    #birdnet_embeddings = np.load(birdnet_embeddings_path)
    perch_embeddings = np.load(perch_embeddings_path)
    
    #birdnet_predictions = np.load(birdnet_predictions_path)
    perch_predictions = np.load(perch_predictions_path)

    embeddings_to_load = []
    duration = len(perch_embeddings) - 3 #min(len(birdnet_predictions), len(perch_embeddings))
    for i in range(duration):
        #birdnet_score = get_bn_score(birdnet_predictions, i, bn_indexes)
        perch_score = get_perch_score(perch_predictions, i, perch_indexes)

        if perch_score >= 0.02:#birdnet_score >= 0.1:
            #birdnet_embedding =  np.mean([birdnet_embeddings[i], birdnet_embeddings[i+2]], axis=0) if i + 2 < len(birdnet_embeddings) else birdnet_embeddings[i]
            #birdnet_embedding = birdnet_embeddings[i+1] if i+2 < len(birdnet_embeddings) else birdnet_embeddings[i]
            #embedding = np.append(birdnet_embedding, perch_embeddings[i])
            embeddings_to_load.append(perch_embeddings[i])
    
    if len(embeddings_to_load) == 0:
        #embedding = np.append(birdnet_embeddings[0], perch_embeddings[0])
        embeddings_to_load.append(perch_embeddings[0])
    
    embeddings_to_load = np.array(embeddings_to_load, dtype=np.float32)
    dest_path = "{}embeddings/{}.npy".format(dest_dir, rec_id)
    np.save(dest_path, embeddings_to_load)

    return len(embeddings_to_load)


def load_birdnet_predictions(species_name: str, rec_id: str, dest_dir: str) -> None:
    source_path = "{}{}/predictions/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    dest_path = "{}birdnet-predictions/{}.npy".format(dest_dir, rec_id)
    shutil.copy(source_path, dest_path)


def load_perch_predictions(species_name: str, rec_id: str, dest_dir: str) -> None:
    source_path = "{}{}/perch-predictions/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    dest_path = "{}perch-predictions/{}.npy".format(dest_dir, rec_id)
    shutil.copy(source_path, dest_path)


def load_complete_embeddings(species_name: str, rec_id: str, dest_dir: str) -> int:
    birdnet_embeddings_path = "{}{}/embeddings/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    perch_embeddings_path = "{}{}/perch-embeddings/{}.npy".format(SOURCE_DIR, species_name, rec_id)
    
    #birdnet_embeddings = np.load(birdnet_embeddings_path)
    perch_embeddings = np.load(perch_embeddings_path)
    '''
    embeddings_to_load = []
    duration = min(len(birdnet_embeddings), len(perch_embeddings))
    for i in range(duration):

        birdnet_embedding = np.mean([birdnet_embeddings[i], birdnet_embeddings[i+2]], axis=0) if i + 2 < len(birdnet_embeddings) else birdnet_embeddings[i]
        embedding = np.append(birdnet_embedding, perch_embeddings[i])
        embeddings_to_load.append(embedding)
    
    embeddings_to_load = np.array(embeddings_to_load, dtype=np.float32)
    '''
    embeddings_to_load = perch_embeddings[:-3] if len(perch_embeddings) > 3 else perch_embeddings[:1]
    dest_path = "{}complete-embeddings/{}.npy".format(dest_dir, rec_id)
    np.save(dest_path, embeddings_to_load)

    return len(embeddings_to_load)


def load_species_data_split(species_name: str, ebird_code, df: pd.DataFrame, split: str, classes: list[str]) -> pd.DataFrame:
    dest_dir = "{}{}/".format(DEST_DIR, split)
    labels = pd.DataFrame(columns=['id', 'primary', 'secondary', 'embeddings'])

    bn_labels = pd.read_csv(cfg.BN_LABELS_PATH, sep='_', header=None)
    bn_indexes = bn_labels.index[bn_labels[0] == species_name].tolist()

    perch_labels = pd.read_csv(cfg.PERCH_MODEL_PATH + "assets/label.csv")
    perch_indexes = perch_labels.index[perch_labels['ebird2021'] == ebird_code].tolist()

    for row in df.itertuples():
        rec_id = row.id
        secondary_labels = get_secondary_labels(row, classes)

        if split == "val": 
            load_complete_embeddings(species_name, rec_id, dest_dir)
            #load_birdnet_predictions(species_name, rec_id, dest_dir)
            load_perch_predictions(species_name, rec_id, dest_dir)
        
        n_embeddings = load_embeddings(species_name, rec_id, dest_dir, bn_indexes, perch_indexes)

        labels.loc[len(labels)] = [rec_id, species_name, secondary_labels, n_embeddings]
   
    return labels



def split_df(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # random shuffle but keep recs from same creator together
    '''
    recs_per_creator = [df for _, df in df.groupby('rec')]
    random.shuffle(recs_per_creator)
    
    train_len = math.ceil(len(df) * cfg.TRAIN_VAL_SPLIT)

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    for recs in recs_per_creator:
        if len(train_df) + len(recs) <= train_len:
            train_df = pd.concat([train_df, recs])
        else:
            val_df = pd.concat([val_df, recs])
    '''
    df = df.sample(frac=1.0, random_state=42)
    train_len = math.ceil(len(df) * cfg.TRAIN_VAL_SPLIT)
    train_df = df.head(train_len)
    val_df = df.tail(len(df) - train_len)

    return train_df, val_df


def load_species_data(species_name: str, train_samples_loaded: int, val_samples_loaded: int, classes: list[str]) -> (pd.DataFrame, pd.DataFrame):
    recs_path = "{}{}/recs.csv".format(SOURCE_DIR, species_name)
    if not os.path.exists(recs_path): return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(recs_path, index_col=0)
    df = df[df['processed']]
    if len(df) == 0: return pd.DataFrame(), pd.DataFrame()

    train_df, val_df = split_df(df)
    train_df = train_df.head(N_SAMPLES_TRAIN - train_samples_loaded)
    val_df = val_df.head(N_SAMPLES_VAL - val_samples_loaded)

    species_list = pd.read_csv(cfg.SPECIES_FILE_PATH)
    ebird_code = species_list[species_list['latin_name'] == species_name]['ebird_code'].values[0]

    train_labels = load_species_data_split(species_name, ebird_code, train_df, "train", classes)
    val_labels = load_species_data_split(species_name, ebird_code, val_df, "val", classes)

    return train_labels, val_labels 


def load_labels(labels, dest_dir):
    if os.path.exists(dest_dir):
        old_labels = pd.read_csv(dest_dir, index_col=0)
        labels = pd.concat([old_labels, labels])
    
    labels.to_csv(dest_dir)


def load_dataset(classes: list[str], samples_per_class: np.ndarray) -> None:
    with Pool(cfg.CPU_THREADS) as pool:
        train_results, val_results = zip(*pool.starmap(load_species_data, zip(classes, samples_per_class[0], samples_per_class[1], repeat(classes))))

    for i in range(samples_per_class.shape[-1]):
        samples_per_class[0][i] += len(train_results[i])
        samples_per_class[1][i] += len(val_results[i])

    train_labels = pd.concat(train_results)
    val_labels = pd.concat(val_results)

    load_labels(train_labels, DEST_DIR + "train/labels.csv")
    load_labels(val_labels, DEST_DIR + "val/labels.csv")

    return samples_per_class
