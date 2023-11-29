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

SOURCE_DIR = cfg.SOUNDSCAPES_DATA_DIR
DEV_DIR = cfg.DATASET_DIR + "dev/"
TEST_DIR = cfg.DATASET_DIR + "test/"


def write_to_file(path: str, content: list[dict]) -> None:
    with open(path, "w") as out:
        out.write(json.dumps(content))


def load_labels(rec_ids: list[str], dest_dir: str) -> None:
    species_df = pd.read_csv(cfg.SPECIES_FILE_PATH)

    labels_path = SOURCE_DIR + "labels.xlsx"
    labels_df = pd.read_excel(labels_path, index_col=0)

    labels_df['Soort'] = labels_df['Soort'].str.lower()
    species_df['dutch_name'] = species_df['dutch_name'].str.lower()

    merged_df = labels_df.merge(species_df, how='inner', left_on='Soort', right_on='dutch_name')
    merged_df = merged_df.astype({'species_key': 'int'})
    
    labels = pd.DataFrame(columns=['id', 'labels'])
    for rec_id in rec_ids:
        rec_rows = merged_df[merged_df['Bestand'].str.startswith(rec_id)]
        rec_labels = list(map(str, rec_rows['latin_name'].unique()))
        labels.loc[len(labels)] = [rec_id, rec_labels]
    
    labels.to_csv("{}labels.csv".format(dest_dir))


def load_file(rec_id: str, type: str, dest_dir: str) -> None:
    source_path = "{}{}/{}.npy".format(SOURCE_DIR, type, rec_id)
    dest_path = "{}{}/{}.npy".format(dest_dir, type, rec_id)
    shutil.copy(source_path, dest_path)


def load_embeddings(rec_id: str, dest_dir: str) -> None:
    perch_embeddings_path = "{}/perch-embeddings/{}.npy".format(SOURCE_DIR, rec_id)
    perch_embeddings = np.load(perch_embeddings_path)
    embeddings_to_load = perch_embeddings

    dest_path = "{}embeddings/{}.npy".format(dest_dir, rec_id)
    np.save(dest_path, embeddings_to_load)


def load_predictions(rec_id: str, dest_dir: str) -> None:
    source_path = "{}predictions/{}.npy".format(SOURCE_DIR, rec_id)
    dest_path = "{}predictions/{}.npy".format(dest_dir, rec_id)
    shutil.copy(source_path, dest_path)


def load_data_recording(rec_id: str, dest_dir: str) -> None:
    load_file(rec_id, "birdnet-predictions", dest_dir)
    load_file(rec_id, "perch-predictions", dest_dir)
    load_file(rec_id, "aquila-predictions", dest_dir)
    load_file(rec_id, "naturalis-predictions", dest_dir)
    load_embeddings(rec_id, dest_dir)


def prepare_dest_folders() -> None:
    for dir in [DEV_DIR, TEST_DIR]:
        if os.path.exists(dir): shutil.rmtree(dir)

        dest_folders = [
            dir + "embeddings/", 
            dir + "birdnet-predictions/", 
            dir + "perch-predictions/",
            dir + "aquila-predictions/",
            dir + "naturalis-predictions/",
            dir + "nlc-predictions/"
        ]
        for dest_folder in dest_folders:
            Path(dest_folder).mkdir(parents=True, exist_ok=True)


def filter_recs_with_labels(rec_ids: list[str]) -> list[str]:
    labels_path = SOURCE_DIR + "labels.xlsx"
    labels_df = pd.read_excel(labels_path, index_col=0)
    labeled_recs = labels_df['Bestand'].unique()

    filtered_rec_ids = []
    for rec_id in rec_ids:
        if rec_id + ".WAV" in labeled_recs:
            filtered_rec_ids.append(rec_id)

    return filtered_rec_ids


def get_all_recs_to_load() -> (list[str], list[str]):
    rec_files = next(os.walk(SOURCE_DIR + "audio/"))[2]
    
    recs_to_process = []
    for rec_file in rec_files:
        rec_id = os.path.splitext(rec_file)[0]
        recs_to_process.append(rec_id)

    recs_to_process = filter_recs_with_labels(recs_to_process)

    dev_recs = []#list(filter(lambda rec_id: "mshg" in rec_id, recs_to_process)) 
    test_recs = recs_to_process

    return dev_recs, test_recs


def load_dataset() -> None:
    prepare_dest_folders()
    
    dev_recs, test_recs = get_all_recs_to_load()
    with Pool(cfg.CPU_THREADS) as pool:
        pool.starmap(load_data_recording, zip(dev_recs, repeat(DEV_DIR)))
        pool.starmap(load_data_recording, zip(test_recs, repeat(TEST_DIR)))
    
    load_labels(dev_recs, DEV_DIR)
    load_labels(test_recs, TEST_DIR)


if __name__ == '__main__':
    load_dataset()