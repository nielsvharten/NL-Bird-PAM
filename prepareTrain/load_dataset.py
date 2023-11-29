import sys
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

import observationOrg.load_dataset
import xenoCanto.load_dataset
import nocall.load_dataset
import config as cfg


def prepare_dest_folders() -> None:
    train_dir = cfg.DATASET_DIR + "train/"
    val_dir = cfg.DATASET_DIR + "val/"
    if os.path.exists(train_dir): shutil.rmtree(train_dir)
    if os.path.exists(val_dir): shutil.rmtree(val_dir)

    dest_folders = [
        train_dir + "embeddings/", train_dir + "complete-embeddings/",
        val_dir + "embeddings/", val_dir + "complete-embeddings/",
        val_dir + "perch-predictions/", val_dir + "birdnet-predictions/",
        val_dir + "own-predictions/"
    ]

    for dest_folder in dest_folders:
        Path(dest_folder).mkdir(parents=True, exist_ok=True)


def load_dataset():
    prepare_dest_folders()
    
    species_list = pd.read_csv(cfg.SPECIES_FILE_PATH)
    classes = species_list['latin_name'].values.tolist()
    samples_per_class = np.zeros(shape=(2, len(classes)), dtype=int)

    if cfg.TRAIN_OBS_ORG:
        samples_per_class = observationOrg.load_dataset.load_dataset(classes, samples_per_class)
        print("Observation.org data loaded")

    if cfg.TRAIN_XENO_CANTO:
        samples_per_class = xenoCanto.load_dataset.load_dataset(classes, samples_per_class)
        print("Xeno Canto data loaded")

    if cfg.TRAIN_NO_CALL:
        nocall.load_dataset.load_dataset()
        print("NoCall data loaded")


if __name__ == '__main__':
    load_dataset()