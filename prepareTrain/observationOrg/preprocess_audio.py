from concurrent import futures
import sys
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.append("C:\\Users\\niels\\Documents\\Github\\Birds-NL-PAM\\")

from utils.store_waveforms import store_waveforms
from utils.embed_and_predict import BirdNETWrapper
import config as cfg


DIR = cfg.OBS_ORG_DATA_DIR


def update_dfs_processed(species_names):
    for species_name in species_names:
        species_dir = "{}{}/".format(DIR, species_name)
        recs_path = species_dir + "recs.csv"
        
        if not os.path.exists(recs_path): continue

        recs = pd.read_csv(recs_path, index_col=0)
        for i, row in recs.iterrows():
            file_name = "{}-{}".format(row['observationKey'], row['nr'])
            embeddings_path = "{}embeddings/{}.npy".format(species_dir, file_name)
            predictions_path = "{}predictions/{}.npy".format(species_dir, file_name)
            processed = os.path.exists(embeddings_path) and os.path.exists(predictions_path)
            recs.at[i, 'processed'] = processed

        recs.to_csv(recs_path)


def store_species_waveforms(species_name: str) -> None:
        species_dir = "{}{}/".format(DIR, species_name)
        store_waveforms(species_dir)


def store_all_waveforms(species_names: list[str]) -> None:
    with futures.ThreadPoolExecutor(cfg.CPU_THREADS) as executor:
        for _ in tqdm(executor.map(store_species_waveforms, species_names), total=len(species_names)):
            pass


def embed_and_predict_all(species_names: list[str]):
    model = BirdNETWrapper()
    for species_name in tqdm(species_names, total=len(species_names)):
        species_dir = "{}{}/".format(DIR, species_name)
        model.embed_and_predict(species_dir)


if __name__ == "__main__":
    species_names = next(os.walk(cfg.OBS_ORG_DATA_DIR))[1]

    update_dfs_processed(species_names)
    store_all_waveforms(species_names)
    embed_and_predict_all(species_names)
    update_dfs_processed(species_names)



