from pathlib import Path
import shutil
from urllib.request import urlopen
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent import futures
import sys
import os

sys.path.append(os.getcwd())

import config as cfg
from utils.utils import convert


DIR = cfg.XENO_CANTO_DATA_DIR
MAX_NR_RECS = cfg.N_RECS_TO_DOWNLOAD


def get_recordings_to_download(species_name: str):
    path = DIR + '{}/recs.csv'.format(species_name)
    
    df = pd.read_csv(path, index_col=0)
    if len(df) == 0: return df

    df['length'] = df['length'].apply(convert)
    df = df[df['length'] >= 1]
    df = df[df['length'] < 300]
    if len(df) == 0: return df

    qualities = df.groupby('q')
    df_recs = pd.DataFrame()

    for _, group in qualities:
        group_short = group[group['length'] < 30]
        group_long = group[group['length'] >= 30].sort_values('length')
        df_recs = pd.concat([df_recs, group_short, group_long])
        
        if len(df_recs) >= MAX_NR_RECS: break

    return df_recs.head(MAX_NR_RECS)


def update_df_recordings(species_name: str, results: dict[str, float], faulty_files: list[str]) -> None:
    path = DIR + '{}/recs.csv'.format(species_name)
    df_recs = pd.read_csv(path, index_col=0)

    for id in results.keys():
        df_recs.duration = np.where(df_recs.id.eq(id), results[id], df_recs.duration)
        df_recs.downloaded = np.where(df_recs.id.eq(id), True, df_recs.downloaded)

    if len(faulty_files) > 0: print("Faulty urls/files:", faulty_files)
    
    df_recs.to_csv(path)


def store_recording_to_path(url: str, path: str) -> bool:
    try: 
        with urlopen(url) as response, open(path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        
        return True
    except:
        return False
    

def get_audio_duration(path: str) -> float:
    try:
        return librosa.get_duration(path=path)
    except:
        return None
    

def download_recording(id: int, species_dir: str) -> float:
    Path(species_dir + "audio/").mkdir(exist_ok=True)

    url = "https://xeno-canto.org/{}/download".format(id)
    audio_path = "{}audio/{}.mp3".format(species_dir, id)
    success = store_recording_to_path(url, audio_path)

    if success:
        duration = get_audio_duration(audio_path)
        if duration:
            return duration
        else:
            return None
    else: 
        return None


def download_species_recordings(species_name: str) -> None:
    df_to_download = get_recordings_to_download(species_name)
    
    results = {}
    faulty_files = []
    for i, row in df_to_download.iterrows():
        if row['downloaded']: continue

        taxon_dir = "{}{}/".format(DIR, species_name)
        id = row['id']
        duration = download_recording(id, taxon_dir)
        if duration:
            results[id] = duration
        else:
            faulty_files.append({id})

    update_df_recordings(species_name, results, faulty_files)
        
    
def download_all_recordings():
    species_list = pd.read_csv(cfg.SPECIES_FILE_PATH, sep=",")
    species_names = species_list['latin_name'].values

    species_names = ["Branta hutchinsii"]
    for species_name in tqdm(species_names, total=len(species_names)):
        download_species_recordings(species_name)


if __name__ == '__main__':
    download_all_recordings()
