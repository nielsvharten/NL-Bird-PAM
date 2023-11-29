from concurrent import futures
import shutil
from urllib.request import urlopen
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append("C:\\Users\\niels\\Documents\\Github\\Birds-NL-PAM\\")

import config as cfg


DIR = cfg.OBS_ORG_DATA_DIR
MAX_NR_RECS = cfg.N_RECS_TO_DOWNLOAD


def get_recordings_to_download(species_name: str):
    path = DIR + '{}/recs.csv'.format(species_name)
    df_recs = pd.read_csv(path, index_col=0)
    if len(df_recs) == 0: 
        return pd.DataFrame()

    creators = df_recs.groupby('creator')
    max_rec_per_creator = max(creators.size())

    df_to_download = pd.DataFrame()
    for i in range(max_rec_per_creator):
        recs = creators.nth(i)
        df_to_download = pd.concat([df_to_download, recs])

        # return if enough samples
        if len(df_to_download) >= MAX_NR_RECS:
            return df_to_download.head(MAX_NR_RECS)
            
    return df_to_download


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
    

def download_recording(url: str, species_dir: str, file: str) -> float:
    Path(species_dir + "audio/").mkdir(exist_ok=True)

    audio_path = "{}audio/{}".format(species_dir, file)
    success = store_recording_to_path(url, audio_path)

    if success:
        duration = get_audio_duration(audio_path)
        if duration:
            return duration
        else:
            return None
    else: 
        return None


def update_df_recordings(species_name: str, results: dict[str, float], faulty_files: list[str]) -> None:
    path = DIR + '{}/recs.csv'.format(species_name)
    df_recs = pd.read_csv(path, index_col=0)

    for file in results.keys():
        df_recs.duration = np.where(df_recs.file.eq(file), results[file], df_recs.duration)
        df_recs.downloaded = np.where(df_recs.file.eq(file), True, df_recs.downloaded)

    if len(faulty_files) > 0: print("Faulty urls/files:", faulty_files)
    
    df_recs.to_csv(path)


def download_species_recordings(species_name: str) -> None:
    df_to_download = get_recordings_to_download(species_name)

    results = {}
    faulty_files = []
    for i in df_to_download.index:
        if df_to_download.downloaded[i]: continue

        taxon_dir = "{}{}/".format(DIR, species_name)
        file = df_to_download.file[i]
        duration = download_recording(df_to_download.url[i], taxon_dir, file)
        if duration:
            results[file] = duration
        else:
            faulty_files.append({file})

    update_df_recordings(species_name, results, faulty_files)
        
    
def download_audio_recordings():
    species_list = pd.read_csv(cfg.SPECIES_FILE_PATH, sep=",")
    species_names = species_list['latin_name'].unique()

    with futures.ThreadPoolExecutor(cfg.CPU_THREADS // 2) as executor:
        for _ in tqdm(executor.map(download_species_recordings, species_names), total=len(species_names)):
            pass


if __name__ == '__main__':
    download_audio_recordings()


