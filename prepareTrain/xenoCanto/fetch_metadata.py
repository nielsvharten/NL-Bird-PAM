import json
from urllib.request import urlopen
import pandas as pd
from concurrent import futures
from tqdm import tqdm
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())

import config as cfg


DIR = cfg.XENO_CANTO_DATA_DIR


def get_species_recordings(species_name: str) -> list[dict]:
    page = 1; endOfRecords = False
    nr_retries = 0

    xeno_canto_name = species_name.replace(' ', '+')
    xeno_canto_name = "Coloeus+monedula" if xeno_canto_name == "Corvus+monedula" else xeno_canto_name
    xeno_canto_name = "Gulosus+aristotelis" if xeno_canto_name == "Phalacrocorax+aristotelis" else xeno_canto_name

    recordings = []
    while not endOfRecords:
        url = "https://xeno-canto.org/api/2/recordings?query={}&page={}".format(xeno_canto_name, page)
        
        try:
            # try retrieving page as json
            with urlopen(url) as response:
                result = json.loads(response.read().decode())
        except Exception as e:
            # on error: first retry species recs for up to three times
            # if failed three times for species, return recordings
            if nr_retries < 3:
                page = 1; endOfRecords = False
                nr_retries += 1
                recordings = []
                continue
            else:
                print(url, type(e).__name__)
                return recordings

        recordings.extend(result['recordings'])

        endOfRecords = page >= result['numPages']
        page += 1

    return recordings


def concat_to_present_recs(recs: pd.DataFrame, species_dir: str) -> pd.DataFrame:
    present_recs = pd.read_csv(species_dir + "recs.csv", index_col=0)

    # check whether present recs has a file column
    if 'file' not in present_recs.columns: 
        return recs

    # only keep recs not in present in present_recs
    recs = recs[~recs['file'].isin(present_recs['file'].values)]

    # concat recs only to present_recs if not empty
    if len(recs.index) > 0:
        recs = pd.concat([present_recs, recs])
    else:
        recs = present_recs

    return recs


def store_species_recordings(recordings: list[dict], taxon_key: int) -> None:
    recs = pd.DataFrame.from_dict(recordings)
    recs['speciesKey'] = taxon_key
    recs['downloaded'] = False
    recs['processed'] = False
    recs['duration'] = None

    # require directory for taxon_key
    taxon_dir = "{}{}/".format(DIR, taxon_key)
    Path(taxon_dir).mkdir(parents=True, exist_ok=True)
    
    if Path(taxon_dir + "recs.csv").exists():
        recs = concat_to_present_recs(recs, taxon_dir)
    
    recs.to_csv(taxon_dir + "recs.csv")


def fetch_species_data(species_name):
    recs = get_species_recordings(species_name)
    store_species_recordings(recs, species_name)


def fetch_metadata() -> None:
    species_list = pd.read_csv(cfg.SPECIES_FILE_PATH, sep=",")
    species_names = species_list['latin_name'].values

    species_names = ["Branta hutchinsii"]
    for species_name in tqdm(species_names, total=len(species_names)):
        fetch_species_data(species_name)


if __name__ == '__main__':
    fetch_metadata()