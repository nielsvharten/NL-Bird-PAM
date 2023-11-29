from concurrent import futures
import os
import sys
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm


sys.path.append("C:\\Users\\niels\\Documents\\Github\\Birds-NL-PAM\\")

import config as cfg

DIR = cfg.OBS_ORG_DATA_DIR

REC_COLUMNS = [
    "datasetKey", "basisOfRecord", "occurrenceStatus", "lifeStage", 
    "taxonKey", "scientificName", "species", "decimalLatitude", "decimalLongitude", 
    "locality", "countryCode", "continent", "year", "month", "day","vernacularName", 
    "recordedBy", "eventTime", "occurrenceID"
]

def get_recording(sound: dict, observation: dict, nr: int) -> dict:
    recording = { k: v for k, v in observation.items() if k in REC_COLUMNS }

    recording['url'] = sound['identifier']
    recording['nr'] = nr

    ext = os.path.splitext(urlparse(sound['identifier']).path)[1]
    recording['file'] = str(observation['key']) + "-" + str(nr) + ext
    recording['observationKey'] = observation['key']
    recording['creator'] = sound['rightsHolder']

    return recording


def get_observation_recordings(observation: dict) -> list[dict]:
    recordings = []
    for item in observation['media']:
        if item['type'] == "Sound" and "identifier" in item.keys():
            recording = get_recording(item, observation, len(recordings))
            recordings.append(recording)
        
    return recordings


def get_species_recordings(taxon_key: int) -> list[dict]:
    offset = 0; endOfRecords = False
    nr_retries = 0
    
    recordings = []
    while not endOfRecords:
        url = "https://api.gbif.org/v1/occurrence/search?" + \
            "dataset_key=8a863029-f435-446a-821e-275f4f641165" + \
            "&media_type=Sound" + \
            "&country=NL" + \
            "&occurrence_status=present" + \
            "&offset={}&taxon_key={}".format(offset, taxon_key)
        
        try:
            # try retrieving page as json
            with urlopen(url) as response:
                page = json.loads(response.read().decode())
        except Exception as e:
            # on error: first retry species recs for up to three times
            # if failed three times for species, return recordings
            if nr_retries < 3:
                offset = 0; endOfRecords = False
                nr_retries += 1
                recordings = []
                continue
            else:
                print(url, type(e).__name__)
                return recordings

        for observation in page['results']:
            obs_recordings = get_observation_recordings(observation)
            recordings.extend(obs_recordings)

        # update offset and endOfRecords before next iteration
        offset += page['limit']
        endOfRecords = page['endOfRecords']

    return recordings


def concat_to_present_recs(recs: pd.DataFrame, taxon_dir: str) -> pd.DataFrame:
    present_recs = pd.read_csv(taxon_dir + "recs.csv", index_col=0)

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


def fetch_species_data(species_data) -> None:
    recs = get_species_recordings(species_data[1]['species_key'])
    store_species_recordings(recs, species_data[1]['latin_name'])


def fetch_metadata() -> None:
    species_list = pd.read_csv(cfg.SPECIES_FILE_PATH, sep=",")
    species_rows = species_list.iterrows()

    with futures.ThreadPoolExecutor(cfg.CPU_THREADS) as executor:
        for _ in tqdm(executor.map(fetch_species_data, species_rows), total=len(species_list)):
            pass


if __name__ == '__main__':
    fetch_metadata()
