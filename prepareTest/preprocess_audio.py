from concurrent import futures
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.store_waveforms import store_waveforms
from utils.embed_and_predict import BirdNETWrapper, PerchWrapper
import config as cfg


DIR = cfg.SOUNDSCAPES_DATA_DIR


if __name__ == "__main__":
    store_waveforms(DIR, 32000)
    #birdnet = BirdNETWrapper()
    #birdnet.embed_and_predict(DIR)

    perch = PerchWrapper()
    perch.embed_and_predict(DIR)

