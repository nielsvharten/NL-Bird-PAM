from concurrent import futures
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.store_waveforms import store_waveforms
from utils.embed_and_predict import BirdNETWrapper, PerchWrapper
import config as cfg


DIR = cfg.NOCALL_DATA_DIR


if __name__ == "__main__":
    store_waveforms(DIR, sample_rate=32000)
    #model = BirdNETWrapper()
    #model.embed_and_predict(DIR)
    model = PerchWrapper()
    model.embed_and_predict(DIR)
