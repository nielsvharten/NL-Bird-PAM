
import os

WORKING_DIR = os.getcwd()
OBS_ORG_DATA_DIR: str = WORKING_DIR + "\\prepareTrain\\observationOrg\\data\\"
XENO_CANTO_DATA_DIR: str = WORKING_DIR + "\\prepareTrain\\xenoCanto\\data\\"
NOCALL_DATA_DIR: str = WORKING_DIR + "\\prepareTrain\\nocall\\data\\"
SOUNDSCAPES_DATA_DIR: str = WORKING_DIR + "\\prepareTest\\data\\"

DATASET_DIR: str = WORKING_DIR + "\\dataset\\"

SPECIES_FILE_PATH: str = WORKING_DIR + "\\nl_species.csv"
BN_LABELS_PATH: str = WORKING_DIR + "\\BirdNET\\checkpoints\\V2.4\\BirdNET_GLOBAL_6K_V2.4_Labels.txt"
PERCH_MODEL_PATH: str = WORKING_DIR + "\\Perch\\"

CPU_THREADS: int = 12
N_RECS_TO_DOWNLOAD: int = 100
N_SAMPLES_PER_CLASS: int = 100
TRAIN_VAL_SPLIT: float = 0.8

TRAIN_XENO_CANTO: bool = True
TRAIN_OBS_ORG: bool = False
TRAIN_NO_CALL: bool = True
TEST_SOUNDSCAPES: bool = True

SAMPLE_RATE: int = 32000