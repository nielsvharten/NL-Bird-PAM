from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

import config as cfg

input_dir = os.getcwd() + "\\naturalis-outputs\\"
dest_dir = cfg.SOUNDSCAPES_DATA_DIR + "naturalis-predictions/"

files = next(os.walk(input_dir))[2]

sovon_species_list = pd.read_csv("sovon_species.csv", index_col=0)
sovon_names = sovon_species_list['latin_name'].values.tolist()
n_classes = len(sovon_names)

naturalis_species_list = pd.read_csv("naturalis_species.csv", header=None, sep="_")
naturalis_names = naturalis_species_list[0].values.tolist()

for file in files:
    df = pd.read_csv(input_dir + file)
    sample_scores = np.zeros((n_classes), dtype=float)

    for i, column in enumerate(df.columns[3:]):
        species_scores = df[column].values
        latin_name = naturalis_names[i]
        latin_name = "Corvus monedula" if latin_name == "Coloeus monedula" else latin_name
        latin_name = "Phalacrocorax aristotelis" if latin_name == "Gulosus aristotelis" else latin_name
        score = max(species_scores)

        if latin_name in sovon_names:
            sample_scores[sovon_names.index(latin_name)] = score
        else:
            print(latin_name)

    file_name = os.path.splitext(file[12:])[0]
    np.save("{}{}.npy".format(dest_dir, file_name), sample_scores)
