from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

import config as cfg


df = pd.read_excel("data-sovon-aquila.xlsx")
groups = df.groupby('filename')

dest_dir = cfg.SOUNDSCAPES_DATA_DIR + "aquila-predictions/"

shutil.rmtree(dest_dir)
Path(dest_dir).mkdir()

species_list = pd.read_csv("sovon_species.csv", index_col=0)
latin_names = species_list['latin_name'].str.lower().values.tolist()
n_classes = len(latin_names)

def get_species_index(name: str) -> int:
    name = name.replace('_', ' ')
    name = "corvus monedula" if name == "coloeus monedula" else name
    
    if name in latin_names:
        return latin_names.index(name)

    return -1

missing = set()

for name, group in groups:
    sample_scores = np.zeros((n_classes), dtype=float)

    for column in group.columns[5:]:
        species_scores = group[column].values
        species_score = np.amax(species_scores) # max-pooling
        # species_score = ((1/len(species_scores)) * np.sum(species_scores**3)) ** (1/3) L3 pooling
        species_index = get_species_index(column)
        
        if species_index >= 0:
            sample_scores[species_index] = species_score
        else:
            missing.add(column)

    file_name = os.path.splitext(name)[0]
    output_path = "{}{}.npy".format(dest_dir, file_name)
    
    np.save(output_path, sample_scores)

print(len(missing))