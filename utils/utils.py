import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
import re
import sys
import os

sys.path.append(os.getcwd())

import config as cfg


def flat_sigmoid(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))


def convert(x):
    x = x.split(":")
    if len(x) == 2:
        return int(x[0]) * 60 + int(x[1])
    return int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2])


def latin_to_dutch(latin):
    species_df = pd.read_csv(cfg.SPECIES_FILE_PATH)
    target_species = species_df[species_df['latin_name'] == latin]
    
    return target_species.iloc[0]['dutch_name']


def latin_to_common(latin):
    species_df = pd.read_csv(cfg.SPECIES_FILE_PATH)
    target_species = species_df[species_df['latin_name'] == latin]
    
    return target_species.iloc[0]['english_name']


def ebird_to_latin(ebird_code):
    species_df = pd.read_excel(cfg.PERCH_MODEL_PATH + "assets/ebird_taxonomy_v2023.xlsx")
    matches = species_df[species_df['SPECIES_CODE'] == ebird_code]['SCI_NAME'].values
    if len(matches) == 0:
        return ebird_code

    latin = re.sub(r'/[a-z]+$', '', matches[0])

    return latin


# https://github.com/foxtrotmike/rocch/blob/master/rocch.py
def rocch(fpr0,tpr0):
    """
    @author: Dr. Fayyaz Minhas (http://faculty.pieas.edu.pk/fayyaz/)
    Construct the convex hull of a Receiver Operating Characteristic (ROC) curve
        Input:
            fpr0: List of false positive rates in range [0,1]
            tpr0: List of true positive rates in range [0,1]
                fpr0,tpr0 can be obtained from sklearn.metrics.roc_curve or 
                    any other packages such as pyml
        Return:
            F: list of false positive rates on the convex hull
            T: list of true positive rates on the convex hull
                plt.plot(F,T) will plot the convex hull
            auc: Area under the ROC Convex hull
    """
    fpr = np.array([0]+list(fpr0)+[1.0,1,0])
    tpr = np.array([0]+list(tpr0)+[1.0,0,0])
    hull = ConvexHull(np.vstack((fpr,tpr)).T)
    vert = hull.vertices
    vert = vert[np.argsort(fpr[vert])]  
    F = [0]
    T = [0]
    for v in vert:
        ft = (fpr[v],tpr[v])
        if ft==(0,0) or ft==(1,1) or ft==(1,0):
            continue
        F+=[fpr[v]]
        T+=[tpr[v]]
    F+=[1]
    T+=[1]
    auc = np.trapz(T,F)
    return F,T,auc


