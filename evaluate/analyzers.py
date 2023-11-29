from abc import ABC, abstractmethod
import ast
from itertools import repeat
import math
import os
import json
import scipy
from sklearn import preprocessing
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys

from utils.utils import flat_sigmoid, ebird_to_latin

sys.path.append(os.getcwd())

import config as cfg


class Analyzer(ABC):

    def __init__(self, classes: list[str], species_list: pd.DataFrame):
        self.classes = classes
        self.species_list = species_list
        
        dir = cfg.DATASET_DIR
        self.TRAIN_DIR = dir + "train/"
        if cfg.TEST_SOUNDSCAPES:
            self.EVAL_DIR = dir + "test/"
        else:
            self.EVAL_DIR = dir + "val/"
            
        self.sample_scores = self.get_sample_scores()


    @abstractmethod
    def get_sample_scores(self) -> np.ndarray:
        pass
    

class CosSimAnalyzer(Analyzer):

    def get_mean_embedding(self, samples: list[str], label: str) -> np.ndarray:
        n_samples = len(samples)

        all_embeddings = np.empty(shape=(n_samples, 1024), dtype=float)
        for sample_i in range(n_samples):
            sample = samples[sample_i]

            embeddings_path = "{}embeddings/{}.npy".format(self.TRAIN_DIR, sample)
            embeddings = np.load(embeddings_path)
            best_segment, _ = self.get_best_segment(sample, label)

            all_embeddings[sample_i] = embeddings[best_segment]

        return np.apply_along_axis(np.mean, 0, all_embeddings)        


    def get_samples_per_label(self, samples: dict[list[str]]) -> dict[list[str]]:
        samples_per_label = {}
        for rec, labels in samples.items():
            for label in labels:
                samples_per_label[label] = [rec] if label not in samples_per_label.keys() else samples_per_label[label] + [rec]

        return samples_per_label


    def get_mean_embeddings(self) -> np.ndarray:
        with open(self.TRAIN_DIR + "labels.json") as f:
                samples = json.load(f)

        samples_per_label = self.get_samples_per_label(samples)
        labels = list(samples_per_label.keys())
        samples = list(samples_per_label.values())

        with Pool(cfg.CPU_THREADS) as pool:
            results = pool.starmap(self.get_mean_embedding, zip(samples, labels))

        n_classes = len(self.classes)
        
        mean_embeddings = np.empty(shape=(n_classes, 1024), dtype=float)
        for i in range(n_classes):
            class_index = self.classes.index(labels[i])
            mean_embeddings[class_index] = results[i]

        return mean_embeddings


    def get_cos_similarity(self, a, b):
        return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


    def get_sample_score(self, embeddings: np.ndarray, species_embedding: np.ndarray) -> float:
        sample_score = 0

        for i in range(0, len(embeddings), 2):
            similarity = self.get_cos_similarity(embeddings[i], species_embedding)
            if similarity > sample_score:
                sample_score = similarity

        return sample_score


    def get_eval_scores_sample(self, sample: str, class_embeddings):
        n_classes = len(class_embeddings)

        sample_scores = np.empty(shape=(n_classes), dtype=float)
        embeddings_path = "{}embeddings/{}.npy".format(self.EVAL_DIR, sample)
        embeddings = np.load(embeddings_path)

        for class_i in range(n_classes):
            sample_score = self.get_sample_score(embeddings, class_embeddings[class_i])
            sample_scores[class_i] = sample_score

        return sample_scores
    

    def get_sample_scores(self) -> np.ndarray:
        class_embeddings = self.get_mean_embeddings()

        with open(self.EVAL_DIR + "labels.json") as f:
            samples = list(json.load(f).keys())

        with Pool(cfg.CPU_THREADS) as pool:
            results = pool.starmap(self.get_eval_scores_sample, zip(samples, repeat(class_embeddings)))

        sample_scores = np.array(results)

        return sample_scores


class LDAAnalyzer(Analyzer):

    def get_train_instance(self, rec_id, rec_labels) -> np.ndarray:
        embeddings_path = "{}embeddings/{}.npy".format(self.TRAIN_DIR, rec_id)
        train_embeddings = np.load(embeddings_path)
        
        class_index = self.classes.index(rec_labels[0])
        segment_index = len(train_embeddings) // 2
        
        n_segments = len(train_embeddings)
        #return train_embeddings[segment_index], class_index

        # return every third embedding
        train_embeddings = train_embeddings[::3]

        return train_embeddings, np.repeat(class_index, len(train_embeddings))


    def get_train_instances(self) -> np.ndarray:
        samples = pd.read_csv(self.TRAIN_DIR + "labels.csv", index_col=0)
        rec_ids = samples['id'].values
        rec_labels = samples['labels'].apply(ast.literal_eval).values

        with Pool(cfg.CPU_THREADS) as pool:
            results = pool.starmap(self.get_train_instance, zip(rec_ids, rec_labels))
            
        train_embeddings, train_targets = zip(*results)
        train_embeddings = [x for l in train_embeddings for x in l]
        train_targets = [x for l in train_targets for x in l]

        train_embeddings = np.array(train_embeddings)
        train_targets = np.array(train_targets)

        return train_embeddings, train_targets    
    

    def get_eval_scores_sample(self, rec_id, clf: LinearDiscriminantAnalysis):
        embeddings_path = "{}embeddings/{}.npy".format(self.EVAL_DIR, rec_id)
        embeddings = np.load(embeddings_path)#np.array(recordings[sample_i]['embeddings'])

        probs = clf.predict_log_proba(embeddings) #clf.predict_proba(embeddings)#
        sample_scores = np.amax(probs, axis=0)      

        return sample_scores


    def train_lda_classifier(self):
        clf = LinearDiscriminantAnalysis()#solver='lsqr', shrinkage='auto')

        train_embeddings, train_targets = self.get_train_instances()
        clf = clf.fit(train_embeddings, train_targets)

        return clf
    

    def rescale_sample_scores(self, sample_scores: np.ndarray):
        resampled = np.log(sample_scores)
        min_score: float = np.amin(resampled)
        
        resampled = resampled / min_score
        resampled = np.where(resampled > 0.0, np.log(resampled), -50)
        
        resampled = resampled / -50
        
        return resampled


    def get_sample_scores(self) -> np.ndarray:
        clf = self.train_lda_classifier()

        samples = pd.read_csv(self.EVAL_DIR + "labels.csv", index_col=0)
        rec_ids = samples['id'].values
        
        with Pool(cfg.CPU_THREADS) as pool:
            results = pool.starmap(self.get_eval_scores_sample, zip(rec_ids, repeat(clf)))
            sample_scores = np.array(results)

        min_score = np.amin(sample_scores)
        rescaled_sample_scores = 1 - sample_scores/min_score #self.rescale_sample_scores(sample_scores)#

        return rescaled_sample_scores 


class BirdNETAnalyzer(Analyzer):

    def get_eval_scores_sample(self, rec_id, bn_class_index: dict):
        predictions_path = "{}birdnet-predictions/{}.npy".format(self.EVAL_DIR, rec_id)
        predictions = np.load(predictions_path)
        n_classes = len(self.classes)

        sample_scores = np.empty(shape=n_classes, dtype=np.float32)
        for i in range(n_classes):
            bn_index = bn_class_index[i]
            if bn_index >= 0:
                scores = flat_sigmoid(predictions[:,bn_index])
                # sample_scores[i] = ((1/len(scores)) * np.sum(scores**9)) ** (1/9) #L9 pooling
                sample_scores[i] = np.amax(scores)
            else:
                sample_scores[i] = 0.0

        return sample_scores

    
    def get_bn_class_index(self):
        bn_labels = pd.read_csv(cfg.BN_LABELS_PATH, sep='_', header=None)

        bn_class_index = np.full(len(self.classes), fill_value=-1, dtype=int)
        for i, class_name in enumerate(self.classes):
            bn_indexes = bn_labels.index[bn_labels[0] == class_name].tolist()
            
            if len(bn_indexes) == 1:
                bn_class_index[i] = bn_indexes[0]

        return bn_class_index


    def get_sample_scores(self) -> np.ndarray:
        samples = pd.read_csv(self.EVAL_DIR + "labels.csv")['id'].values
        bn_class_index = self.get_bn_class_index()

        with Pool(cfg.CPU_THREADS) as pool:
            results = pool.starmap(self.get_eval_scores_sample, zip(samples, repeat(bn_class_index)))

        return np.array(results)
    

class PerchAnalyzer(Analyzer):

    def get_eval_scores_sample(self, rec_id: str, perch_dict: dict):
        predictions_path = "{}perch-predictions/{}.npy".format(self.EVAL_DIR, rec_id)
        predictions = np.load(predictions_path)
        
        sample_scores = np.zeros(shape=len(self.species_list), dtype=np.float32)
        for i, class_name in enumerate(self.classes):
            if class_name in perch_dict:
                index = perch_dict[class_name]
                scores = predictions[:,index]
                #sample_scores[i] = ((1/len(scores)) * np.sum(scores**9)) ** (1/9) #L9 pooling
                sample_scores[i] = np.amax(scores[:-3]) if len(scores) > 3 else scores[0] 

        return sample_scores


    def get_perch_dict(self):
        labels_path = cfg.PERCH_MODEL_PATH + "/assets/label.csv"
        perch_labels = list(pd.read_csv(labels_path).values)

        perch_dict = {}
        for _, row in self.species_list.iterrows():
            species_name = row['latin_name']
            ebird_code = row['ebird_code']

            if ebird_code in perch_labels:
                perch_dict[species_name] = perch_labels.index(ebird_code)

        return perch_dict


    def get_sample_scores(self) -> np.ndarray:
        samples = pd.read_csv(self.EVAL_DIR + "labels.csv")['id'].values
        perch_dict = self.get_perch_dict()
        
        with Pool(cfg.CPU_THREADS) as pool:
            results = pool.starmap(self.get_eval_scores_sample, zip(samples, repeat(perch_dict)))

        return np.array(results)
    

class AquilaAnalyzer(Analyzer):

    def get_sample_scores(self) -> np.ndarray:
        samples = pd.read_csv(self.EVAL_DIR + "labels.csv")['id'].values

        n_samples = len(samples)
        n_classes = len(self.classes)

        sample_scores = np.empty((n_samples, n_classes), dtype=float) 
        for i, sample in enumerate(samples):
            sample_scores[i] = np.load("{}aquila-predictions/{}.npy".format(self.EVAL_DIR, sample))

        return sample_scores
    

class NaturalisAnalyzer(Analyzer):

    def get_sample_scores(self) -> np.ndarray:
        samples = pd.read_csv(self.EVAL_DIR + "labels.csv")['id'].values

        n_samples = len(samples)
        n_classes = len(self.classes)

        sample_scores = np.empty((n_samples, n_classes), dtype=float) 
        for i, sample in enumerate(samples):
            sample_scores[i] = np.load("{}naturalis-predictions/{}.npy".format(self.EVAL_DIR, sample))

        return sample_scores
    

class OwnAnalyzer(Analyzer):
    def __init__(self, classes: list[str], species_list: pd.DataFrame, folder="own-predictions"):
        self.folder = folder
        super().__init__(classes, species_list)

    def get_sample_scores(self) -> np.ndarray:
        samples = pd.read_csv(self.EVAL_DIR + "labels.csv")['id'].values

        n_samples = len(samples)
        n_classes = len(self.classes)

        sample_scores = np.empty((n_samples, n_classes), dtype=float) 
        for i, sample in enumerate(samples):
            sample_scores[i] = np.load("{}{}/{}.npy".format(self.EVAL_DIR, self.folder, sample))

        return sample_scores