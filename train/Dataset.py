import ast
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, classes: list[str], DIR, split, min_samples=0, p_secondary=1.0, alpha=0):
        self.classes = classes
        self.dir = DIR
        self.split = split
        self.p_secondary = p_secondary
        self.alpha = alpha

        self.ids = []
        self.primary = []
        self.secondary = []
        self.add_samples(min_samples)


    def add_sample(self, row):
        rec_primary = row['primary']
        rec_primary = self.classes.index(rec_primary) if rec_primary in self.classes else -1

        rec_secondary = ast.literal_eval(row['secondary'])
        rec_secondary = [self.classes.index(label) for label in rec_secondary]
        
        self.ids.append(row['id'])
        self.primary.append(rec_primary)
        self.secondary.append(rec_secondary)


    def add_samples(self, min_samples: int):
        df = pd.read_csv("{}{}/labels.csv".format(self.dir, self.split), index_col=0)

        for _, row in df.iterrows():
            self.add_sample(row)

        # upsampling if not enough samples
        for name, group in df.groupby('primary'):
            n_samples = len(group)
            if n_samples < min_samples:
                weights = list(group['embeddings'].values)
                indexes = list(group.index.values)

                extra_samples = random.choices(indexes, weights=weights, k=min_samples-n_samples)
                for index in extra_samples:
                    self.add_sample(group.iloc[index])

    
    '''
    def get_sample_weights(self):
        samples_per_class = { class_name: 0 for class_name in self.classes}
        for idx in range(len(self.samples)):
            sample = self.samples.iloc[idx]
            rec_labels = ast.literal_eval(sample['labels'])
            if len(rec_labels) > 0:
                samples_per_class[rec_labels[0]] += 1

        sample_weights = []
        for idx in range(len(self.samples)):
            sample = self.samples.iloc[idx]
            rec_labels = ast.literal_eval(sample['labels'])

            if len(rec_labels) == 0:
                sample_weights.append(0.1)
            else:
               sample_weights.append(1 / samples_per_class[rec_labels[0]])

        return sample_weights
    '''
        
    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        rec_id = self.ids[idx]
        rec_primary = self.primary[idx]
        rec_secondary = self.secondary[idx]

        embeddings = np.load("{}{}/embeddings/{}.npy".format(self.dir, self.split, rec_id))
        index = np.random.choice(embeddings.shape[0]) # PICK RANDOM
        embedding = torch.from_numpy(embeddings[index])
        
        n_classes = len(self.classes)
        if self.split == 'train' and rec_primary != 'Noise':
            targets = np.full(n_classes, self.alpha / n_classes, dtype=np.float32)
        else:
            targets = np.zeros(n_classes, dtype=np.float32)
        
        if rec_primary >= 0:
            targets[rec_primary] = 1.0 - self.alpha

        for class_index in rec_secondary:
            targets[class_index] = self.p_secondary - self.alpha

        targets = torch.from_numpy(targets)
        
        return embedding, targets


class EvaluationDataset(Dataset):
    def __init__(self, classes, DIR, split):
        self.classes = classes
        self.dir = DIR
        self.split = split

        self.ids = []
        self.primary = []
        self.secondary = []
        self.add_samples()
    

    def add_sample(self, row):
        rec_primary = row['primary']
        rec_primary = self.classes.index(rec_primary) if rec_primary in self.classes else -1

        rec_secondary = ast.literal_eval(row['secondary'])
        rec_secondary = [self.classes.index(label) for label in rec_secondary]
        
        self.ids.append(row['id'])
        self.primary.append(rec_primary)
        self.secondary.append(rec_secondary)


    def add_samples(self):
        df = pd.read_csv("{}{}/labels.csv".format(self.dir, self.split), index_col=0)

        for _, row in df.iterrows():
            self.add_sample(row)
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        rec_id = self.ids[idx]
        rec_primary = self.primary[idx]
        rec_secondary = self.secondary[idx]

        n_classes = len(self.classes)
        targets = np.zeros(n_classes, dtype=int)
        if rec_primary >= 0:
            targets[rec_primary] = 1
            embeddings = np.load("{}{}/complete-embeddings/{}.npy".format(self.dir, self.split, rec_id))
        else:
            embeddings = np.load("{}{}/embeddings/{}.npy".format(self.dir, self.split, rec_id))
        
        for class_index in rec_secondary:
            targets[class_index] = 1

        embeddings = torch.from_numpy(embeddings)
        targets = torch.from_numpy(targets)
        
        return embeddings, targets