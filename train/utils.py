import numpy as np
import torch

rng = np.random.default_rng()

def stitchup(X, y, alpha=0):
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return
        
    # make sure to have even inputs
    if n_samples % 2 == 1:
        X , y = X[:-1], y[:-1]

    mid = n_samples//2
    X1, y1 = X[:mid], y[:mid]
    X2, y2 = X[mid:], y[mid:]

    splits = rng.beta(alpha, alpha, mid)
    i1 = torch.empty((mid, n_features), dtype=torch.bool)
    for i in range(mid):
        features = range(n_features)
        n_positives = round(splits[i] * n_features)
        i1[i,:n_positives] = True
        i1[i,n_positives:] = False
        #indexes = np.array([1]*n_positives + [0]*(n_features-n_positives), dtype=np.float32)
        #np.random.shuffle(indexes)
        #i1[i] = torch.Tensor(indexes)

    i1 = torch.BoolTensor(rng.permuted(i1, axis=1))
    
    #splits = torch.Tensor(splits)
    y1 = y1 * splits.reshape(-1, 1)
    y2 = y2 * (1 - splits.reshape(-1, 1))
    
    X_stitchup = torch.where(i1, X1, X2)
    y_stitchup = y1 + y2

    return X_stitchup, y_stitchup.float()