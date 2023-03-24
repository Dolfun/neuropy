import numpy as np


def z_score_normalization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_normalized = (x - mu) / sigma
    return x_normalized


def min_max_normalization(x):
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    x_normalized = (x - min_val) / (max_val - min_val)
    return x_normalized


def mean_normalization(x):
    mean = np.mean(x, axis=0)
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    x_normalized = (x - mean) / (max_val - min_val)
    return x_normalized
