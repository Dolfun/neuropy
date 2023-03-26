import numpy as np


def mean_squared_error(x, y, w):
    error = np.matmul(x, w) - y
    n = error.size
    return np.dot(error, error) / n


def regularized_mean_squared_error(x, y, w, _lambda):
    return mean_squared_error(x, y, w) + _lambda * np.dot(w, w)
