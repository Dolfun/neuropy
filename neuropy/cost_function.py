import numpy as np


def mean_squared_error(x, y, w):
    error = np.matmul(x, w) - y
    n = error.size
    return np.dot(error, error) / n
