import numpy as np


def mean_squared_error(y_predict, y_real):
    error = y_predict - y_real
    return np.dot(error, error) / error.size
