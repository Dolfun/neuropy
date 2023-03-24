import numpy as np


def linear_regression_analytical_solution(x, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)


__all__ = [
    'linear_regression_analytical_solution',
]
