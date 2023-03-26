import numpy as np


def linear_regression_analytical_solution(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y


def linear_regression_regularized_analytical_solution(x, y, _lambda):
    n, p = x.shape
    return np.linalg.inv(x.T @ x + _lambda * n * np.eye(p)) @ x.T @ y

