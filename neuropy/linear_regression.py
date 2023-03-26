import numpy as np


def linear_regression_analytical_solution(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y


def linear_regression_regularized_analytical_solution(x, y, _lambda):
    n, p = x.shape
    return np.linalg.inv(x.T @ x + _lambda * n * np.eye(p)) @ x.T @ y


def generate_power_set(n, s, curr_list, *, power_set):
    if n == 1:
        curr_list.append(s)
        power_set.append(curr_list)
        return
    for i in range(s + 1):
        if i <= s:
            new_list = curr_list.copy()
            new_list.append(i)
            generate_power_set(n - 1, s - i, new_list, power_set=power_set)


def generate_polynomial_features(x_in, power_set):
    n, p = x_in.shape
    x = np.ones((n, len(power_set)))
    for i, curr_set in enumerate(power_set):
        for j in range(n):
            x[:, i] *= np.power(x_in[:, j], curr_set[j])

    return x

