import numpy as np
from neuropy.autodiff import sigmoid


def predict_linear(x, w):
    return x @ w


def mean_squared_error(y_p, y):
    error = y_p - y
    return np.dot(error, error) / error.size


def predict_logistic(x, w):
    return sigmoid(x @ w)


def log_loss(y_p, y):
    np.sum(-y * np.log(y_p) - (1 - y) * np.log(1 - y.p))


def linear_regression_analytical_solution(x, y):
    return np.linalg.pinv(x.T @ x) @ x.T @ y


def _generate_power_set(n, s, curr_list, power_set):
    if n == 1:
        curr_list.append(s)
        power_set.append(curr_list)
        return
    for i in range(s + 1):
        if i <= s:
            new_list = curr_list.copy()
            new_list.append(i)
            _generate_power_set(n - 1, s - i, new_list, power_set=power_set)


def generate_power_set(n, s, power_set):
    _generate_power_set(n, s, [], power_set)


def generate_polynomial_features(x_in, power_set):
    n, p = x_in.shape
    x = np.ones((n, len(power_set)))
    for i, curr_set in enumerate(power_set):
        for j in range(p):
            x[:, i] *= np.power(x_in[:, j], curr_set[j])

    return x
