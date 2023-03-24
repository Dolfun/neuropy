import numpy as np


def gradient_descent(x, y, w_in, *, alpha_init=1.0,
                     calculate_gradient, calculate_cost, calculate_learning_rate,
                     nr_iterations, nr_output=10):

    w = np.copy(w_in)

    cost_history = []
    alpha_history = []

    prev_w = prev_grad = np.zeros(w.shape)

    for i in range(nr_iterations):
        grad = calculate_gradient(x, y, w)
        alpha = calculate_learning_rate(w, prev_w, grad, prev_grad) if i != 0 else alpha_init

        prev_w = np.copy(w)
        prev_grad = np.copy(grad)

        w -= alpha * grad

        cost_history.append(calculate_cost(x, y, w))
        alpha_history.append(alpha)

        if i % np.ceil(nr_iterations / nr_output) == 0:
            print(f'Iteration {i:4d}: Cost {cost_history[-1]:8.5f}')

        if np.allclose(grad, prev_grad):
            break

    return w, cost_history, alpha_history


def barzilai_borwein_learning_rate(w, prev_w, grad, prev_grad):
    w_diff = w - prev_w
    grad_diff = grad - prev_grad
    return np.dot(w_diff, grad_diff) / np.dot(grad_diff, grad_diff)
