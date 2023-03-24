import numpy as np
import neuropy.autodiff as ad


def gradient_descent(x, y, w_in, *, alpha_init=1.0,
                     compute_gradient=None, compute_cost, compute_learning_rate,
                     nr_iterations, nr_output=10):

    w = np.copy(w_in)

    cost_history = []
    alpha_history = []

    prev_w = prev_grad = np.zeros(w.shape)

    g = w_ = None
    if compute_gradient is None:
        g = ad.Graph()
        w_ = g.create_variable(w)
        compute_cost(x, y, w_)

    for i in range(nr_iterations):
        if compute_gradient is None:
            g.compute_gradient()
            grad = w_.gradient()
        else:
            grad = compute_gradient(x, y, w)

        if np.allclose(grad, prev_grad):
            print(f'Converged at iteration {i}')
            break

        alpha = compute_learning_rate(w, prev_w, grad, prev_grad) if i != 0 else alpha_init

        prev_w = np.copy(w)
        prev_grad = np.copy(grad)

        w -= alpha * grad

        cost_history.append(compute_cost(x, y, w))
        alpha_history.append(alpha)

        if i % np.ceil(nr_iterations / nr_output) == 0:
            print(f'Iteration {i:4d}: Cost {cost_history[-1]:8.5f}')

    return w, cost_history, alpha_history


def barzilai_borwein_learning_rate(w, prev_w, grad, prev_grad):
    w_diff = w - prev_w
    grad_diff = grad - prev_grad
    return np.dot(w_diff, grad_diff) / np.dot(grad_diff, grad_diff)
