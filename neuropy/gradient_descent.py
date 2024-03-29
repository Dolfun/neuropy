import numpy as np
import neuropy.autodiff as ad


def gradient_descent(x, y, w_in, *, alpha, const_alpha=False,
                     prediction_function,
                     cost_function,
                     gradient_function=None,
                     nr_iterations, nr_output=10):

    w = np.copy(w_in)

    cost_history = []
    alpha_history = []

    prev_w = prev_grad = np.zeros(w.shape)

    g = w_ = None
    if gradient_function is None:
        g = ad.Graph()
        w_ = g.create_variable(w)
        cost_function(prediction_function(x, w_), y)

    def print_iteration(iteration_no):
        print(f'Iteration {iteration_no:4d}: Cost {cost_history[-1]:8.5f}')

    for i in range(1, nr_iterations + 1):
        if gradient_function is None:
            g.compute_gradient()
            grad = w_.gradient()
        else:
            grad = gradient_function(x, y, w)

        if i != 0 and (np.allclose(grad, prev_grad) or np.allclose(w, prev_w)):
            print_iteration(i)
            print(f'Converged at iteration {i}')
            break

        w_diff = w - prev_w
        grad_diff = grad - prev_grad
        if i > 0 and not const_alpha:
            alpha = np.dot(w_diff, grad_diff) / np.dot(grad_diff, grad_diff)

        prev_w = np.copy(w)
        prev_grad = np.copy(grad)

        w -= alpha * grad

        cost_history.append(cost_function(prediction_function(x, w), y))
        alpha_history.append(alpha)

        if i % np.ceil(nr_iterations / nr_output) == 0 or i == nr_iterations or i == 1:
            print_iteration(i)

    return w, cost_history, alpha_history
