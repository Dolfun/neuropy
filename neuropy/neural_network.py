import numpy as np
import autodiff as ad

activation_functions = {
    'sigmoid': ad.sigmoid,
    'relu': ad.relu,
    'tanh': np.tanh,
}


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.activation_functions_names = []

        self.graph = ad.Graph()
        self._y = None

    def add_layer(self, size, activation_function=None):
        self.layers.append(size)
        self.activation_functions_names.append(activation_function)

    def ready(self):
        nr_layers = len(self.layers)
        self.layers[0] = self.graph.create_variable(np.zeros(self.layers[0]))
        self.weights.append(None)
        self.biases.append(None)

        for i in range(1, nr_layers):
            n_prev_layer = self.layers[i - 1].shape[0]
            n_curr_layer = self.layers[i]

            activation_function_name = self.activation_functions_names[i]
            if activation_function_name == 'relu':
                std = np.sqrt(2 / n_prev_layer)
            else:
                std = np.sqrt(1 / n_prev_layer)

            self.weights.append(self.graph.create_variable(np.random.normal(0.0, std, (n_curr_layer, n_prev_layer))))
            # self.weights.append(self.graph.create_variable(np.random.rand(n_curr_layer, n_prev_layer)))
            self.biases.append(self.graph.create_variable(np.zeros(n_curr_layer)))

            self.layers[i] = self.layers[i - 1] @ np.transpose(self.weights[i]) + self.biases[i]

            if activation_function_name is not None:
                self.layers[i] = activation_functions[activation_function_name](self.layers[i])

    def train(self, x, y, *, nr_batches, nr_iterations, learning_rate=0.0001, nr_output=10):
        self._y = self.graph.create_variable(np.zeros(self.layers[-1].shape))
        error = self.layers[-1] - self._y
        mse = np.sum(np.square(error))

        batch_no = 0
        y_split = np.array_split(y, nr_batches)

        parameters = self.weights[1:] + self.biases[1:]
        nr_parameters = len(parameters)

        for x_mini in np.array_split(x, nr_batches):
            y_mini = y_split[batch_no]

            batch_size = x_mini.shape[0]
            print(f'[Batch {batch_no+1}/{nr_batches}: size {batch_size}]')
            batch_no += 1

            self.layers[0].value = x_mini
            self._y.value = y_mini

            m_p = [np.zeros_like(p.value) for p in parameters]
            v_p = [np.zeros_like(p.value) for p in parameters]

            b1 = 0.9
            b2 = 0.999
            epsilon = 1e-8
            alpha = 0.002

            for i in range(1, nr_iterations + 1):
                self.graph.compute_gradient()

                for j in range(nr_parameters):
                    g = parameters[j].gradient()
                    m_p[j] = b1 * m_p[j] + (1 - b1) * g
                    m = m_p[j] / (1 - np.power(b1, i))
                    v_p[j] = b2 * v_p[j] + (1 - b2) * np.square(g)
                    v = v_p[j] / (1 - np.power(b2, i))
                    parameters[j].value -= alpha * m / (np.sqrt(v) + epsilon)

                if i % np.ceil(nr_iterations / nr_output) == 0 or i == nr_iterations - 1:
                    print(f'Iteration {i:4d}: Cost {mse.value / batch_size:8.5f}')

    def evaluate(self, x):
        self.layers[0].value = x
        self._y.value = np.zeros((x.shape[0], self.layers[-1].shape[1]))
        self.graph.forward_pass()
        return self.layers[-1].value
