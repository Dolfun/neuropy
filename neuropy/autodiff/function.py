from numbers import Number
import numpy as np

# x, y -> Input
# r -> result of operation
# d -> adjoint of contributing node
BINARY_UFUNCS = {
    np.add: (lambda x, y, r, d: d,
             lambda x, y, r, d: d),
    np.subtract: (lambda x, y, r, d: d,
                  lambda x, y, r, d: -d),
    np.multiply: (lambda x, y, r, d: d * y,
                  lambda x, y, r, d: d * x),
    np.divide: (lambda x, y, r, d: d / y,
                lambda x, y, r, d: -d * r / y),
    np.power: (lambda x, y, r, d: d * r * y / x,
               lambda x, y, r, d: d * r * np.log(x)),
    np.matmul: (lambda x, y, r, d: np.matmul(d, y.T),
                lambda x, y, r, d: np.matmul(x.T, d)),
}

UNARY_UFUNCS = {
    np.square: lambda x, r, d: 2 * x * d,
    np.sqrt: lambda x, r, d: d / (2 * r),
    np.exp: lambda x, r, d: d * r,
    np.exp2: lambda x, r, d: d * r * np.log(2),
    np.log: lambda x, r, d: d / x,
    np.log2: lambda x, r, d: d / (x * np.log(2)),
    np.log10: lambda x, r, d: d / (x * np.log(10)),
    np.reciprocal: lambda x, r, d: -d / np.square(x),
    np.negative: lambda x, r, d: -d,

    np.tanh: lambda x, r, d: d * (1 - r * r),
}

BINARY_ARRAY_FUNCTIONS = {
    np.dot: (lambda x, y, r, d: d * y,
             lambda x, y, r, d: d * x),
}

UNARY_ARRAY_FUNCTIONS = {
    np.sum: lambda x, r, d: d * np.ones(x.shape),
    np.linalg.norm: lambda x, r, d: 2 * x * d,
}

# Custom functions


def generate_placeholder_ufunc():
    def empty_function(*args, **kwargs):
        pass
    return np.frompyfunc(empty_function, 1, 1)


def is_computable(z):
    return isinstance(z, Number) or isinstance(z, np.ndarray)


sigmoid_ufunc = generate_placeholder_ufunc()
relu_ufunc = generate_placeholder_ufunc()


def sigmoid(z):
    if is_computable(z):
        return 1 / (1 + np.exp(-z))
    return sigmoid_ufunc(z)


def relu(z):
    if is_computable(z):
        return z * (z > 0)
    return relu_ufunc


CUSTOM_UNARY_UFUNC = {
    sigmoid_ufunc: (lambda x: sigmoid(x), lambda x, r, d: d * r * (1 - r)),
    relu_ufunc: (lambda x: relu(x), lambda x, r, d: d * (x > 0))
}

BINARY_OPERATIONS = BINARY_UFUNCS | BINARY_ARRAY_FUNCTIONS
UNARY_OPERATIONS = UNARY_UFUNCS | UNARY_ARRAY_FUNCTIONS | CUSTOM_UNARY_UFUNC
HANDLED_UFUNCS = BINARY_UFUNCS | UNARY_UFUNCS | CUSTOM_UNARY_UFUNC
HANDLED_ARRAY_FUNCTIONS = BINARY_ARRAY_FUNCTIONS | UNARY_ARRAY_FUNCTIONS
CUSTOM_UFUNCS = CUSTOM_UNARY_UFUNC

__all__ = [
    'sigmoid',

    'BINARY_UFUNCS',
    'UNARY_UFUNCS',
    'BINARY_ARRAY_FUNCTIONS',
    'UNARY_ARRAY_FUNCTIONS',
    'CUSTOM_UNARY_UFUNC',
    'BINARY_OPERATIONS',
    'UNARY_OPERATIONS',
    'HANDLED_UFUNCS',
    'HANDLED_ARRAY_FUNCTIONS',
    'CUSTOM_UFUNCS',
]
