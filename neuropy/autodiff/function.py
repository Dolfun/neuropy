import numpy as np

BINARY_UFUNCS = {
    np.add: lambda x, y: (1, 1),
    np.subtract: lambda x, y: (1, -1),
    np.multiply: lambda x, y: (y, x),
    np.divide: lambda x, y: (1 / y, -x / np.square(x)),
    np.power: lambda x, y: (y * np.power(x, y - 1), np.log(x) * np.power(x, y)),
    # np.matmul: None,
}

UNARY_UFUNCS = {
    np.square: lambda x: 2 * x,
    np.sqrt: lambda x: 1 / (2 * np.sqrt(x)),
    np.exp: lambda x: np.exp(x),
    np.exp2: lambda x: np.exp2(x) * np.log(2),
    np.log: lambda x: 1 / x,
    np.log2: lambda x: 1 / (x * np.log(2)),
    np.log10: lambda x: 1 / (x * np.log(10)),
    np.reciprocal: lambda x: -1 / np.square(x)
}

BINARY_ARRAY_FUNCTIONS = {
    np.dot: lambda x, y: (y, x)
}

UNARY_ARRAY_FUNCTIONS = {
    np.sum: lambda x: np.ones(x.shape)
}

BINARY_OPERATIONS = BINARY_UFUNCS | BINARY_ARRAY_FUNCTIONS
UNARY_OPERATIONS = UNARY_UFUNCS | UNARY_ARRAY_FUNCTIONS
HANDLED_UFUNCS = BINARY_UFUNCS | UNARY_UFUNCS
HANDLED_ARRAY_FUNCTIONS = BINARY_ARRAY_FUNCTIONS | UNARY_ARRAY_FUNCTIONS


__all__ = ['BINARY_UFUNCS',
           'UNARY_UFUNCS',
           'BINARY_ARRAY_FUNCTIONS',
           'UNARY_ARRAY_FUNCTIONS',
           'BINARY_OPERATIONS',
           'UNARY_OPERATIONS',
           'HANDLED_UFUNCS',
           'HANDLED_ARRAY_FUNCTIONS',
           ]
