import numpy as np

BINARY_UFUNCS = {
    np.add: (lambda x, y, res: 1,
             lambda x, y, res: 1),
    np.subtract: (lambda x, y, res: 1,
                  lambda x, y, res: -1),
    np.multiply: (lambda x, y, res: y,
                  lambda x, y, res: x),
    np.divide: (lambda x, y, res: 1 / y,
                lambda x, y, res: -res / y),
    np.power: (lambda x, y, res: res * y / x,
               lambda x, y, res: res * np.log(x)),
    np.matmul: (lambda x, y, res: np.matmul(np.ones(res.shape), y.T),
                lambda x, y, res: np.matmul(x.T, np.ones(res.shape))),
}

UNARY_UFUNCS = {
    np.square: lambda x, res: 2 * x,
    np.sqrt: lambda x, res: 1 / (2 * res),
    np.exp: lambda x, res: res,
    np.exp2: lambda x, res: res * np.log(2),
    np.log: lambda x, res: 1 / x,
    np.log2: lambda x, res: 1 / (x * np.log(2)),
    np.log10: lambda x, res: 1 / (x * np.log(10)),
    np.reciprocal: lambda x, res: -1 / np.square(x)
}

BINARY_ARRAY_FUNCTIONS = {
    np.dot: (lambda x, y, res: y,
             lambda x, y, res: x),
}

UNARY_ARRAY_FUNCTIONS = {
    np.sum: lambda x, res: np.ones(x.shape)
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
