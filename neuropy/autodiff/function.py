import numpy as np

BINARY_FUNCTIONS = {
    np.add: None,
    np.subtract: None,
    np.multiply: None,
    np.divide: None,
    np.dot: None,
    np.matmul: None,

    np.equal: None,
}


UNARY_FUNCTIONS = {
    np.square: None,
    np.sqrt: None,
    np.exp: None,
    np.power: None,
}


__all__ = ['BINARY_FUNCTIONS', 'UNARY_FUNCTIONS']
