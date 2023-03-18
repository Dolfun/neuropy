import numpy as np
import numpy.lib.mixins
from .function import *
from .node import *


class Variable(numpy.lib.mixins.NDArrayOperatorsMixin):
    count = 0

    def __init__(self, value, graph=None):
        if graph is None:
            raise Exception('variable must be initialized from graph')
        self.value = np.array(value)
        self.graph = graph
        self.node = Node(self.__array__())
        self.graph.nodes.append(self.node)
        self.graph.variables.append(self)

        self.index = self.__class__.count
        self.__class__.count += 1

    def __repr__(self):
        array_repr = self.value.__array__().__repr__()
        return f'{self.__class__.__name__}({array_repr[array_repr.find("["):-1]})'

    def __hash__(self):
        return hash(self.__class__.count)

    def __array__(self):
        return self.value.__array__()

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == '__call__':
            if ufunc not in UNARY_FUNCTIONS and ufunc not in BINARY_FUNCTIONS:
                return NotImplemented

            new_args = []
            for arg in args:
                if hasattr(arg, '__array__'):
                    new_args.append(arg.__array__())
                else:
                    new_args.append(arg)

            evaluated_value = self.__class__(ufunc(*new_args, **kwargs), self.graph)
            self.graph.insert_node(evaluated_value, ufunc, args, kwargs)
            return evaluated_value
        else:
            return NotImplemented


__all__ = ['Variable']
