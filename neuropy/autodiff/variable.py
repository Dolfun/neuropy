import numpy as np
import numpy.lib.mixins
from .function import *
from .node import *


class Variable(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value, graph=None):
        if graph is None:
            raise Exception('Variable must be initialized with a Graph')
        self.graph = graph
        self.node = None
        self.value = np.array(value)
        self.graph.nodes.append(self.node)
        self.graph.variables.append(self)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not hasattr(value, '__array__'):
            raise TypeError('Expected a numpy array-like object')
        self._value = value
        if self.node is None:
            self.node = Node(value)
        else:
            self.node.value = value

    def get_adj(self):
        return self.node.adj_value

    def __repr__(self):
        array_repr = self.value.__array__().__repr__()
        return f'{self.__class__.__name__}({array_repr[array_repr.find("(")+1:-1]})'

    def __array__(self):
        return self.value.__array__()

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == '__call__':
            if ufunc not in HANDLED_UFUNCS:
                return NotImplemented

            print(f'{ufunc.__name__} called with args: {args}')

            new_args = []
            for arg in args:
                if hasattr(arg, '__array__'):
                    new_args.append(arg)
                else:
                    shape = None
                    if hasattr(args[0], '__array__'):
                        shape = args[0].__array__().shape
                    elif hasattr(args[1], '__array__'):
                        shape = args[1].__array__().shape
                    new_args.append(Variable(arg * np.ones(shape), self.graph))

            val_args = []
            for arg in args:
                if hasattr(arg, '__array__'):
                    val_args.append(arg.__array__())
                else:
                    val_args.append(arg)

            evaluated_value = self.__class__(ufunc(*val_args, **kwargs), self.graph)
            self.graph.insert_node(evaluated_value, ufunc, new_args)
            return evaluated_value
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_ARRAY_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        new_args = []
        for arg in args:
            if hasattr(arg, '__array__'):
                new_args.append(arg.__array__())
            else:
                new_args.append(arg)

        evaluated_value = self.__class__(func(*new_args, **kwargs), self.graph)
        self.graph.insert_node(evaluated_value, func, args)
        return evaluated_value


__all__ = ['Variable']
