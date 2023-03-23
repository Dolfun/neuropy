import numpy.lib.mixins
from .function import *


class Node(numpy.lib.mixins.NDArrayOperatorsMixin):
    index_count = 0

    def __init__(self, value, graph=None):
        if graph is None:
            raise Exception('Variable must be initialized with a Graph')
        self.value = value

        self.adj_value = 0
        self.operation = None
        self.is_constant = False

        self.next_nodes = []
        self.prev_nodes = []

        self.graph = graph
        self.graph.nodes.append(self)

        self.index = self.__class__.index_count
        self.__class__.index_count += 1

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    def gradient(self):
        return self.adj_value

    def __repr__(self):
        array_repr = self.value.__array__().__repr__()
        return f'Variable({array_repr[array_repr.find("(")+1:-1]})'

    def __array__(self):
        return self.value.__array__()

    def process_operation(self, operation, args, kwargs):
        value_args = []
        node_args = []
        for arg in args:
            if isinstance(arg, Node):
                value_args.append(arg.value)
                node_args.append(arg)
            else:
                new_node = Node(arg, graph=self.graph)
                new_node.is_constant = True
                value_args.append(new_node.value)
                node_args.append(new_node)

        evaluated_value = operation(*value_args, **kwargs)
        new_node = Node(evaluated_value, graph=self.graph)
        new_node.operation = operation
        for arg in node_args:
            arg.next_nodes.append(new_node)
            new_node.prev_nodes.append(arg)
        return new_node

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != '__call__' or ufunc not in HANDLED_UFUNCS:
            return NotImplemented

        return self.process_operation(ufunc, args, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_ARRAY_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return self.process_operation(func, args, kwargs)


__all__ = ['Node']
