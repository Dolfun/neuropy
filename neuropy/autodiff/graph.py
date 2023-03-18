from .variable import *
from .function import *
from .node import *


class Graph:
    def __init__(self):
        self.nodes = []
        self.variables = []

    def create_variable(self, value):
        return Variable(value, self)

    @staticmethod
    def insert_node(value, ufunc, args, kwargs):
        node = value.node
        node.ufunc = ufunc
        if ufunc in BINARY_FUNCTIONS:
            connect_node(args[0].node, node)
            connect_node(args[1].node, node)
        elif ufunc in UNARY_FUNCTIONS:
            connect_node(args[0].node, node)


__all__ = ['Graph', 'Node']
