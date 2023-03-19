import numpy as np

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
    def insert_node(value, operation, args):
        node = value.node
        node.operation = operation
        if operation in BINARY_OPERATIONS:
            connect_node(args[0].node, node)
            connect_node(args[1].node, node)
        elif operation in UNARY_OPERATIONS:
            connect_node(args[0].node, node)

    def compute_values(self):
        for node in self.nodes:
            if node.operation is None:
                continue
            elif node.operation in BINARY_OPERATIONS:
                node.value = node.operation(node.prev_nodes[0].value, node.prev_nodes[1].value)
            elif node.operation in UNARY_OPERATIONS:
                node.value = node.operation(node.prev_nodes[0].value)

    def ready(self):
        self.compute_values()

        nr_out = 0
        for node in self.nodes:
            node.adj_value = np.zeros(node.value.shape)
            if len(node.next_nodes) == 0:
                nr_out += 1
        if nr_out > 1:
            raise Exception('Expected only one output variable')

    def differentiate(self):
        self.ready()

        output_node = self.nodes[-1]
        output_node.adj_value = np.ones(output_node.value.shape)

        for node in reversed(self.nodes):
            if node.operation in BINARY_OPERATIONS:
                u0 = node.prev_nodes[0]
                u1 = node.prev_nodes[1]
                partial_diff = BINARY_OPERATIONS[node.operation](u0.value, u1.value)
                u0.adj_value += node.adj_value * partial_diff[0]
                u1.adj_value += node.adj_value * partial_diff[1]
            elif node.operation in UNARY_OPERATIONS:
                u = node.prev_nodes[0]
                partial_diff = UNARY_OPERATIONS[node.operation](u.value)
                u.adj_value += node.adj_value * partial_diff


__all__ = ['Graph', 'Node']
