import numpy as np
from numbers import Number
from .node import *
from .function import *


class Graph:
    def __init__(self):
        self.nodes = []

    def create_variable(self, value):
        if not hasattr(value, '__array__'):
            raise TypeError('Expected a numpy array-like object')
        return Node(value, self)

    def create_constant(self, value):
        if not hasattr(value, '__array__'):
            raise TypeError('Expected a numpy array-like object')
        node = Node(value, self)
        node.is_constant = True
        return node

    def forward_pass(self):
        for node in self.nodes:
            if node.operation is None:
                continue
            elif node.operation in CUSTOM_UFUNCS:
                node.value = CUSTOM_UFUNCS[node.operation][0](node.prev_nodes[0].value)
            elif node.operation in BINARY_OPERATIONS:
                node.value = node.operation(node.prev_nodes[0].value, node.prev_nodes[1].value)
            elif node.operation in UNARY_OPERATIONS:
                node.value = node.operation(node.prev_nodes[0].value)

    def ready(self):
        nr_output = 0
        for node in self.nodes:
            node.adj_value = np.zeros_like(node.value)
            if len(node.next_nodes) == 0:
                nr_output += 1
        if nr_output > 1:
            raise Exception('Expected only one output variable')

    def backward_pass(self):
        output_node = self.nodes[-1]
        output_node.adj_value = np.ones_like(output_node.value)

        def add_contribution(u_, value):
            if isinstance(value, Number) or u_.shape == value.shape:
                u_.adj_value += value
            else:
                u_.adj_value += np.sum(value, axis=0)

        for node in reversed(self.nodes):
            adj_value = node.adj_value
            if node.operation in CUSTOM_UFUNCS:
                u = node.prev_nodes[0]
                partial_diff = CUSTOM_UFUNCS[node.operation][1](u.value, node.value, adj_value)
                add_contribution(u, partial_diff)
            elif node.operation in BINARY_OPERATIONS:
                u0 = node.prev_nodes[0]
                u1 = node.prev_nodes[1]
                partial_diff = [0, 0]
                if not u0.is_constant:
                    partial_diff[0] = BINARY_OPERATIONS[node.operation][0](u0.value, u1.value, node.value, adj_value)
                if not u1.is_constant:
                    partial_diff[1] = BINARY_OPERATIONS[node.operation][1](u0.value, u1.value, node.value, adj_value)
                add_contribution(u0, partial_diff[0])
                add_contribution(u1, partial_diff[1])
            elif node.operation in UNARY_OPERATIONS:
                u = node.prev_nodes[0]
                partial_diff = UNARY_OPERATIONS[node.operation](u.value, node.value, adj_value)
                add_contribution(u, partial_diff)

    def compute_gradient(self):
        self.forward_pass()
        self.ready()
        self.backward_pass()


__all__ = ['Graph']
