class Node:
    count = 0

    def __init__(self, value, ufunc=None):
        self.prev_nodes = []
        self.next_nodes = []
        self.value = value
        self.ufunc = ufunc
        self.index = self.__class__.count
        self.__class__.count += 1

    def __hash__(self):
        return hash(self.__class__.count)


def connect_node(prev_node, curr_node):
    prev_node.next_nodes.append(curr_node)
    curr_node.prev_nodes.append(prev_node)


__all__ = ['Node', 'connect_node']
