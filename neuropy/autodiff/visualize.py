from graphviz import Digraph


def visualize(graph):
    f = Digraph()
    f.attr(rankdir='LR', size='40, 32')
    f.attr('node', shape='circle')
    for node in graph.nodes:
        label = f'x{node.index}: {node.value}, {node.adj_value}'
        if node.operation is not None:
            label += f'\n{node.operation.__name__}'
        f.node(f'{node.index}', label=label, shape='circle')
    for node in graph.nodes:
        for next_node in node.next_nodes:
            f.edge(str(node.index), str(next_node.index))

    return f


__all__ = ['visualize']
