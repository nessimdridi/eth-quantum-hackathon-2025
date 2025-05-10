import networkx as nx


def create_trap_graph() -> nx.Graph:
    """Create a graph representing the Penning trap.

    The Penning trap is represented as a grid of nodes, where each node can be
    either an interaction node or a standard node. The interaction nodes are
    connected to their corresponding idle nodes, and the standard nodes are
    connected to their neighboring standard nodes.
    """

    trap = nx.Graph()

    rows = 5
    cols = 7

    interaction_nodes = [(1, 1), (1, 3), (3, 1), (3, 3), (1, 5), (3, 5)]

    for r in range(rows):
        for c in range(cols):
            base_node_id = (r, c)

            if base_node_id in interaction_nodes:
                trap.add_node(base_node_id, type="interaction")
            else:
                trap.add_node(base_node_id, type="standard")
                rest_node_id = (r, c, "idle")
                trap.add_node(rest_node_id, type="idle")
                trap.add_edge(base_node_id, rest_node_id)

    for r in range(rows):
        for c in range(cols):
            node_id = (r, c)
            if c + 1 < cols:
                neighbor_id = (r, c + 1)
                trap.add_edge(node_id, neighbor_id)
            if r + 1 < rows:
                neighbor_id = (r + 1, c)
                trap.add_edge(node_id, neighbor_id)
    return trap
