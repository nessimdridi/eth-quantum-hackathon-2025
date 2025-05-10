import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
from matplotlib.lines import Line2D

positions_history = [
    [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)],  # Initial positions at t=0
    [(0, 0), (0, 1), (1, 1), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],  # Qubit 2/3 moved to (1, 1)
    [(0, 0), (0, 1), (1, 1), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],  # No movement at t=2
]

gates_schedule = [
    [("RX", 1.57, 0), ("RY", 3.14, 1)],  # Gates applied at t=0
    [("MS", 0.78, (2, 3))],              # MS gate applied at t=1
    [],                                  # (No gates applied at t=2) There is no need to repeat MS for its duration in the gates schedule.
]



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

# Define the layout for consistent positioning
def get_trap_positions(trap):
    """Get the positions of the nodes in the trap graph for visualization."""
    pos = {}
    for node in trap.nodes():
        if isinstance(node, tuple) and len(node) == 2:
            # Standard or interaction node
            pos[node] = (node[1], -node[0])
        elif len(node) == 3 and node[2] == 'idle':
            # Idle node (slightly offset below)
            pos[node] = (node[1], -node[0] - 0.3)
    return pos

def visualize_movement_on_trap(trap, positions_history):
    """
    Visualize the movement of qubits on the Penning trap graph.
    The function creates an animation showing the positions of qubits at each timestep.
    The trap graph is displayed with different colors for interaction, standard, and idle nodes.
    The qubit positions are represented as blue dots.
    """
    pos = get_trap_positions(trap)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Qubit Positions at Timestep 0", fontsize=14)

    # Draw static trap graph
    node_colors = [
        'orange' if trap.nodes[n].get("type") == "interaction" else
        'lightblue' if trap.nodes[n].get("type") == "standard" else
        'lightgray'
        for n in trap.nodes
    ]
    nx.draw(trap, pos, ax=ax, node_color=node_colors, edge_color='gray', node_size=300, with_labels=False)

    scat = ax.scatter([], [], c='blue', s=250, edgecolors='black', zorder=3)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Interaction Node',
               markerfacecolor='orange', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Standard Node',
               markerfacecolor='lightblue', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Idle Node',
               markerfacecolor='lightgray', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Qubit Position',
               markerfacecolor='blue', markersize=12)
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    def update(frame):
        positions = positions_history[frame]
        xy = [(p[1], -p[0]) for p in positions]
        scat.set_offsets(xy)
        ax.set_title(f"Qubit Positions at Timestep {frame}", fontsize=14)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=1000, blit=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    
    
# Run visualization
trap = create_trap_graph()
visualize_movement_on_trap(trap, positions_history)