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
            pos[node] = (node[0], node[1])
        elif len(node) == 3 and node[2] == 'idle':
            # Idle node (slightly offset below)
            pos[node] = (node[0], node[1] + 0.3)
    return pos

def visualize_movement_on_trap(trap, positions_history, gates_schedule):
    pos = get_trap_positions(trap)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(right=0.75)  # Space for legend and text

    # Draw the trap graph
    node_colors = [
        'orange' if trap.nodes[n].get("type") == "interaction" else
        'lightblue' if trap.nodes[n].get("type") == "standard" else
        'lightgray'
        for n in trap.nodes
    ]
    nx.draw(trap, pos, ax=ax, node_color=node_colors, edge_color='gray', node_size=300, with_labels=False)

    # Plot qubit positions
    scat = ax.scatter([], [], c='blue', s=250, edgecolors='black', zorder=3)
    beam_line, = ax.plot([], [], color='red', linewidth=2, alpha=0.6, zorder=4)

    # Circle-based legend
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
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

    # Add text annotation for gate schedule
    gate_text = ax.text(1.05, 1.0, "", transform=ax.transAxes, fontsize=10, verticalalignment='top', family='monospace')

    def format_gate_info(gates):
        if not gates:
            return "No gates applied"
        lines = []
        for gate in gates:
            if isinstance(gate[2], int):
                lines.append(f"{gate[0]}({gate[1]:.2f}) on q{gate[2]}")
            elif isinstance(gate[2], tuple):
                qstr = ','.join(f"q{q}" for q in gate[2])
                lines.append(f"{gate[0]}({gate[1]:.2f}) on {qstr}")
        return "\n".join(lines)

    def update(frame):
        positions = positions_history[frame]
        xy = [(p[0], p[1]) for p in positions]
        scat.set_offsets(xy)

        ax.set_title(f"Qubit Positions at Timestep {frame}", fontsize=14)

        # Gate display
        gates = gates_schedule[frame] if frame < len(gates_schedule) else []
        gate_info = format_gate_info(gates)
        gate_text.set_text(f"Gates at t={frame}:\n{gate_info}")

        # Draw beam for 2-qubit gates
        beam_drawn = False
        for gate in gates:
            if isinstance(gate[2], tuple) and len(gate[2]) == 2:
                q1, q2 = gate[2]
                x1, y1 = positions[q1][0], positions[q1][1]
                x2, y2 = positions[q2][0], positions[q2][1]
                beam_line.set_data([x1, x2], [y1, y2])
                beam_drawn = True
                break
        if not beam_drawn:
            beam_line.set_data([], [])

        return scat, gate_text, beam_line

    ani = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=2000, blit=False)
    plt.show()
    
    
# Run visualization
trap = create_trap_graph()
visualize_movement_on_trap(trap, positions_history, gates_schedule)