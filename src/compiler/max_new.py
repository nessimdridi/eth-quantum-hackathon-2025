import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

positions_history = [
    [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)],  # Initial positions at t=0
    [(0, 0), (0, 1), (1, 1), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],  # Qubit 2/3 moved to (1, 1)
    [(0, 0), (0, 1), (1, 1), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],  # No movement at t=2
    [(0, 0), (0, 1), (1, 1), (1, 1), (2, 0, "idle"), (2, 1), (3, 0), (3, 1)],  # No movement at t=2
]

gates_schedule = [
    [("RX", 1.57, 0), ("RY", 3.14, 1)],  # Gates applied at t=0
    [("MS", 0.78, (2, 3))],              # MS gate applied at t=1
    [],
    []# (No gates applied at t=2) There is no need to repeat MS for its duration in the gates schedule.
]



def create_trap_graph() -> nx.Graph:
    """Create a graph representing the Penning trap.

    The Penning trap is represented as a grid of nodes, where each node can be
    either an interaction node or a standard node. The interaction nodes are
    connected to their corresponding idle nodes, and the standard nodes are
    connected to their neighboring standard nodes.
    """

    trap = nx.Graph()

    rows = 7
    cols = 5

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
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
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
    beam_circle = Circle((0, 0), radius=0.2, edgecolor='blue', facecolor='blue', linewidth=2, alpha=0.4, zorder=4)
    ax.add_patch(beam_circle)
    beam_circle.set_visible(False)

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
        resolved_positions = []
        for p in positions:
            if isinstance(p, tuple) and len(p) == 3 and p[2] == "idle":
                resolved_positions.append(pos.get(p, (p[0], p[1] + 0.3)))  # default offset if not found
            else:
                resolved_positions.append(p)
        scat.set_offsets(resolved_positions)

        ax.set_title(f"Qubit Positions at Timestep {frame}", fontsize=14)

        # Gate display
        gates = gates_schedule[frame] if frame < len(gates_schedule) else []
        gate_info = format_gate_info(gates)
        gate_text.set_text(f"Gates at t={frame}:\n{gate_info}")

        # Draw beam for 2-qubit gates
        beam_circle.set_visible(False)
        for gate in gates:
            if isinstance(gate[2], tuple) and len(gate[2]) == 2:
                q1, q2 = gate[2]
                try:
                    x1, y1 = resolved_positions[q1]
                    beam_circle.center = (x1, y1)
                    beam_circle.set_visible(True)
                except IndexError:
                    print(f"Invalid qubit indices: {q1}, {q2}")
                break
        return scat, gate_text

    ani = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=2000, blit=False)
    # plt.show()
    ani.save("qubit_movement.gif", writer='ffmpeg', fps=1)
    # ani.save("qubit_movement.gif", writer='pillow', fps=1)
    
    
# Run visualization
trap = create_trap_graph()
visualize_movement_on_trap(trap, positions_history, gates_schedule)