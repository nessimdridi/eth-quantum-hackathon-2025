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
    trap = nx.Graph()
    rows, cols = 5, 7
    interaction_nodes = [(1, 1), (1, 3), (3, 1), (3, 3), (1, 5), (3, 5)]

    for r in range(rows):
        for c in range(cols):
            node_id = (r, c)
            node_type = "interaction" if node_id in interaction_nodes else "standard"
            trap.add_node(node_id, type=node_type)

    for r in range(rows):
        for c in range(cols):
            node_id = (r, c)
            if c + 1 < cols:
                trap.add_edge(node_id, (r, c + 1))
            if r + 1 < rows:
                trap.add_edge(node_id, (r + 1, c))
    return trap

# Define the layout for consistent positioning
def get_trap_positions(trap):
    return {node: (node[0], node[1]) for node in trap.nodes}

def visualize_movement_on_trap(trap, positions_history, gates_schedule):
    
    
    pos = get_trap_positions(trap)
    fig, ax = plt.subplots(figsize=(10, 10))
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
    beam_circle1 = Circle((0, 0), radius=0.2, edgecolor='blue', facecolor='blue', linewidth=2, alpha=0.4, zorder=4)
    ax.add_patch(beam_circle)
    beam_circle.set_visible(False)
    ax.add_patch(beam_circle1)
    beam_circle1.set_visible(False)

    # Circle-based legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Interaction Node',
               markerfacecolor='orange', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Standard Node',
               markerfacecolor='lightblue', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Idle Qubit',
               markerfacecolor='blue', alpha=0.3, markersize=12),
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
    
    frame_state = {
        "last_ms_qubits": set(),
        "linger_ms": False,
    }

    def update(frame):

        positions = positions_history[frame]
        gates = gates_schedule[frame] if frame < len(gates_schedule) else []

        current_ms = next((gate for gate in gates if gate[0] == "MS"), None)
        if current_ms:
            frame_state["last_ms_qubits"] = set(current_ms[2])
            frame_state["linger_ms"] = True
        elif frame_state["linger_ms"]:
            # Linger one extra frame
            frame_state["linger_ms"] = False
        else:
            frame_state["last_ms_qubits"] = set()

        interacting_qubits = frame_state["last_ms_qubits"]
       # Clear previous X markers
        for marker in getattr(update, "idle_markers", []):
            marker.remove()
        update.idle_markers = []

        resolved_positions = []
        alphas = []
        for idx, p in enumerate(positions):
            if isinstance(p, tuple) and len(p) == 3 and p[2] == "idle":
                base_pos = (p[0], p[1])
                alpha = 0.3
            else:
                base_pos = p
                alpha = 1.0

            if idx in interacting_qubits:
                offset = -0.1 if idx == list(interacting_qubits)[0] else 0.1
                base_pos = (base_pos[0], base_pos[1] + offset)

            resolved_positions.append(base_pos)
            alphas.append(alpha)

            # Draw "X" marker for idle qubits
            if isinstance(p, tuple) and len(p) == 3 and p[2] == "idle":
                x, y = base_pos
                size = 0.085
                line1 = ax.plot([x - size, x + size], [y - size, y + size], color='b', lw=2, zorder=6)[0]
                line2 = ax.plot([x - size, x + size], [y + size, y - size], color='b', lw=2, zorder=6)[0]
                update.idle_markers.extend([line1, line2])

        scat.set_offsets(resolved_positions)
        scat.set_alpha(None)  # Enable per-point alpha
        scat.set_facecolor([(0, 0, 1, a) for a in alphas])

        ax.set_title(f"Qubit Positions at Timestep {frame}", fontsize=14)

        gate_info = format_gate_info(gates)
        gate_text.set_text(f"Gates at t={frame}:\n{gate_info}")

        beam_circle.set_visible(False)
        beam_circle1.set_visible(False)
        for gate in gates:
            if isinstance(gate[2], tuple) and len(gate[2]) == 2:
                q1_idx, q2_idx = gate[2]
                try:
                    x1, y1 = resolved_positions[q1_idx]
                    x2, y2 = resolved_positions[q2_idx]

                    # Compute midpoint
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2

                    # Symmetrical offset for visual beam
                    offset = 0.1
                    beam_circle.center = (mid_x, mid_y - offset)
                    beam_circle1.center = (mid_x, mid_y + offset)

                    beam_circle.set_visible(True)
                    beam_circle1.set_visible(True)
                except IndexError:
                    print(f"Invalid qubit indices: {q1_idx}, {q2_idx}")
                break

        return scat, gate_text


    ani = animation.FuncAnimation(fig, update, frames=len(positions_history), interval=2000, blit=False)
    # plt.show()
    ani.save("qubit_movement_compiler_v01.gif", writer='ffmpeg', fps=1)
    # ani.save("qubit_movement.gif", writer='pillow', fps=1)
    
    
# Run visualization
trap = create_trap_graph()
visualize_movement_on_trap(trap, positions_history, gates_schedule)