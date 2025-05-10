import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from src import trap
from src import fidelity
from src import verifier
from functools import lru_cache
from src.modules import visualization 

trap_graph = trap.create_trap_graph()

num_ions = 8

device = qml.device("default.mixed", wires=num_ions)

# ---------------------------------------------------------------------
#                                 CIRCUIT
# ---------------------------------------------------------------------

def make_Hadamard(wire):
    # these calls go directly into whatever QNode is active
    qml.RY(np.pi/2, wires=wire)
    qml.RX(np.pi,   wires=wire)

def make_Z_rotation(theta, wire):
    qml.RY(np.pi/2, wires = wire)
    qml.RX(theta, wires = wire)
    qml.RY(-np.pi/2, wires = wire)

def controlled_not_gate(control, target):
    qml.RY(np.pi/2, wires=control)
    qml.IsingXX(np.pi/2, wires=[control, target])
    qml.RX(-np.pi/2, wires=control)
    qml.RX(-np.pi/2, wires=target)
    qml.RY(-np.pi/2, wires=control)

def controlled_phase_gate(k, control, target):
    controlled_not_gate(control, target)
    make_Z_rotation(-0.5*2*np.pi/(2**k), wire=target)
    controlled_not_gate(control, target)
    make_Z_rotation(0.5*2*np.pi/(2**k), wire=control)
    make_Z_rotation(0.5*2*np.pi/(2**k), wire=target)

@qml.qnode(device=device)
def circuit():
    # QFT sequence of single qubit rotations (RX and RY) and MS gates
    for i in range(num_ions):
        make_Hadamard(wire=i)
        t = i+1
        for k in range(2, num_ions-i+1):
            controlled_phase_gate(k, control=t, target=i)
            t += 1
    return qml.density_matrix(wires=range(num_ions))

def extract_gate_sequence(qnode):
    """
    Given a Pennylane QNode, returns a flat list of tuples
    (gate_name, parameter, wire) in the order they were applied.
    """
    qnode()
    tape = qnode.tape
    
    seq = []
    for op in tape.operations:
        if op.name in ("RX", "RY", "IsingXX"):
            name = "MS" if op.name == "IsingXX" else op.name

            # assume single-parameter gates
            angle = float(op.parameters[0])
            
            # handle wires for 1 or 2 qubit gates
            wires = list(op.wires)
            wire = wires[0] if len(wires) == 1 else tuple(wires)
            
            seq.append((name, angle, wire))
    return seq


# ---------------------------------------------------------------------
#                                 BATCHING
# ---------------------------------------------------------------------


def batch_circuit():
    """
    Maps the sequences of the QFT to batched, preselected gates

    Returns:
        batched_circuit 
    """
    
    batched_circuit      = []
    batched_circuit_t    = []

    gate_seq = extract_gate_sequence(circuit)
    
    # construction of the gate sequence and position history
    used_qubits = set()
    for gate in gate_seq:
        is_two_qubit = isinstance(gate[2], tuple)

        if not is_two_qubit: 
            if gate[2] in used_qubits:
                batched_circuit.append(batched_circuit_t)
                batched_circuit_t = []
                used_qubits = set()

            batched_circuit_t.append(gate)
            used_qubits.add(gate[2])
        else:
            # close any pending 1 qubit group if a 2 qubit gate appears
            if batched_circuit_t:
                batched_circuit.append(batched_circuit_t)
                batched_circuit_t = []
                used_qubits = set()
                
            # now, append the 2 qubit gate itself
            batched_circuit.append([gate])
            # and start the next time step
            batched_circuit_t = []

    # if any leftover 1 qubit gates are left the end, append them too or control in general
    # do we need this?
    if batched_circuit_t:
        if batched_circuit:
            last = batched_circuit[-1]
            # check that last is also a 1 qubit group
            if all(not isinstance(g[2], tuple) for g in last):
                last_wires = {g[2] for g in last}
                new_wires  = {g[2] for g in batched_circuit_t}
                # if disjoint, merge; else append separately
                if last_wires.isdisjoint(new_wires):
                    batched_circuit[-1] = last + batched_circuit_t
                    return batched_circuit
        # otherwise, or if merge failed, append as its own step
        batched_circuit.append(batched_circuit_t)
    
    return batched_circuit

# ---------------------------------------------------------------------
#                                 MAPPING
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def shortest_path(src, dst):
    """Return list of nodes along the shortest path from src to dst."""
    return nx.shortest_path(trap_graph, src, dst)

def is_single_qubit_op(circuit_column) -> bool:
    operation = circuit_column[0]
    if isinstance(operation[2], int):
        return True
    return False

def is_two_qubit_op(circuit_column) -> bool:
    operation = circuit_column[0]
    if isinstance(operation[2], tuple):
        return True
    return False

def adjusted(pos, idx):
    x, y = pos
    if 0 <= idx < 3:
        return (x + 1, y)
    elif 3 <= idx < 5:
        return (x, y - 1)
    elif 5 <= idx < 8:
        return (x - 1, y)
    raise ValueError(f"Qubit index {idx} out of supported range")

def select_int_point(current_positions, interaction_positions, qu1, qu2):
    """
    Selects the interaction point index between two qubits qu1 and qu2.

    Args:
        current_positions (list of tuple): Positions of all qubits.
        interaction_positions (list of tuple): Positions of all interaction points.
        qu1 (int): Index of the first qubit.
        qu2 (int): Index of the second qubit.

    Returns:
        int: Index of the interaction point in interaction_positions.
    """

    try:
        qu1_pos = adjusted(current_positions[qu1], qu1)
        qu2_pos = adjusted(current_positions[qu2], qu2)
        int_pos = (qu1_pos[0], qu2_pos[1])
        return interaction_positions.index(int_pos), int_pos
    except (IndexError, ValueError) as e:
        raise ValueError(f"Failed to find interaction point: {e}")


def get_qubits_paths(current_positions, interaction_positions, qu1, qu2):
    """
    Get the simultaneous movement paths from qubit 1 and qubit 2 to their interaction point.

    Args:
        current_positions (list of tuple): Positions of all qubits.
        interaction_positions (list of tuple): Positions of all interaction points.
        qu1 (int): Index of the first qubit.
        qu2 (int): Index of the second qubit.
        G (networkx.Graph): The trap graph.

    Returns:
        tuple: (path1, path2) where
            path1 is the path from qu1 to the interaction point,
            path2 is the path from qu2 to the interaction point.
    """
    int_idx, int_pos  = select_int_point(current_positions, interaction_positions, qu1, qu2)
    path1 = shortest_path(adjusted(current_positions[qu1], qu1), interaction_positions[int_idx] )
    path2 = shortest_path(adjusted(current_positions[qu2], qu2), interaction_positions[int_idx])
    return [current_positions[qu1]] + path1, [current_positions[qu2]] + path2

def get_two_qubit_path(current_ion_pos, interaction_points, qu1, qu2):
    """
    Replace current_ion_pos with step-by-step updates for qubit 1 and qubit 2
    as they move along their respective paths to the interaction point.

    Args:
        current_ion_pos (list of tuple): Initial positions of all qubits.
        path1 (list of tuple): Path for qubit 1.
        path2 (list of tuple): Path for qubit 2.
        qu1 (int): Index of the first qubit.
        qu2 (int): Index of the second qubit.

    Returns:
        list of list of tuple: A list where each item is a full current_ion_pos snapshot for that step.
    """
    path1, path2 = get_qubits_paths(current_ion_pos, interaction_points, qu1, qu2)
    max_len = max(len(path1), len(path2))
    
    # Pad paths to same length (qubit stays at interaction point once arrived)
    padded1 = path1 + [path1[-1]] * (max_len - len(path1))
    padded2 = path2 + [path2[-1]] * (max_len - len(path2))

    stepwise_positions = []
    for i in range(max_len):
        step = list(current_ion_pos)  # Copy the current state
        step[qu1] = padded1[i]
        step[qu2] = padded2[i]
        stepwise_positions.append(step)

    forward = [pos.copy() for pos in stepwise_positions]

    stepwise_positions = forward + list(reversed(forward))
    
    #stepwise_positions.extend(pos.copy() for pos in reversed(forward))

    return stepwise_positions


def get_two_qubit_gate_schedule(two_qubit_path, circuit_column):
    """
    Inserts circuit_column at the first index i where
    two_qubit_path[i] == two_qubit_path[i+1].
    All other slots are [].
    """
    n = len(two_qubit_path)
    schedule = []
    inserted = False

    for i in range(n):
        if not inserted and i < n-1 and two_qubit_path[i] == two_qubit_path[i+1]:
            schedule.append(circuit_column)
            inserted = True
        else:
            schedule.append([])

    return schedule



# ---------------------------------------------------------------------
#                                 DEBUG
# ---------------------------------------------------------------------


def debug():
    test_device = qml.device("default.mixed", wires=num_ions)

    @qml.qnode(device=test_device)
    def test_qft_circuit():
        qml.QFT(wires=range(num_ions))
        return qml.density_matrix(wires=range(num_ions))

    fid = qml.math.fidelity(circuit(), test_qft_circuit())
    print("FIDELITY:", fid)
    #qml.drawer.use_style("black_white")
    #fig, ax = qml.draw_mpl(circuit)()
    #plt.show()


# ---------------------------------------------------------------------
#                                 MAIN
# ---------------------------------------------------------------------

def main():
    # call and interface everything
    initial_ion_pos = [(0, 1), (0, 3), (0, 5), (1, 6), (3, 6), (4, 1), (4, 3), (4, 5)]  # Initial positions at t=0
    interaction_points = [(1, 1), (1, 3), (3, 1), (3, 3), (1, 5), (3, 5)]
    # debug()
    batched_circuit = batch_circuit()

    gates_schedule    = []
    positions_history = []
    current_ion_pos = initial_ion_pos

    cnt_single_op = 0 
    cnt_two_op    = 0
    for circuit_column in batched_circuit:
        if is_single_qubit_op(circuit_column):
            cnt_single_op += 1
            gates_schedule.append(circuit_column)
            current_ion_pos = current_ion_pos # placeholder
            positions_history.append(current_ion_pos)
        elif is_two_qubit_op(circuit_column):
            cnt_two_op += 1
            two_qubit_path = get_two_qubit_path(
                current_ion_pos, interaction_points, circuit_column[0][2][0], circuit_column[0][2][1])
            two_qubit_gate_schedule = get_two_qubit_gate_schedule(two_qubit_path, circuit_column)
            gates_schedule = gates_schedule + two_qubit_gate_schedule
            current_ion_pos = two_qubit_path[-1]
            positions_history = positions_history + two_qubit_path

    # interface to verifier and fidelity
    verifier.verifier(positions_history, gates_schedule, trap_graph)
    fidelity.fidelity(positions_history, gates_schedule, trap_graph)

    visualization.visualize_movement_on_trap(trap_graph, positions_history, gates_schedule)

    return None

if __name__ == '__main__':
    main()
