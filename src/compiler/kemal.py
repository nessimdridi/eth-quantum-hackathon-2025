import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from .. import trap
from .. import fidelity

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

def get_two_qubit_path(circuit_column, current_ion_pos):
    return None

def get_two_qubit_gate_schedule(two_qubit_path, circuit_column):
    two_qubit_gate_schedule = []
    prev = None
    run_length = 0
    inserted = False

    for entry in two_qubit_path:
        if not inserted:
            if entry == prev:
                run_length += 1
            else:
                prev = entry
                run_length = 1
            
            #TODO: rethink when introducing 3D
            if run_length >= 2:
                two_qubit_gate_schedule.append(circuit_column)
                inserted = True
                continue

        two_qubit_gate_schedule.append([])

    return two_qubit_gate_schedule


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
    qml.drawer.use_style("black_white")
    fig, ax = qml.draw_mpl(circuit)()
    plt.show()


# ---------------------------------------------------------------------
#                                 MAIN
# ---------------------------------------------------------------------

def main():
    # call and interface everything
    initial_ion_pos = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)]  # Initial positions at t=0
    #debug()
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
            two_qubit_path = get_two_qubit_path(circuit_column, current_ion_pos)
            two_qubit_gate_schedule = get_two_qubit_gate_schedule(two_qubit_path, circuit_column)
            gates_schedule = gates_schedule + two_qubit_gate_schedule
            current_ion_pos = two_qubit_path[-1]
            positions_history = positions_history + two_qubit_path

    return None

if __name__ == '__main__':
    main()
