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
#                              CHECKS
# ---------------------------------------------------------------------

def correct_position() -> bool:

    # checks for 1 qubit gates if the position is idle or not

    # checks for 2 qubit gates if both ions are ready in the interaction zone

    return True

def no_conflict(gate, gate_schedule_t, position_history) -> bool:

    if not correct_position():
        return False
    
    # and finally, if we have conflicting wires/qubits
    
    return True

# ---------------------------------------------------------------------
#                                 MAPPING
# ---------------------------------------------------------------------

#TODO: implement moving ions for 2 qubit gates
def map_ion_movement(initial_ion_pos, trap_graph): # do we need trap_graph?
    """
    Maps the sequences of the QFT to ion coordinates and sequence of gates

    Args:
        initial_ion_pos (list of tuples): initial positions of the ion at t=0

    Returns:
        
    """
    
    positions_history   = []
    gates_schedule      = []
    gates_schedule_t    = []

    positions_history.append(initial_ion_pos) # t = 0
    current_ion_pos = initial_ion_pos

    gate_seq = extract_gate_sequence(circuit)
    
    # construction of the gate sequence and position history
    used_qubits = set()
    for gate in gate_seq:
        is_two_qubit = isinstance(gate[2], tuple)

        if not is_two_qubit: 
            if gate[2] in used_qubits:
                gates_schedule.append(gates_schedule_t)
                gates_schedule_t = []
                used_qubits = set()

            gates_schedule_t.append(gate)
            used_qubits.add(gate[2])
        else:
            # close any pending 1 qubit group if a 2 qubit gate appears
            if gates_schedule_t:
                gates_schedule.append(gates_schedule_t)
                gates_schedule_t = []
                used_qubits = set()
                
            # now, append the 2 qubit gate itself
            gates_schedule.append([gate])
            # and start the next time step
            gates_schedule_t = []

    # if any leftover 1 qubit gates are left the end, append them too or control in general
    if gates_schedule_t:
        if gates_schedule:
            last = gates_schedule[-1]
            # check that last is also a 1 qubit group
            if all(not isinstance(g[2], tuple) for g in last):
                last_wires = {g[2] for g in last}
                new_wires  = {g[2] for g in gates_schedule_t}
                # if disjoint, merge; else append separately
                if last_wires.isdisjoint(new_wires):
                    gates_schedule[-1] = last + gates_schedule_t
                    return gates_schedule
        # otherwise, or if merge failed, append as its own step
        gates_schedule.append(gates_schedule_t)
    
    print(gates_schedule)
    return None

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
    map_ion_movement(initial_ion_pos, trap_graph)

    return None

if __name__ == '__main__':
    main()
