import pennylane as qml
from pennylane.tape import QuantumTape
import numpy as np
from .. import trap
from .. import fidelity

trap_graph = trap.create_trap_graph()

num_ions = 8

device = qml.device("default.mixed", wires=num_ions)
test_device = qml.device("default.mixed", wires=num_ions)

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
        #for k,l in zip(range(2, num_ions-i+1), range(t, num_ions)):
        for k in range(2, num_ions-i+1):
            print(i, k, t)
            controlled_phase_gate(k, control=t, target=i)
            t += 1
    return qml.density_matrix(wires=range(num_ions))

@qml.qnode(device=test_device)
def test_qft_circuit():
    qml.QFT(wires=range(8))
    return qml.density_matrix(wires=range(8))

def extract_gate_sequence(qnode):
    """
    Given a Pennylane QNode, returns a flat list of tuples
    (gate_name, parameter, wire) in the order they were applied.
    """
    qnode()
    tape = qnode.tape
    
    seq = []
    for op in tape.operations:
        if op.name in ("RX", "RY"):
            name = op.name

            # assume single-parameter gates
            angle = float(op.parameters[0])
            
            # handle wires for 1 or 2 qubit gates
            wires = list(op.wires)
            wire = wires[0] if len(wires) == 1 else tuple(wires)
            
            seq.append((name, angle, wire))
    return seq

# ---------------------------------------------------------------------
#                                 CHECK
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
    print(gate_seq)
    
    # construction of the gate sequence and position history
    t = 0
    for gate in gate_seq:
         # only if not idle for 1 qu gates and for 2 qu gates if they are in the interaction zone
        if gates_schedule_t == [] and correct_position(): 
            gates_schedule_t.append(gate)
        if no_conflict(gate, gates_schedule_t, positions_history): 
            gates_schedule_t.append(gate)
        else: # if we need to wait for some wires or need to move, then we stop adding anything
            continue
    return None

# ---------------------------------------------------------------------
#                                 MAIN
# ---------------------------------------------------------------------

def main():
    # call and interface everything
    initial_ion_pos = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)]  # Initial positions at t=0

    #map_ion_movement(initial_ion_pos, trap_graph)
    fid = qml.math.fidelity(circuit(), test_qft_circuit())
    print("fid:", fid)
    return None

if __name__ == '__main__':
    main()
