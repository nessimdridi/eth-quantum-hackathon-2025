import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from src import trap

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

def make_Z_rotation(use_Z, theta, wire):
    if use_Z:
        qml.RZ(theta, wires=wire)
    else:
        qml.RY(np.pi/2, wires = wire)
        qml.RX(theta, wires = wire)
        qml.RY(-np.pi/2, wires = wire)

def controlled_not_gate(control, target):
    qml.RY(np.pi/2, wires=control)
    qml.IsingXX(np.pi/2, wires=[control, target])
    qml.RX(-np.pi/2, wires=control)
    qml.RX(-np.pi/2, wires=target)
    qml.RY(-np.pi/2, wires=control)

def controlled_phase_gate(k, use_Z, use_cont_MS, control, target):
    if use_cont_MS:
        qml.RY(np.pi/2, wires=control)
        qml.RY(np.pi/2, wires=target)
        qml.IsingXX(-0.5*2*np.pi/(2**k), wires=[control, target])
        qml.RY(-np.pi/2, wires=control)
        qml.RY(-np.pi/2, wires=target)
        make_Z_rotation(use_Z, 0.5*2*np.pi/(2**k), wire=control)
        make_Z_rotation(use_Z, 0.5*2*np.pi/(2**k), wire=target)
    else:
        controlled_not_gate(control, target)
        make_Z_rotation(use_Z, -0.5*2*np.pi/(2**k), wire=target)
        controlled_not_gate(control, target)
        make_Z_rotation(use_Z, 0.5*2*np.pi/(2**k), wire=control)
        make_Z_rotation(use_Z, 0.5*2*np.pi/(2**k), wire=target)

@qml.qnode(device=device)
def circuit(use_Z, use_cont_MS):
    # QFT sequence of single qubit rotations (RX and RY) and MS gates
    for i in range(num_ions):
        make_Hadamard(wire=i)
        t = i+1
        for k in range(2, num_ions-i+1):
            controlled_phase_gate(k, use_Z, use_cont_MS, control=t, target=i)
            t += 1
    return qml.density_matrix(wires=range(num_ions))

def extract_gate_sequence(qnode, use_Z, use_cont_MS):
    """
    Given a Pennylane QNode, returns a flat list of tuples
    (gate_name, parameter, wire) in the order they were applied.
    """
    qnode(use_Z, use_cont_MS)
    tape = qnode.tape
    
    seq = []
    for op in tape.operations:
        if op.name in ("RX", "RY", "RZ", "IsingXX"):
            name = "MS" if op.name == "IsingXX" else op.name

            # assume single-parameter gates
            angle = float(op.parameters[0])
            
            # handle wires for 1 or 2 qubit gates
            wires = list(op.wires)
            wire = wires[0] if len(wires) == 1 else tuple(wires)
            
            seq.append((name, angle, wire))
    return seq

