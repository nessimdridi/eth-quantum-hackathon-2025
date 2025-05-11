# Quantum Circuit Compilation Challenge: 8-Qubit QFT on a Penning Trapped Ion Architecture

This repository contains the work for the Quantum Circuit Compilation Challenge. The goal is to compile an 8-qubit Quantum Fourier Transform (QFT) circuit for a Penning trapped ion architecture.

## Overview

The primary task is to design and implement a compiler that translates the QFT circuit into a sequence of ion positions and gate operations. This compilation must respect the physical constraints of the trap architecture, optimize for gate scheduling, minimize reconfiguration costs, and consider qubit coherence times (see docs/CHALLENGE.md).

## Implementing the Compiler

The core of the work is implemented in a compiler that performs the following:

1.  **Decompose the QFT Circuit**:
    *   Start with the standard 8-qubit QFT circuit.
    *   Break it down into constituent single-qubit rotations and controlled-phase gates.
    *   Map these logical gates to the physically valid operations (`RX`, `RY`, `MS`) supported by the ion trap.

2.  **Plan Ion Shuttling**:
    *   Define initial positions for the 8 ions on the trap grid.
    *   Strategize ion movement (shuttling) to enable gate execution at appropriate nodes while minimizing shuttling costs.
    *   Ensure all shuttling adheres to constraints like adjacency and no ion overlap.

3.  **Schedule Gates**:
    *   Assign time steps for each gate operation.
    *   Ensure single-qubit gates (`RX`, `RY`) occur at `standard` nodes.
    *   Ensure two-qubit gates (`MS`) occur at `interaction` nodes with participating ions correctly positioned for the gate's duration.
    *   Optimize the schedule to reduce total time steps and associated temperature costs.

4.  **Optimize for Fidelity**:
    *   Minimize temperature costs arising from ion shuttling to reduce noise on `MS` gates.
    *   Aim for the highest possible fidelity between the ideal QFT circuit and the compiled, noisy circuit.

5.  **Generate Outputs**:
    *   The compiler must produce two primary outputs:
        *   `positions_history`: A list tracking the (x, y) or (x, y, "idle") location of each ion at every time step.
        *   `gates_schedule`: A list detailing the quantum gates applied at each time step.

## Project Structure

*   `docs/`: Contains challenge documentation, including `CHALLENGE.md`.
*   `src/`:
    *   `compiler/`: This is where we implement our compiler logic. You can create different versions of your compiler here (e.g., `compiler_v01.py`, `compiler_v02.py`, `compiler_v03.py`).
    *   `modules/`: Contains helper modules like `qft.py` and `visualization.py`.
    *   `fidelity.py`: Script to calculate the fidelity of the compiled circuit.
    *   `trap.py`: Defines the ion trap graph and properties.
    *   `verifier.py`: Script to verify the correctness of `positions_history` and `gates_schedule`.
*   `README.md`: This file.

## Getting Started

1.  **Understand the Challenge**: Thoroughly read the [docs/CHALLENGE.md](docs/CHALLENGE.md) file to understand the trap architecture, constraints, and evaluation criteria.
2.  **Implement Your Compiler**:
    *   Develop your compiler logic within the `src/compiler/` directory. You can start with a new file (e.g., `my_compiler.py`) or modify existing versions like [`src/compiler/compiler_v03.py`](src/compiler/compiler_v03.py).
    *   The `main()` function in your compiler script should orchestrate the compilation process and generate `positions_history` and `gates_schedule`.
3.  **Run Your Compiler**:
    Execute your compiler script from the command line. For example:
    ```bash
    python src/compiler/compiler_v03.py
    ```
    This will typically run the compilation, and the script might also invoke verification, fidelity calculation, and visualization.

## Verification and Evaluation

After your compiler generates `positions_history` and `gates_schedule`, you need to verify its correctness and evaluate its performance.

*   **Verifier**: The `verifier` function checks if your outputs adhere to all rules.
    ```python
    # Inside your compiler script, after generating outputs:
    from src import verifier
    from src import trap

    # ... your code to generate positions_history and gates_schedule ...
    trap_graph = trap.create_trap_graph()
    verifier.verifier(positions_history, gates_schedule, trap_graph)
    ```
*   **Fidelity**: The `fidelity` function assesses the performance of your compiled circuit.
    ```python
    # Inside your compiler script, after generating outputs:
    from src import fidelity
    from src import trap

    # ... your code to generate positions_history and gates_schedule ...
    trap_graph = trap.create_trap_graph()
    fidelity.fidelity(positions_history, gates_schedule, trap_graph)
    ```

Refer to the `main()` functions in [`src/compiler/compiler_v01.py`](src/compiler/compiler_v01.py), [`src/compiler/compiler_v02.py`](src/compiler/compiler_v02.py), and [`src/compiler/compiler_v03.py`](src/compiler/compiler_v03.py) for examples of how these are integrated.

## Visualization

The challenge also requires visualizing the ion shuttling and gate executions. The [`src/modules/visualization.py`](src/modules/visualization.py) module provides tools for this.
```python
# Inside your compiler script, after generating outputs:
from src.modules import visualization
from src import trap

# ... your code to generate positions_history and gates_schedule ...
trap_graph = trap.create_trap_graph()
visualization.visualize_movement_on_trap(trap_graph, positions_history, gates_schedule)
```

Have fun,

Nessim, Kemal, Arne, Maximilian