# src/compiler/compiler01.py
"""
compiler01  – a step-by-step scaffold that
• batches QFT gates via kemal_nessim.batch_circuit()
• produces *real* shuttling snapshots for every 2-qubit gate
• prints a quick sanity line for each 2-qubit column
"""

# ------------------------------------------------------------------ imports
from compiler import kemal_nessim as kn   # make sure kemal_nessim.py is on this path

# helpers we need
batch_circuit              = kn.batch_circuit
is_single_qubit_op         = kn.is_single_qubit_op
is_two_qubit_op            = kn.is_two_qubit_op
get_two_qubit_path         = kn.get_two_qubit_path
get_two_qubit_gate_schedule= kn.get_two_qubit_gate_schedule

# ------------------------------------------------------------------ constants
initial_ion_pos = [
    (0, 1), (0, 3), (0, 5),
    (1, 6), (3, 6),
    (4, 1), (4, 3), (4, 5),
]
interaction_points = [(1,1),(1,3),(3,1),(3,3),(1,5),(3,5)]

# ------------------------------------------------------------------ compiler01
def compiler01(verbose=True):
    """
    Returns:
        positions_history : List[List[tuple]]  – length == timesteps
        gates_schedule    : List[List[tuple]]  – same length
    """

    batched = batch_circuit()

    positions_history = []
    gates_schedule    = []

    current_pos = list(initial_ion_pos)

    for idx, column in enumerate(batched):
        if is_single_qubit_op(column):
            # single-qubit group — nothing moves
            positions_history.append(list(current_pos))
            gates_schedule.append(column)

        elif is_two_qubit_op(column):
            q0, q1 = column[0][2]          # the two qubit indices

            # build the full in-and-out path
            twoq_path = get_two_qubit_path(current_pos,
                                           interaction_points,
                                           q0, q1)
            twoq_sched= get_two_qubit_gate_schedule(twoq_path, column)

            # ⬇︎  debug print
            if verbose:
                print(f"[2Q #{idx}] len(path)={len(twoq_path)}  "
                      f"len(schedule)={len(twoq_sched)} "
                      f"gate={column}")

            # append snapshots & gate lists
            positions_history.extend(twoq_path)
            gates_schedule.extend(twoq_sched)

            # update where everyone ended up
            current_pos = list(twoq_path[-1])

        else:
            raise RuntimeError("Unknown column type")

    # final parity check
    assert len(positions_history) == len(gates_schedule), \
        "timeline length mismatch"

    return positions_history, gates_schedule