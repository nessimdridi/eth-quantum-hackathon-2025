"""
compiler01.py  – “kemal logic” wrapped in a compile() function
--------------------------------------------------------------

• Same batching / routing heuristics you prototyped
• No verifier, fidelity, or visualisation calls inside
• Returns positions_history, gates_schedule so an external harness
  (e.g. a notebook) can test / compare compilers uniformly
"""

import sys, os, networkx as nx
from functools import lru_cache

# ---------------------------------------------------------------------
#  import project packages
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from src import trap
from src.modules import qft

# ---------------------------------------------------------------------
trap_graph = trap.create_trap_graph()
INTERACTION_POINTS = [(1,1), (1,3), (3,1), (3,3), (1,5), (3,5)]

@lru_cache(maxsize=None)
def shortest_path(a, b):
    return nx.shortest_path(trap_graph, a, b)

# -------------------- batching (unchanged from kemal) ----------------
def batch_circuit(use_Z, use_cont_MS):
    batched, tmp, used = [], [], set()
    seq = qft.extract_gate_sequence(qft.circuit, use_Z, use_cont_MS)

    for g in seq:
        is_two = isinstance(g[2], tuple)
        if not is_two:
            if g[2] in used:
                batched.append(tmp); tmp=[]; used=set()
            tmp.append(g); used.add(g[2])
        else:
            if tmp: batched.append(tmp); tmp=[]; used=set()
            batched.append([g])
    if tmp: batched.append(tmp)
    return batched

# -------------------- helper funcs (unchanged from kemal) ------------
def _is_single(col): return isinstance(col[0][2], int)
def _adjust(pos, idx):
    x,y = pos
    if   0<=idx<3:   return (x+1, y)
    elif 3<=idx<5:   return (x,   y-1)
    elif 5<=idx<8:   return (x-1, y)
    raise ValueError(idx)

def _select_int(current, qu1, qu2):
    a = _adjust(current[qu1], qu1)
    b = _adjust(current[qu2], qu2)
    cand = (a[0], b[1])
    if cand in INTERACTION_POINTS:
        return cand
    # fallback: minimal sum distance
    return min(INTERACTION_POINTS,
               key=lambda n: nx.shortest_path_length(trap_graph, current[qu1], n)
                           + nx.shortest_path_length(trap_graph, current[qu2], n))

def _get_two_qubit_path(cur, qu1, qu2):
    tgt  = _select_int(cur, qu1, qu2)
    p1   = shortest_path(_adjust(cur[qu1], qu1), tgt)
    p2   = shortest_path(_adjust(cur[qu2], qu2), tgt)
    L    = max(len(p1), len(p2))

    # pad so both lists length L
    p1 += [p1[-1]]*(L-len(p1))
    p2 += [p2[-1]]*(L-len(p2))

    # build forward list of full-state snapshots
    snaps=[]
    for i in range(L):
        s = list(cur)
        s[qu1], s[qu2] = p1[i], p2[i]
        snaps.append(s)

    # mirror back to home
    snaps_back = list(reversed(snaps))
    return snaps + snaps_back

def _two_qubit_gate_schedule(path, column):
    sch=[]
    inserted=False
    for i in range(len(path)):
        if not inserted and i<len(path)-1 and path[i]==path[i+1]:
            sch.append(column); inserted=True
        else:
            sch.append([])
    return sch

# ---------------------------------------------------------------------
#  compile()  – public entry point
# ---------------------------------------------------------------------
def compile(*, use_Z=False, use_cont_MS=False,
            init_positions=None):
    """
    Returns
    -------
    positions_history, gates_schedule
    """
    if init_positions is None:
        init_positions = [(0,1), (0,3), (0,5), (1,6),
                          (3,6), (4,1), (4,3), (4,5)]

    current = init_positions[:]
    pos_hist, gate_sched = [], []

    for col in batch_circuit(use_Z, use_cont_MS):

        if _is_single(col):                         # single-qubit column
            pos_hist.append(current[:])
            gate_sched.append(col)

        else:                                       # two-qubit MS column
            q1, q2 = col[0][2]
            path   = _get_two_qubit_path(current, q1, q2)
            sched  = _two_qubit_gate_schedule(path, col)

            pos_hist.extend(path)
            gate_sched.extend(sched)
            current = path[-1]                      # end-state

    return pos_hist, gate_sched