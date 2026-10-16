import re
import time
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz, StatePreparation, XXPlusYYGate

from qsegment import formulation as fm


@lru_cache(maxsize=None)
def build_ansatz(p, measure=True):
    """QAOA circuit for the 2-zone Hamiltonian with p layers, decomposed to H, RZZ, RZ, RX so that Aer can run it.
    Do not modify the returned circuit."""
    H, _ = fm.ising_2zone()
    circuit = QAOAAnsatz(cost_operator=H, reps=p)
    if measure:
        circuit.measure_all()
    return circuit.decompose(reps=2)


@lru_cache(maxsize=None)
def _w_circuit():
    vector = np.zeros(8)
    vector[[1, 2, 4]] = 1 / np.sqrt(3)
    prep = QuantumCircuit(3)
    prep.append(StatePreparation(vector), [0, 1, 2])
    return transpile(prep, basis_gates=["cx", "u"], optimization_level=1)


def w_state(circuit, a, b, c):
    """Put qubits a, b, c in (|100> + |010> + |001>) / sqrt(3): the 3-zone analogue of |+>, equal weight on the three zones.
    Qiskit's state preparation, written out in CX and single-qubit gates, needs 4 CX for this state."""
    circuit.compose(_w_circuit(), [a, b, c], inplace=True)


def ring_mixer(circuit, beta, a, b, c):
    """XY mixer on one system: exp(-i beta (XX + YY)/2) on the pairs (a,b), (b,c), (c,a) in that order.
    It moves weight between zones and never leaves the one-hot states."""
    for x, y in ((a, b), (b, c), (c, a)):
        circuit.append(XXPlusYYGate(2 * beta), [x, y])


@lru_cache(maxsize=None)
def build_ansatz_3zone(p, measure=True):
    """3-zone QAOA circuit (21 qubits): W-state start, cost layer of RZZ and RZ, XY ring mixer.
    Parameters are named γ[k] and β[k]. Do not modify the returned circuit."""
    H, _ = fm.ising_3zone()
    n_systems = H.num_qubits // 3
    gamma, beta = ParameterVector("γ", p), ParameterVector("β", p)
    circuit = QuantumCircuit(H.num_qubits)
    for k in range(n_systems):
        w_state(circuit, 3 * k, 3 * k + 1, 3 * k + 2)
    circuit.barrier()
    for layer in range(p):
        for pauli, coeff in zip(H.paulis, H.coeffs):
            where = [q for q, ch in enumerate(reversed(pauli.to_label())) if ch == "Z"]
            angle = 2 * gamma[layer] * float(coeff.real)
            circuit.rzz(angle, *where) if len(where) == 2 else circuit.rz(angle, where[0])
        circuit.barrier()
        for k in range(n_systems):
            ring_mixer(circuit, beta[layer], 3 * k, 3 * k + 1, 3 * k + 2)
        circuit.barrier()
    if measure:
        circuit.measure_all()
    return circuit


def bind_values(circuit, gammas, betas):
    """Parameter values in the order of circuit.parameters (the names are β[k] and γ[k])."""
    values = []
    for par in circuit.parameters:
        kind, k = re.fullmatch(r"(β|γ)\[(\d+)\]", par.name).groups()
        values.append(float((betas if kind == "β" else gammas)[int(k)]))
    return values


def gate_stats(circuit):
    two_qubit = lambda inst: inst.operation.num_qubits == 2 and inst.operation.name != "barrier"
    return {"two_qubit_gates": sum(1 for inst in circuit.data if two_qubit(inst)),
            "depth": circuit.depth(), "two_qubit_depth": circuit.depth(lambda inst: inst.operation.num_qubits == 2)}


def pick_transpiled(circuit, backend, levels, seeds):
    """Transpile at every (level, seed). Keep the one with the fewest two-qubit gates, then the smallest depth."""
    options, best, best_key = [], None, None
    for level in levels:
        for seed in seeds:
            start = time.perf_counter()
            transpiled = transpile(circuit, backend=backend, optimization_level=level, seed_transpiler=seed)
            option = {"level": level, "seed": seed, "seconds": time.perf_counter() - start} | gate_stats(transpiled)
            options.append(option)
            key = (option["two_qubit_gates"], option["depth"])
            if best_key is None or key < best_key:
                best, best_key, chosen = transpiled, key, option
    return best, chosen, options
