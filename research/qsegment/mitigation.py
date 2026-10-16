import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit


def calibration_circuit(backend_qubits, physical, prepared):
    """Prepare every measured qubit in `prepared` (0 or 1) and measure it into classical bit k of register meas."""
    meas = ClassicalRegister(len(physical), "meas")
    circuit = QuantumCircuit(backend_qubits)
    circuit.add_register(meas)
    if prepared:
        for q in physical:
            circuit.x(q)
    for k, q in enumerate(physical):
        circuit.measure(q, meas[k])
    return circuit


def matrices_from_counts(counts_by_prepared, n):
    """One 2x2 confusion matrix per measured qubit, assuming readout errors are independent between qubits.
    `counts_by_prepared` maps 0 and 1 to the counts of the calibration circuit. A[m, t] = P(read m | prepared t)."""
    flips = {}
    for prepared, counts in counts_by_prepared.items():
        read_one = np.zeros(n)
        for bits, count in counts.items():
            bits = bits.replace(" ", "")
            for k in range(n):
                read_one[k] += count * (bits[n - 1 - k] == "1")
        flips[prepared] = read_one / sum(counts.values())
    return [np.array([[1 - flips[0][k], 1 - flips[1][k]], [flips[0][k], flips[1][k]]]) for k in range(n)]


def calibration_matrices(sim, backend_qubits, physical, shots, seed):
    counts = {prepared: sim.run(calibration_circuit(backend_qubits, physical, prepared), shots=shots,
                                seed_simulator=seed + prepared).result().get_counts() for prepared in (0, 1)}
    return matrices_from_counts(counts, len(physical))


def project_to_simplex(v):
    """Closest probability distribution to v in Euclidean distance."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    ind = np.arange(1, len(v) + 1)
    rho = ind[u - css / ind > 0][-1]
    return np.maximum(v - css[rho - 1] / rho, 0)


def mitigate(probs, matrices):
    """Apply the inverse confusion matrix of every qubit, then project to a valid distribution."""
    n = len(matrices)
    tensor = np.asarray(probs, dtype=float).reshape([2] * n)
    for k, A in enumerate(matrices):
        axis = n - 1 - k
        tensor = np.moveaxis(np.tensordot(np.linalg.inv(A), tensor, axes=([1], [axis])), 0, axis)
    return project_to_simplex(tensor.reshape(-1))
