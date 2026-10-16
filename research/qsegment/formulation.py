from qiskit.quantum_info import SparsePauliOp

from qsegment import scenario as sc


def qubo_2zone():
    """x_k = 0 for zone A and 1 for zone B, with the nurse fixed in zone A. With w_ij = r_ij - t_ij,
    one pair costs r_ij - w_ij x_i - w_ij x_j + 2 w_ij x_i x_j. A pair with the nurse keeps only
    r_ij - w_ij x_j. Returns (constant, linear, quadratic), keyed by system index."""
    const = float(sc.total_risk())
    linear = {j: 0.0 for j in sc.FREE}
    quad = {}
    for i, j in sc.PAIRS:
        w = sc.net_weight(i, j)
        if w == 0:
            continue
        if i == sc.FIXED:
            linear[j] -= w
        else:
            linear[i] -= w
            linear[j] -= w
            quad[(i, j)] = 2.0 * w
    return const, linear, quad


def qubo_value(qubo, index):
    const, linear, quad = qubo
    x = {j: (index >> (j - 1)) & 1 for j in sc.FREE}
    return (const + sum(c * x[j] for j, c in linear.items())
            + sum(c * x[i] * x[j] for (i, j), c in quad.items()))


def qubit(system, zone):
    """Qubit that is 1 when free system `system` (index 1..7) is in zone `zone` (3-zone one-hot encoding)."""
    return 3 * (system - 1) + zone


def ising_3zone():
    """One-hot encoding, y = (1 - Z)/2 for the qubit of each (system, zone). One pair costs
    t_ij + w_ij * sum_k y_ik y_jk, and a pair with the nurse (zone 0) costs w_0j * y_j0.
    Returns (H, constant) with H + constant equal to the cost on every valid (one-hot) basis state."""
    terms, const = [], float(sc.total_traffic())
    for i, j in sc.PAIRS:
        w = sc.net_weight(i, j)
        if w == 0:
            continue
        if i == sc.FIXED:
            terms.append(("Z", [qubit(j, 0)], -w / 2))
            const += w / 2
        else:
            for z in range(3):
                terms.append(("ZZ", [qubit(i, z), qubit(j, z)], w / 4))
                terms.append(("Z", [qubit(i, z)], -w / 4))
                terms.append(("Z", [qubit(j, z)], -w / 4))
            const += 3 * w / 4
    return SparsePauliOp.from_sparse_list(terms, num_qubits=3 * len(sc.FREE)).simplify(), const


def ising_2zone():
    """Spin s = +1 for zone A and -1 for zone B. One pair costs (t_ij + r_ij)/2 + (w_ij/2) s_i s_j, and Z
    replaces s. Returns (H, constant) with H + constant equal to the cost. Qubit k is free system k,
    and |0> is zone A."""
    terms = []
    for i, j in sc.PAIRS:
        w = sc.net_weight(i, j)
        if w == 0:
            continue
        if i == sc.FIXED:
            terms.append(("Z", [j - 1], w / 2))
        else:
            terms.append(("ZZ", [i - 1, j - 1], w / 2))
    const = sum(sc.traffic(i, j) + sc.risk(i, j) for i, j in sc.PAIRS) / 2
    return SparsePauliOp.from_sparse_list(terms, num_qubits=len(sc.FREE)), const
