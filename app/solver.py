import time

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

MAX_STATES = 20000
SHOTS = 1024


class NetworkError(ValueError):
    pass


def parse_network(payload):
    """Validate the page's network JSON. Returns (names, links) with links as (i, j, traffic, risk), i < j."""
    nodes, links = payload.get("nodes") or [], payload.get("links") or []
    names = [str(n.get("name", f"system {k}"))[:40] for k, n in enumerate(nodes)]
    if len(names) < 2:
        raise NetworkError("Add at least two systems.")
    out = []
    for link in links:
        if isinstance(link, (list, tuple)):
            link = {"a": link[0], "b": link[1], "t": link[2], "r": link[3]}
        a, b = int(link["a"]), int(link["b"])
        if not (0 <= a < len(names) and 0 <= b < len(names)) or a == b:
            raise NetworkError("A link points to a system that does not exist.")
        t, r = float(link.get("t", 0)), float(link.get("r", 0))
        if not (0 <= t <= 10 and 0 <= r <= 10):
            raise NetworkError("Traffic and risk must be between 0 and 10.")
        out.append((min(a, b), max(a, b), t, r))
    return names, out


def variable(i, k, K):
    """QUBO variable (and qubit) for system i in zone k. Two zones use one bit per system, three or more one bit per zone."""
    return i - 1 if K == 2 else (i - 1) * K + k


def build_qubo(n, links, K):
    """Cost of a split = traffic on links that cross zones + risk on links kept inside a zone.
    With w = risk - traffic, each link adds t + w * [same zone]. System 0 is fixed in zone 0."""
    nv = n - 1 if K == 2 else (n - 1) * K
    constant, lin, quad = 0.0, np.zeros(nv), {}
    def add(a, b, c):
        key = (min(a, b), max(a, b)); quad[key] = quad.get(key, 0.0) + c
    for i, j, t, r in links:
        w = r - t
        constant += t
        if w == 0:
            continue
        if K == 2:
            constant += w
            if i == 0:
                lin[variable(j, 0, K)] -= w
            else:
                lin[variable(i, 0, K)] -= w; lin[variable(j, 0, K)] -= w; add(variable(i, 0, K), variable(j, 0, K), 2 * w)
        elif i == 0:
            lin[variable(j, 0, K)] += w
        else:
            for k in range(K):
                add(variable(i, k, K), variable(j, k, K), w)
    return constant, lin, quad


def qubo_to_ising(constant, lin, quad):
    """Swap each bit for a qubit, x = (1 - Z) / 2. Returns the Ising Hamiltonian as a Qiskit SparsePauliOp and its constant."""
    terms = []
    for a, c in enumerate(lin):
        if c:
            terms.append(("Z", [a], -c / 2)); constant += c / 2
    for (a, b), c in quad.items():
        if c:
            terms.append(("ZZ", [a, b], c / 4)); terms.append(("Z", [a], -c / 4)); terms.append(("Z", [b], -c / 4)); constant += c / 4
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=len(lin)).simplify() if terms else SparsePauliOp.from_sparse_list([("", [], 0.0)], num_qubits=len(lin))
    return op, constant


def splits(n, K):
    """Every valid split as a zone list, system 0 fixed in zone 0; index digit i-1 is system i's zone."""
    total = K ** (n - 1)
    idx = np.arange(total)
    zones = np.zeros((total, n), dtype=int)
    for i in range(1, n):
        zones[:, i] = (idx // K ** (i - 1)) % K
    return zones


def ising_energies(op, constant, zones, K):
    """Energy of every valid split, read straight from the Qiskit operator's Z and ZZ terms."""
    n = zones.shape[1]
    Z = np.ones((zones.shape[0], op.num_qubits))
    for i in range(1, n):
        for k in range(1 if K == 2 else K):
            bit = (zones[:, i] == 1) if K == 2 else (zones[:, i] == k)
            Z[bit, variable(i, k, K)] = -1
    E = np.full(zones.shape[0], float(constant))
    for pauli, qubits, coeff in op.to_sparse_list():
        if pauli:
            E += float(np.real(coeff)) * np.prod(Z[:, qubits], axis=1)
    return E


def split_costs(zones, links):
    c = np.zeros(zones.shape[0])
    for i, j, t, r in links:
        same = zones[:, i] == zones[:, j]
        c += np.where(same, r, t)
    return c


def qaoa_circuit(n, K, op):
    """The QAOA circuit we run on hardware, built in Qiskit for this network: equal mix of zones per system,
    cost layer from the Ising operator (RZ and RZZ), then the mixer (RX for 2 zones, XY ring for 3+ zones)."""
    from qiskit.circuit import Parameter
    g, b = Parameter("γ"), Parameter("β")
    qc = QuantumCircuit(op.num_qubits)
    if K == 2:
        qc.h(range(op.num_qubits))
    else:
        vec = np.zeros(2 ** K); vec[[1 << k for k in range(K)]] = 1 / np.sqrt(K)
        for i in range(1, n):
            qc.append(StatePreparation(vec), [variable(i, k, K) for k in range(K)])
    for pauli, qubits, coeff in op.to_sparse_list():
        if pauli == "Z":
            qc.rz(2 * float(np.real(coeff)) * g, qubits[0])
        elif pauli == "ZZ":
            qc.rzz(2 * float(np.real(coeff)) * g, qubits[0], qubits[1])
    if K == 2:
        qc.rx(2 * b, range(op.num_qubits))
    else:
        for i in range(1, n):
            q = [variable(i, k, K) for k in range(K)]
            for a in range(K):
                qc.append(XXPlusYYGate(2 * b), [q[a], q[(a + 1) % K]])
    qc.measure_all()
    return qc


def mixer(K, beta):
    c, s = np.cos(beta), np.sin(beta)
    if K == 2:
        return np.array([[c, -1j * s], [-1j * s, c]])
    U = np.eye(K, dtype=complex)
    for a in range(K):
        b = (a + 1) % K
        R = np.eye(K, dtype=complex); R[a, a] = R[b, b] = c; R[a, b] = R[b, a] = -1j * s
        U = R @ U
    return U


def evolve(E, n, K, gamma, beta):
    """Exact state of the one-layer circuit inside the valid splits (the mixer never leaves them)."""
    psi = np.exp(-1j * gamma * E) / np.sqrt(len(E))
    U, N = mixer(K, beta), n - 1
    tensor = psi.reshape([K] * N)
    for axis in range(N):
        tensor = np.moveaxis(np.tensordot(U, tensor, axes=([1], [axis])), 0, axis)
    return np.abs(tensor.reshape(-1)) ** 2


def local_search(costs, idx, n, K):
    cur, val = idx, costs[idx]
    while True:
        best, best_val, stride = -1, val, 1
        for _ in range(1, n):
            digit = (cur // stride) % K
            for d in range(K):
                if d != digit:
                    cand = cur + (d - digit) * stride
                    if costs[cand] < best_val - 1e-12:
                        best, best_val = cand, costs[cand]
            stride *= K
        if best < 0:
            return cur
        cur, val = best, best_val


def canonical(z):
    seen = {}
    return [seen.setdefault(v, len(seen)) for v in z]


def prepare(payload):
    """Everything both backends share: QUBO, Qiskit Ising operator, the QAOA circuit and tuned angles."""
    t0 = time.perf_counter()
    K = int(payload.get("zones", 3))
    if K not in (2, 3):
        raise NetworkError("This demo plans 2 or 3 zones.")
    names, links = parse_network(payload)
    n = len(names)
    if K ** (n - 1) > MAX_STATES:
        raise NetworkError("Too many systems for this demo.")
    constant, lin, quad = build_qubo(n, links, K)
    op, const = qubo_to_ising(constant, lin, quad)
    zones = splits(n, K)
    E, direct = ising_energies(op, const, zones, K), split_costs(zones, links)
    gap = float(np.max(np.abs(E - direct)))
    if gap > 1e-6:
        raise RuntimeError(f"Ising energy does not match the cost (gap {gap}).")
    circuit = qaoa_circuit(n, K, op)
    sd = float(np.std(E)) or 1.0
    norm = (E - E.mean()) / sd
    energy = lambda x: float(evolve(norm, n, K, x[0], x[1]) @ norm)
    grid = [(g, b) for g in np.linspace(-2, 2, 20) for b in np.linspace(-1.6, 1.6, 20)]
    res = minimize(energy, min(grid, key=energy), method="Nelder-Mead", options={"maxiter": 120, "xatol": 1e-4, "fatol": 1e-6})
    gamma, beta = float(res.x[0]), float(res.x[1])
    bound = circuit.assign_parameters({p: (gamma / sd if p.name == "γ" else beta) for p in circuit.parameters})
    return {"t0": t0, "K": K, "n": n, "links": links, "op": op, "const": const, "zones": zones, "direct": direct,
            "norm": norm, "gamma": gamma, "beta": beta, "circuit": circuit, "bound": bound}


def summary(prep, best_idx, extra):
    op, direct = prep["op"], prep["direct"]
    return {
        "qiskit": qiskit.__version__, "zones_requested": prep["K"], "zones": prep["zones"][best_idx].tolist(),
        "cost": float(direct[best_idx]), "best_possible": float(direct.min()), "qubits": op.num_qubits,
        "ising_z_terms": sum(1 for p, _, _ in op.to_sparse_list() if p == "Z"),
        "ising_zz_terms": sum(1 for p, _, _ in op.to_sparse_list() if p == "ZZ"), "ising_constant": float(prep["const"]),
        "compute_ms": round((time.perf_counter() - prep["t0"]) * 1000, 1), **extra,
    }


def best_after_cleanup(prep, indices):
    finals = [local_search(prep["direct"], int(i), prep["n"], prep["K"]) for i in indices]
    return min(finals, key=lambda i: prep["direct"][i])


def counts_to_indices(counts, prep):
    """Turn chip readouts into valid splits. Returns ({split index: count}, share of valid readouts)."""
    n, K, nv = prep["n"], prep["K"], prep["op"].num_qubits
    out, total, valid = {}, 0, 0
    for bits, c in counts.items():
        bits = bits.replace(" ", ""); total += c
        bit = lambda q: bits[len(bits) - 1 - q] == "1"
        idx, ok = 0, True
        for i in range(1, n):
            if K == 2:
                z = 1 if bit(variable(i, 0, K)) else 0
            else:
                hot = [k for k in range(K) if bit(variable(i, k, K))]
                if len(hot) != 1:
                    ok = False; break
                z = hot[0]
            idx += z * K ** (i - 1)
        if ok:
            out[idx] = out.get(idx, 0) + c; valid += c
    return out, (valid / total if total else 0.0)


def measured_details(counts, prep):
    """Raw chip readouts, before post-processing: the lowest-cost valid bitstring (zone qubits grouped per
    system, first zone first) and the share of all readouts that were an optimal split."""
    n, K, direct = prep["n"], prep["K"], prep["direct"]
    best_cost, total, best_bits, best_count, optimal, best_idx = None, 0, None, 0, 0, None
    for bits, c in counts.items():
        bits = bits.replace(" ", ""); total += c
        bit = lambda q: bits[len(bits) - 1 - q] == "1"
        idx, ok, groups = 0, True, []
        for i in range(1, n):
            if K == 2:
                z = 1 if bit(variable(i, 0, K)) else 0
                groups.append(str(z))
            else:
                hot = [k for k in range(K) if bit(variable(i, k, K))]
                if len(hot) != 1:
                    ok = False; break
                z = hot[0]
                groups.append("".join("1" if k == z else "0" for k in range(K)))
            idx += z * K ** (i - 1)
        if not ok:
            continue
        cost = float(direct[idx])
        if cost <= float(direct.min()) + 1e-9:
            optimal += c
        if best_cost is None or cost < best_cost - 1e-9 or (abs(cost - best_cost) <= 1e-9 and c > best_count):
            best_cost, best_bits, best_count, best_idx = cost, " ".join(groups), c, idx
    return {"best_measured_bitstring": best_bits, "best_measured_cost": best_cost,
            "optimal_share": round(optimal / total, 5) if total else 0.0, "best_measured_index": best_idx}


def run(payload):
    prep = prepare(payload)
    logical = transpile(prep["circuit"], basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=7)
    probs = evolve(prep["norm"], prep["n"], prep["K"], prep["gamma"], prep["beta"])
    rng = np.random.default_rng(2026)
    samples = np.unique(rng.choice(len(probs), size=SHOTS, p=probs / probs.sum()))
    best_idx = best_after_cleanup(prep, samples)
    return summary(prep, best_idx, {"engine": "qiskit-simulator", "two_qubit_gates_logical": int(logical.count_ops().get("cx", 0)), "shots": SHOTS})
