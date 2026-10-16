import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider as fp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import circuits as cx, formulation as fm, scenario as sc, subspace  # noqa: E402

SEED = 1
checks, gates = [], []


def check(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


costs = np.array(sc.cost_table(3), dtype=float)
best = costs.min()
optimal = [int(i) for i in np.flatnonzero(costs == best)]
swap = lambda z: tuple({1: 2, 2: 1}.get(v, v) for v in z)
check("problem size", len(costs) == 2187, f"{len(costs)} valid partitions of 7 free systems into 3 zones")
check("optimum cost is 25 with 2 optimal partitions", best == 25 and len(optimal) == 2, f"best {best:.0f}, optimal partitions {len(optimal)}, worst {costs.max():.0f}, mean {costs.mean():.1f}")
check("the two optimal partitions differ only by swapping zones B and C",
      swap(sc.decode(optimal[0], 3)) == sc.decode(optimal[1], 3), f"{sc.describe(optimal[0], 3)} and {sc.describe(optimal[1], 3)}")

H, const = fm.ising_3zone()
basis_of = lambda index: sum(1 << fm.qubit(system, zone) for system, zone in zip(sc.FREE, sc.decode(index, 3)[1:]))
valid_basis = np.array([basis_of(i) for i in range(len(costs))])
masks = [sum(1 << q for q, ch in enumerate(reversed(p.to_label())) if ch == "Z") for p in H.paulis]
coeffs = np.real(H.coeffs)
energy = lambda b: sum(c * (-1) ** bin(b & m).count("1") for c, m in zip(coeffs, masks))
diff = max(abs(energy(valid_basis[i]) + const - costs[i]) for i in range(len(costs)))
n_zz = sum(1 for m in masks if bin(m).count("1") == 2)
check("Ising H + constant equals cost on all 2187 valid states", diff < 1e-9, f"max difference {diff:.1e}; constant {const}; {n_zz} ZZ and {len(masks) - n_zz} Z terms on {H.num_qubits} qubits")

w = QuantumCircuit(3)
cx.w_state(w, 0, 1, 2)
amp = Statevector(w).data
target = np.zeros(8, complex)
target[[1, 2, 4]] = 1 / np.sqrt(3)
w_overlap = abs(np.vdot(amp, target))
check("W-state circuit gives equal weight on the three one-hot states", abs(w_overlap - 1) < 1e-12, f"overlap with the W state {w_overlap:.12f}")
notes = []
for beta in (0.1, 0.3, 0.8):
    m = QuantumCircuit(3)
    cx.w_state(m, 0, 1, 2)
    cx.ring_mixer(m, beta, 0, 1, 2)
    notes.append(f"ring mixer at beta={beta}: overlap of the W state with itself afterwards {abs(np.vdot(amp, Statevector(m).data)):.4f}")

sim = AerSimulator(method="statevector")
rng = np.random.default_rng(SEED)
for p in (1, 2, 3):
    gammas, betas = rng.uniform(-0.5, 0.5, p), rng.uniform(-0.8, 0.8, p)
    circuit = cx.build_ansatz_3zone(p, measure=False)
    values = cx.bind_values(circuit, gammas, betas)
    bound = circuit.assign_parameters(dict(zip(circuit.parameters, values)))
    bound.save_statevector()
    start = time.perf_counter()
    state = sim.run(transpile(bound, sim)).result().get_statevector().data
    seconds = time.perf_counter() - start
    full = np.abs(state) ** 2
    delta = np.max(np.abs(full[valid_basis] - subspace.probs(gammas, betas)))
    check(f"Qiskit circuit equals the exact valid-subspace simulation, p={p}", delta < 1e-10 and abs(1 - full[valid_basis].sum()) < 1e-10,
          f"max probability difference {delta:.1e}; probability outside the valid states {1 - full[valid_basis].sum():.1e}; Aer statevector {seconds:.2f} s")

fez = fp.FakeFez()
BASIS = ["cx", "u"]
for p in (1, 2, 3):
    circuit = cx.build_ansatz_3zone(p, measure=False)
    logical = transpile(circuit, basis_gates=BASIS, optimization_level=1).count_ops().get("cx", 0)
    start = time.perf_counter()
    transpiled, chosen, _ = cx.pick_transpiled(circuit, fez, (2,), (0, 1, 2))
    gates.append({"design": "W-state start, XXPlusYY mixer", "p": p, "cx_before_routing": logical, "two_qubit_gates_fake_fez": chosen["two_qubit_gates"],
                  "depth": chosen["depth"], "seconds": time.perf_counter() - start})

H3, _ = fm.ising_3zone()
init = QuantumCircuit(21)
for k in range(7):
    init.x(3 * k)
mixer = SparsePauliOp.from_sparse_list([(pauli, [3 * k + a, 3 * k + b], 0.5) for k in range(7) for a, b in ((0, 1), (1, 2), (2, 0)) for pauli in ("XX", "YY")], num_qubits=21)
old = QAOAAnsatz(cost_operator=H3, mixer_operator=mixer, initial_state=init, reps=1)
logical = transpile(old.decompose(reps=3), basis_gates=BASIS, optimization_level=1).count_ops().get("cx", 0)
_, chosen, _ = cx.pick_transpiled(old, fez, (2,), (0, 1, 2))
gates.append({"design": "old: basis-state start, PauliEvolution mixer", "p": 1, "cx_before_routing": logical,
              "two_qubit_gates_fake_fez": chosen["two_qubit_gates"], "depth": chosen["depth"]})

width = max(len(c["check"]) for c in checks)
for c in checks:
    print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']:<{width}}  {c['detail']}")
print("\nnotes (measured, not pass or fail):", *notes, sep="\n  ")
print("\ntwo-qubit gates (before routing: CX; after: FakeFez, level 2, best of 3 seeds)")
for g in gates:
    print(f"  p={g['p']} {g['design']:<48} CX before routing {g['cx_before_routing']:>4} | FakeFez {g['two_qubit_gates_fake_fez']:>4} gates, depth {g['depth']}")

out = ROOT / "results" / "08_verify_3zone.json"
out.write_text(json.dumps({"experiment": "08_verify_3zone", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
                           "versions": {"qiskit": qiskit.__version__, "numpy": np.__version__}, "checks": checks, "notes": notes, "gate_counts": gates,
                           "all_pass": all(c["pass"] for c in checks)}, indent=2, ensure_ascii=False))
print(f"\nwrote {out.relative_to(ROOT)}")
sys.exit(0 if all(c["pass"] for c in checks) else 1)
