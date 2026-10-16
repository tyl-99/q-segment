import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import formulation as fm, scenario as sc  # noqa: E402

SEED = 1
EXPECTED_BEST = {"A": ["nurse", "mri", "portal", "email"], "B": ["pacs", "pharm", "ehr", "billing"]}
n = len(sc.FREE)
checks = []


def check(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


def numpy_qaoa(cost_vec, gammas, betas):
    """Independent reference: |+>^n, then exp(-i g C) and a product of RX(2b) for each layer."""
    psi = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = np.exp(-1j * g * cost_vec) * psi
        c, s = np.cos(b), -1j * np.sin(b)
        psi = psi.reshape([2] * n)
        for ax in range(n):
            a, d = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([c * a + s * d, s * a + c * d], axis=ax)
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2


def qiskit_probs(H, gammas, betas):
    qc = QAOAAnsatz(cost_operator=H, reps=len(gammas))
    values = {}
    for par in qc.parameters:
        kind, k = re.fullmatch(r"(β|γ)\[(\d+)\]", par.name).groups()
        values[par] = (betas if kind == "β" else gammas)[int(k)]
    return Statevector(qc.assign_parameters(values)).probabilities()


costs = np.array(sc.cost_table(2), dtype=float)
best = int(np.argmin(costs))
check("problem size", len(costs) == 128, f"{len(costs)} partitions, {n} free systems")
check("optimum cost is 26 and unique", costs.min() == 26 and (costs == 26).sum() == 1,
      f"min {costs.min():.0f}, count at min {(costs == costs.min()).sum()}, worst {costs.max():.0f}, mean {costs.mean():.1f}")
check("optimal partition as expected", sc.describe(best, 2) == EXPECTED_BEST, str(sc.describe(best, 2)))

qubo = fm.qubo_2zone()
diff_qubo = max(abs(fm.qubo_value(qubo, b) - costs[b]) for b in range(128))
check("QUBO equals cost on all 128", diff_qubo == 0, f"max difference {diff_qubo}; constant {qubo[0]:.0f}, "
      f"{sum(1 for c in qubo[1].values() if c)} nonzero linear, {len(qubo[2])} quadratic terms")

H, const = fm.ising_2zone()
diag = np.real(np.diag(H.to_matrix()))
diff_ising = np.max(np.abs(diag + const - costs))
check("Ising H + constant equals cost on all 128", diff_ising < 1e-12, f"max difference {diff_ising:.1e}; constant {const}")

nzz = sum(1 for p in H.paulis if p.to_label().count("Z") == 2)
nz = sum(1 for p in H.paulis if p.to_label().count("Z") == 1)
check("Hamiltonian has 13 ZZ and 5 Z terms", (nzz, nz) == (13, 5), f"{nzz} ZZ, {nz} Z, {H.num_qubits} qubits")

rng = np.random.default_rng(SEED)
for p in (1, 2, 3):
    gammas, betas = rng.uniform(-0.5, 0.5, p), rng.uniform(-0.8, 0.8, p)
    delta = np.max(np.abs(qiskit_probs(H, gammas, betas) - numpy_qaoa(costs, gammas, betas)))
    check(f"Qiskit circuit equals numpy reference, p={p}", delta < 1e-10, f"max probability difference {delta:.1e}")

width = max(len(c["check"]) for c in checks)
for c in checks:
    print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']:<{width}}  {c['detail']}")

out = ROOT / "results" / "01_verify.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "experiment": "01_verify", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
    "versions": {"qiskit": qiskit.__version__, "numpy": np.__version__},
    "checks": checks, "all_pass": all(c["pass"] for c in checks)}, indent=2, ensure_ascii=False))
print(f"\nwrote {out.relative_to(ROOT)}")
sys.exit(0 if all(c["pass"] for c in checks) else 1)
