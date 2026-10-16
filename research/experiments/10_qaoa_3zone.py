import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import circuits as cx, config as cfg, metrics as mt, postprocess as pp, qaoa, scenario as sc, subspace  # noqa: E402

K = 3
COSTS = qaoa.COSTS3
table, refined = pp.refine_all(K)
random_starts = pp.summarize(np.ones(len(table)), refined, table)
sim = AerSimulator()

results = []
for p in cfg.P_VALUES:
    circuit = transpile(cx.build_ansatz_3zone(p), sim)
    for seed in cfg.SEEDS3:
        run = qaoa.run_qaoa_3zone(p, seed)
        counts = np.zeros(len(table))
        for index, count in run.pop("final_counts").items():
            counts[int(index)] = count
        start = time.perf_counter()
        job = SamplerV2(seed=seed).run([(circuit, cx.bind_values(circuit, run["gammas"], run["betas"]))], shots=cfg.SHOTS).result()
        circuit_seconds = time.perf_counter() - start
        qiskit_probs, invalid = mt.counts_to_partition_probs(job[0].data.meas.get_counts(), K)
        ideal = subspace.probs(run["gammas"], run["betas"])
        run["qiskit_circuit_check"] = {"valid_share": 1 - invalid, "p_best": mt.prob_best(qiskit_probs, COSTS),
                                       "hellinger_vs_ideal": mt.hellinger_fidelity(qiskit_probs / qiskit_probs.sum(), ideal), "seconds": circuit_seconds}
        run["post_processing"] = {"qaoa_exact_distribution": pp.summarize(ideal, refined, table),
                                  "qaoa_4096_samples": pp.summarize(counts, refined, table), "random_starts": random_starts}
        results.append(run)
        pq, pr = run["post_processing"]["qaoa_exact_distribution"], random_starts
        print(f"p={p} seed {seed}: P(best) {100 * run['ideal']['p_best']:.2f}% avg cost {run['ideal']['avg_cost']:.1f} | training {run['tuning_seconds']:.1f} s, "
              f"{run['tuning_shots']:,} shots | Qiskit circuit: {100 * (1 - invalid):.0f}% valid shots, P(best) {100 * run['qiskit_circuit_check']['p_best']:.2f}%, {circuit_seconds:.2f} s | "
              f"refined: QAOA success {pq['success_rate']:.2f}, {pq['evals_per_success']:.0f} evals per success; random starts {pr['success_rate']:.2f}, {pr['evals_per_success']:.0f}", flush=True)

print(f"\nstart cost: random starts {random_starts['start_cost']:.1f} (best {COSTS.min():.0f}, worst {COSTS.max():.0f})")
out = ROOT / "results" / "10_qaoa_3zone.json"
out.write_text(json.dumps({
    "experiment": "10_qaoa_3zone", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "settings": {"seeds": cfg.SEEDS3, "shots_opt": cfg.SHOTS_OPT, "shots_final": cfg.SHOTS, "maxiter": cfg.MAXITER, "rhobeg": cfg.RHOBEG, "tol": cfg.TOL,
                 "restarts": cfg.RESTARTS, "optimizer": "COBYLA", "start_range": {"gamma": [-0.6, 0.6], "beta": [-0.8, 0.8]},
                 "simulator": "exact valid-subspace simulation for training (shots drawn from it); Aer SamplerV2 on the Qiskit circuit for the check",
                 "post_processing": "steepest descent over one-system moves, evaluations = cost lookups"},
    "machine": {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__, "numpy": np.__version__},
    "random_starts": random_starts, "runs": results}, indent=2))
print(f"wrote {out.relative_to(ROOT)}")
