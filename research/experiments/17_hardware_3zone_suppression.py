import json
import sys
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import SamplerV2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import circuits as cx, config as cfg, hardware as hw, metrics as mt, mitigation as mg, noise, postprocess as pp, qaoa, subspace  # noqa: E402

P = 1
BACKEND = "ibm_kingston"
MAX_SECONDS = 45
OUT = ROOT / "results" / "17_hardware_3zone_suppression.json"
COSTS = qaoa.COSTS3
N_BITS = 21
VARIANTS = [("plain", {}), ("dd_twirl", {"dynamical_decoupling": {"enable": True}, "twirling": {"enable_gates": True, "enable_measure": True}})]


def probs_from_counts(counts):
    v = np.zeros(2 ** N_BITS)
    for bits, c in counts.items():
        v[int(bits.replace(" ", ""), 2)] += c
    return v / v.sum()


def as_counts(vector):
    return {format(i, f"0{N_BITS}b"): float(vector[i]) for i in np.flatnonzero(vector > 1e-12)}


def score(counts, ideal, table, refined):
    probs, invalid = mt.counts_to_partition_probs(counts, 3)
    valid = probs / probs.sum()
    return {"valid_share": 1 - invalid, "p_best_of_valid_shots": mt.prob_best(valid, COSTS), "avg_cost_of_valid_shots": mt.average_cost(valid, COSTS),
            "hellinger_valid_vs_ideal": mt.hellinger_fidelity(valid, ideal), "post_processing": pp.summarize(valid, refined, table)}


runs = [r for r in json.loads((ROOT / "results" / "10_qaoa_3zone.json").read_text())["runs"] if r["p"] == P]
service = hw.get_service()
backend = service.backend(BACKEND)
transpiled, chosen, _ = cx.pick_transpiled(cx.build_ansatz_3zone(P), backend, (2, 3), (0, 1, 2, 3, 4))
physical = noise.measured_qubits(transpiled)
remaining = service.usage()["usage_remaining_seconds"]
print(f"{BACKEND}, 3 zones, p={P}: {len(runs)} parameter sets (seeds {[r['seed'] for r in runs]}), {cfg.SHOTS} shots each, plus 2 readout calibration circuits per job")
print(f"routed circuit: {chosen['two_qubit_gates']} two-qubit gates, depth {chosen['depth']}; measured qubits {physical}")
print(f"jobs: {[v for v, _ in VARIANTS]}, each capped at {MAX_SECONDS} s; QPU allowance remaining {remaining} s")
if "--submit" not in sys.argv:
    print("dry run: nothing submitted. Add --submit to send the jobs.")
    sys.exit(0)

pubs = [(transpiled, cx.bind_values(transpiled, r["gammas"], r["betas"])) for r in runs]
pubs += [mg.calibration_circuit(backend.num_qubits, physical, prepared) for prepared in (0, 1)]
jobs = {name: SamplerV2(mode=backend, options={"max_execution_time": MAX_SECONDS} | opts).run(pubs, shots=cfg.SHOTS) for name, opts in VARIANTS}
record = {"backend": BACKEND, "p": P, "calibration": hw.calibration_summary(backend), "chosen_transpile": chosen, "remaining_before": remaining,
          "job_ids": {name: job.job_id() for name, job in jobs.items()}, "status": "submitted", "seeds": [r["seed"] for r in runs]}
OUT.write_text(json.dumps(record, indent=2))
table, refined = pp.refine_all(3)
results = {}
for name, job in jobs.items():
    counts = [pub.data.meas.get_counts() for pub in job.result()]
    matrices = mg.matrices_from_counts({0: counts[len(runs)], 1: counts[len(runs) + 1]}, N_BITS)
    rows = []
    for r, c in zip(runs, counts[:len(runs)]):
        ideal = subspace.probs(r["gammas"], r["betas"])
        mitigated = as_counts(mg.mitigate(probs_from_counts(c), matrices))
        rows.append({"seed": r["seed"], "ideal_p_best": mt.prob_best(ideal, COSTS), "raw": score(c, ideal, table, refined),
                     "readout_mitigated": score(mitigated, ideal, table, refined), "counts": c})
    results[name] = {"job": hw.job_record(job), "readout_flip_median": float(np.median([1 - (A[0, 0] + A[1, 1]) / 2 for A in matrices])), "rows": rows}
record |= {"status": "done", "results": results, "remaining_after": service.usage()["usage_remaining_seconds"]}
OUT.write_text(json.dumps(record, indent=2))
print(f"allowance left {record['remaining_after']} s")
for name, res in results.items():
    print(f"{name}: QPU {res['job']['usage_seconds']} s, median readout error {100 * res['readout_flip_median']:.2f}%")
    for row in res["rows"]:
        for mode in ("raw", "readout_mitigated"):
            s = row[mode]
            print(f"  seed {row['seed']} {mode:>17}: valid {100 * s['valid_share']:.1f}% | best split {100 * s['p_best_of_valid_shots']:.2f}% of valid (ideal {100 * row['ideal_p_best']:.2f}%) | avg cost {s['avg_cost_of_valid_shots']:.1f} | after post-processing {s['post_processing']['mean_final_cost']:.1f}")
