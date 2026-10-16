import json
import sys
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import SamplerV2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import circuits as cx, config as cfg, hardware as hw, metrics as mt, postprocess as pp, qaoa, subspace  # noqa: E402

P = 1
BACKEND = "ibm_kingston"
MAX_SECONDS = 45
OUT = ROOT / "results" / "12_hardware_3zone.json"
COSTS = qaoa.COSTS3

runs = [r for r in json.loads((ROOT / "results" / "10_qaoa_3zone.json").read_text())["runs"] if r["p"] == P]
service = hw.get_service()
backend = service.backend(BACKEND)
transpiled, chosen, _ = cx.pick_transpiled(cx.build_ansatz_3zone(P), backend, (2, 3), (0, 1, 2, 3, 4))
remaining = service.usage()["usage_remaining_seconds"]
random_valid = (3 / 8) ** 7

print(f"{BACKEND}, 3 zones, p={P}: {len(runs)} parameter sets (seeds {[r['seed'] for r in runs]}), {cfg.SHOTS} shots each, no readout calibration circuits")
print(f"routed circuit: {chosen['two_qubit_gates']} two-qubit gates, depth {chosen['depth']} (level {chosen['level']}, seed {chosen['seed']})")
print(f"QPU allowance remaining {remaining} s; the job is capped at {MAX_SECONDS} s. A uniformly random 21-bit string is valid with probability {random_valid:.4%}")
if "--submit" not in sys.argv:
    print("dry run: nothing submitted. Add --submit to send the job.")
    sys.exit(0)

job = SamplerV2(mode=backend, options={"max_execution_time": MAX_SECONDS}).run(
    [(transpiled, cx.bind_values(transpiled, r["gammas"], r["betas"])) for r in runs], shots=cfg.SHOTS)
record = {"backend": BACKEND, "p": P, "calibration": hw.calibration_summary(backend), "chosen_transpile": chosen, "remaining_before": remaining,
          "random_bitstring_valid_share": random_valid, "job_id": job.job_id(), "status": "submitted", "seeds": [r["seed"] for r in runs]}
OUT.write_text(json.dumps(record, indent=2))
counts = [pub.data.meas.get_counts() for pub in job.result()]
table, refined = pp.refine_all(3)
rows = []
for r, c in zip(runs, counts):
    probs, invalid = mt.counts_to_partition_probs(c, 3)
    valid = probs / probs.sum() if probs.sum() > 0 else probs
    ideal = subspace.probs(r["gammas"], r["betas"])
    rows.append({"seed": r["seed"], "valid_share": 1 - invalid, "p_best_of_all_shots": mt.prob_best(probs, COSTS),
                 "p_best_of_valid_shots": mt.prob_best(valid, COSTS), "avg_cost_of_valid_shots": mt.average_cost(valid, COSTS),
                 "ideal_p_best": mt.prob_best(ideal, COSTS), "ideal_avg_cost": mt.average_cost(ideal, COSTS),
                 "hellinger_valid_vs_ideal": mt.hellinger_fidelity(valid, ideal),
                 "post_processing": pp.summarize(valid, refined, table) if probs.sum() > 0 else None, "counts": c})
record |= hw.job_record(job) | {"status": "done", "results": rows, "remaining_after": service.usage()["usage_remaining_seconds"]}
OUT.write_text(json.dumps(record, indent=2))
print(f"QPU {record['usage_seconds']} s | allowance left {record['remaining_after']} s")
for row in rows:
    pp_ = row["post_processing"]
    print(f"seed {row['seed']}: valid shots {100 * row['valid_share']:.2f}% (random bit strings {100 * random_valid:.2f}%) | best partition: {100 * row['p_best_of_all_shots']:.3f}% of all shots, "
          f"{100 * row['p_best_of_valid_shots']:.2f}% of valid shots (ideal {100 * row['ideal_p_best']:.2f}%) | avg cost of valid shots {row['avg_cost_of_valid_shots']:.1f} (ideal {row['ideal_avg_cost']:.1f}, random 43.0)"
          + (f" | refined success {pp_['success_rate']:.2f}, {pp_['evals_per_success']:.0f} lookups per success" if pp_ else ""))
