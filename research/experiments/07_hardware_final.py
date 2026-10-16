import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import config as cfg, hardware as hw, metrics as mt, mitigation as mg, noise, qaoa  # noqa: E402

VARIANTS = [("main", 1, {}, 20), ("main", 2, {}, 20),
            ("dd", 2, {"dynamical_decoupling": {"enable": True}}, 5),
            ("dd_twirl", 2, {"dynamical_decoupling": {"enable": True}, "twirling": {"enable_gates": True, "enable_measure": True}}, 5)]
SAFETY = 1.3
RESERVE_SECONDS = 60
OUT = ROOT / "results" / "07_hardware_final.json"

probe = json.loads((ROOT / "results" / "06_hardware_probe.json").read_text())
short, long = probe["jobs"]
name = probe["backend"]
source = json.loads((ROOT / "results" / "03_qaoa_repeats.json").read_text())["runs"]
service = hw.get_service()
backend = service.backend(name)
worst_overhead = short["usage_seconds"]
worst_rate = long["usage_seconds"] / (3 * long["shots"])
predict = lambda sets: worst_overhead + worst_rate * cfg.SHOTS * (sets + 2)


def affordable_sets(remaining, cap):
    for sets in range(cap, 0, -1):
        if SAFETY * predict(sets) <= remaining - RESERVE_SECONDS:
            return sets
    return 0


remaining = service.usage()["usage_remaining_seconds"]
print(f"backend {name} | allowance remaining {remaining} s")
print(f"probe usage was {short['usage_seconds']} s and {long['usage_seconds']} s (whole seconds only), so the plan uses a worst case: "
      f"{worst_overhead} s per job plus {worst_rate * 1e3:.2f} ms per circuit-shot, measured on p=2 (p=1 is shorter)")
print(f"planned jobs, in this order, each only if it still fits with a safety factor of {SAFETY} and {RESERVE_SECONDS} s kept in reserve:")
for variant, p, _, cap in VARIANTS:
    print(f"  {variant:<9} p={p}: up to {cap} parameter sets, worst-case prediction {predict(cap):.0f} s")
print(f"worst case for all four jobs: {sum(predict(cap) for *_, cap in VARIANTS):.0f} s of {remaining} s")
if "--submit" not in sys.argv:
    print("dry run: nothing submitted. Add --submit to send the jobs.")
    sys.exit(0)

sim = hw.noise_simulator(backend)
record = {"backend": name, "calibration_at_start": hw.calibration_summary(backend), "remaining_before": remaining, "jobs": []}
for variant, p, options, cap in VARIANTS:
    remaining = service.usage()["usage_remaining_seconds"]
    sets = affordable_sets(remaining, cap)
    if sets == 0:
        print(f"stop: {variant} p={p} does not fit in the remaining {remaining} s")
        break
    runs = hw.pick_parameter_sets(source[str(p)], sets)
    transpiled, chosen, _ = hw.transpile_for(backend, p)
    time_cap = int(1.5 * predict(sets)) + 5
    job, physical = hw.submit(backend, transpiled, runs, cfg.SHOTS, {"max_execution_time": time_cap} | options)
    entry = {"variant": variant, "p": p, "sets": sets, "seeds": [r["seed"] for r in runs], "options": options, "chosen_transpile": chosen,
             "physical_qubits": physical, "job_id": job.job_id(), "status": "submitted"}
    record["jobs"].append(entry)
    OUT.write_text(json.dumps(record, indent=2))
    data = hw.read(job, sets)
    entry |= hw.job_record(job) | {"status": "done", "counts": data["counts"], "results": []}
    for k, run in enumerate(runs):
        ideal = qaoa.ideal_probs(p, run["gammas"], run["betas"])
        raw = data["probs"][k]
        row = {"seed": run["seed"], "ideal_p_best": mt.prob_best(ideal, qaoa.COSTS), "hardware_raw": mt.score(raw, ideal, qaoa.COSTS),
               "hardware_mitigated": mt.score(mg.mitigate(raw, data["matrices"]), ideal, qaoa.COSTS)}
        if variant == "main":
            same_day = noise.noisy_probs(sim, transpiled, run["gammas"], run["betas"], cfg.SHOTS, run["seed"], qaoa.N_QUBITS)
            row["same_day_simulation"] = mt.score(same_day, ideal, qaoa.COSTS)
        entry["results"].append(row)
    OUT.write_text(json.dumps(record, indent=2))

    mean = lambda mode, key: np.mean([r[mode][key] for r in entry["results"]])
    line = f"{variant:<9} p={p} sets={sets} QPU {entry['usage_seconds']} s | ideal P(best) {np.mean([r['ideal_p_best'] for r in entry['results']]):.4f}"
    for mode in ("hardware_raw", "hardware_mitigated", "same_day_simulation"):
        if mode in entry["results"][0]:
            line += f" | {mode}: F_H {mean(mode, 'hellinger'):.4f}, P(best) {mean(mode, 'p_best'):.4f}"
    print(line, flush=True)

record["remaining_after"] = service.usage()["usage_remaining_seconds"]
OUT.write_text(json.dumps(record, indent=2))
print(f"allowance remaining after: {record['remaining_after']} s\nwrote {OUT.relative_to(ROOT)}")
