import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import hardware as hw, metrics as mt, mitigation as mg, qaoa  # noqa: E402

P = 2
PROBE_SHOTS = (256, 1024)
MAX_SECONDS_PER_JOB = 30
OUT = ROOT / "results" / "06_hardware_probe.json"

name = json.loads((ROOT / "results" / "05_backend_choice.json").read_text())["chosen"]
runs = json.loads((ROOT / "results" / "03_qaoa_repeats.json").read_text())["runs"][str(P)]
service = hw.get_service()
backend = service.backend(name)
run = hw.pick_parameter_sets(runs, 1)[0]
transpiled, chosen, _ = hw.transpile_for(backend, P)
remaining = service.usage()["usage_remaining_seconds"]

print(f"backend {name}, p={P}, seed {run['seed']}, {chosen['two_qubit_gates']} two-qubit gates, depth {chosen['depth']}")
print(f"2 jobs, {PROBE_SHOTS} shots per circuit, 3 circuits per job (1 QAOA circuit and 2 readout calibration circuits)")
print(f"QPU allowance remaining: {remaining} s | each job is capped at {MAX_SECONDS_PER_JOB} s of QPU time, so the probe can use at most {MAX_SECONDS_PER_JOB * len(PROBE_SHOTS)} s")
if "--submit" not in sys.argv:
    print("dry run: nothing submitted. Add --submit to send the jobs.")
    sys.exit(0)

record = {"backend": name, "p": P, "seed": run["seed"], "calibration": hw.calibration_summary(backend), "chosen_transpile": chosen,
          "remaining_before": remaining, "jobs": []}
ideal = qaoa.ideal_probs(P, run["gammas"], run["betas"])
for shots in PROBE_SHOTS:
    job, physical = hw.submit(backend, transpiled, [run], shots, {"max_execution_time": MAX_SECONDS_PER_JOB})
    record["jobs"].append({"shots": shots, "job_id": job.job_id(), "physical_qubits": physical, "status": "submitted"})
    OUT.write_text(json.dumps(record, indent=2))
    data = hw.read(job, 1)
    raw, mitigated = data["probs"][0], mg.mitigate(data["probs"][0], data["matrices"])
    record["jobs"][-1] |= hw.job_record(job) | {"status": "done", "counts": data["counts"],
                                                "raw": mt.score(raw, ideal, qaoa.COSTS), "mitigated": mt.score(mitigated, ideal, qaoa.COSTS)}
    OUT.write_text(json.dumps(record, indent=2))
    j = record["jobs"][-1]
    print(f"{shots} shots: QPU {j['usage_seconds']} s | F_H raw {j['raw']['hellinger']:.4f} mitigated {j['mitigated']['hellinger']:.4f} | "
          f"P(best) raw {j['raw']['p_best']:.4f} mitigated {j['mitigated']['p_best']:.4f} (ideal {mt.prob_best(ideal, qaoa.COSTS):.4f})", flush=True)

(j1, j2) = record["jobs"]
per_shot = (j2["usage_seconds"] - j1["usage_seconds"]) / (3 * (j2["shots"] - j1["shots"]))
overhead = j1["usage_seconds"] - per_shot * 3 * j1["shots"]
record["cost_model"] = {"overhead_seconds": overhead, "seconds_per_circuit_shot": per_shot,
                        "note": "usage = overhead + seconds_per_circuit_shot x (circuits per job x shots); fitted through exactly two jobs, so no error estimate"}
OUT.write_text(json.dumps(record, indent=2))
print(f"cost model from two jobs: {overhead:.2f} s per job + {per_shot * 1e3:.3f} ms per circuit-shot")
