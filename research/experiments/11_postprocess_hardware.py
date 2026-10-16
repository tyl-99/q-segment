import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import metrics as mt, postprocess as pp, qaoa  # noqa: E402

K = 2
hardware = json.loads((ROOT / "results" / "07_hardware_final.json").read_text())
source = json.loads((ROOT / "results" / "03_qaoa_repeats.json").read_text())["runs"]
table, refined = pp.refine_all(K)
random_starts = pp.summarize(np.ones(len(table)), refined, table)

report = {"random_starts": random_starts, "jobs": []}
print(f"2 zones, refining every start with one-system moves. Random starts: start cost {random_starts['start_cost']:.1f}, success {random_starts['success_rate']:.2f}, "
      f"{random_starts['mean_evals_per_start']:.1f} lookups per start, {random_starts['evals_per_success']:.1f} lookups per success")
for job in hardware["jobs"]:
    if job["variant"] != "main":
        continue
    rows = []
    for k, seed in enumerate(job["seeds"]):
        run = next(r for r in source[str(job["p"])] if r["seed"] == seed)
        hard = mt.counts_to_probs(job["counts"][k], 7)
        ideal = qaoa.ideal_probs(job["p"], run["gammas"], run["betas"])
        rows.append({"seed": seed, "hardware": pp.summarize(hard, refined, table), "ideal": pp.summarize(ideal, refined, table),
                     "hardware_p_best": mt.prob_best(hard, qaoa.COSTS)})
    mean = lambda who, key: float(np.mean([r[who][key] for r in rows]))
    report["jobs"].append({"p": job["p"], "sets": len(rows), "rows": rows})
    print(f"p={job['p']} ({len(rows)} parameter sets, means): start cost hardware {mean('hardware', 'start_cost'):.1f} / ideal {mean('ideal', 'start_cost'):.1f}; "
          f"success after refinement hardware {mean('hardware', 'success_rate'):.3f} / ideal {mean('ideal', 'success_rate'):.3f}; "
          f"lookups per start {mean('hardware', 'mean_evals_per_start'):.1f} / {mean('ideal', 'mean_evals_per_start'):.1f}; "
          f"lookups per success {mean('hardware', 'evals_per_success'):.1f} / {mean('ideal', 'evals_per_success'):.1f}")
(ROOT / "results" / "11_postprocess_hardware.json").write_text(json.dumps(report, indent=2))
print("wrote results/11_postprocess_hardware.json")
