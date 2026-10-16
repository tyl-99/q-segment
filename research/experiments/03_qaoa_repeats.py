import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import config as cfg, metrics as mt, qaoa  # noqa: E402

runs = {p: [qaoa.run_qaoa(p, seed) for seed in cfg.SEEDS] for p in cfg.P_VALUES}


def stat(p, getter):
    return mt.summarize([getter(r) for r in runs[p]])


summary = {}
for p in cfg.P_VALUES:
    summary[p] = {
        "p_best": stat(p, lambda r: r["ideal"]["p_best"]),
        "avg_cost": stat(p, lambda r: r["ideal"]["avg_cost"]),
        "approx_ratio": stat(p, lambda r: r["ideal"]["approx_ratio"]),
        "shots_to_hit": stat(p, lambda r: r["ideal"]["shots_to_hit"]),
        "total_shots_to_hit": stat(p, lambda r: r["ideal"]["total_shots_to_hit"]),
        "iterations": stat(p, lambda r: r["iterations"]),
        "tuning_shots": stat(p, lambda r: r["tuning_shots"]),
        "tuning_seconds": stat(p, lambda r: r["tuning_seconds"]),
        "hellinger_shot_floor": stat(p, lambda r: r["final_sample"]["hellinger_vs_ideal"]),
        "final_sample_hit_rate": float(np.mean([r["final_sample"]["hit"] for r in runs[p]])),
        "runs_that_hit_maxiter": int(sum(r["hit_maxiter"] for r in runs[p])),
    }

print(f"{'p':<3}{'P(best) mean':<14}{'std':<8}{'min':<8}{'max':<8}{'avg cost':<10}{'ratio':<8}{'shots/hit':<11}{'tuning shots':<14}{'iter':<7}{'sec':<8}{'F_H floor':<10}{'maxiter':<8}")
for p, s in summary.items():
    print(f"{p:<3}{s['p_best']['mean']:<14.4f}{s['p_best']['std']:<8.4f}{s['p_best']['min']:<8.4f}{s['p_best']['max']:<8.4f}"
          f"{s['avg_cost']['mean']:<10.2f}{s['approx_ratio']['mean']:<8.3f}{s['shots_to_hit']['median']:<11.1f}"
          f"{s['tuning_shots']['mean']:<14.0f}{s['iterations']['mean']:<7.0f}{s['tuning_seconds']['median']:<8.1f}"
          f"{s['hellinger_shot_floor']['mean']:<10.4f}{s['runs_that_hit_maxiter']:<8}")

out = ROOT / "results" / "03_qaoa_repeats.json"
out.write_text(json.dumps({
    "experiment": "03_qaoa_repeats", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "settings": {"seeds": cfg.SEEDS, "shots_opt": cfg.SHOTS_OPT, "shots_final": cfg.SHOTS, "maxiter": cfg.MAXITER,
                 "rhobeg": cfg.RHOBEG, "tol": cfg.TOL, "restarts": cfg.RESTARTS, "optimizer": "COBYLA",
                 "selection": "restart with the lowest average cost on a fresh sample of shots_final shots",
                 "start_range": {"gamma": [-0.6, 0.6], "beta": [-0.8, 0.8]},
                 "simulator": "qiskit_aer SamplerV2, noiseless"},
    "machine": {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
                "qiskit_aer": qiskit_aer.__version__, "numpy": np.__version__},
    "summary": summary, "runs": runs}, indent=2))
print(f"\nwrote {out.relative_to(ROOT)}")
