import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import baselines as bl, config as cfg, gurobi_solver as gs, metrics as mt, scenario as sc  # noqa: E402

K = 2
brute = bl.brute_force(K)
runs = [gs.solve() for _ in cfg.SEEDS]

for r in runs:
    r["partition_cost"] = sc.cost(sc.decode(r["index"], K))
matches = all(r["objective"] == brute["best_cost"] and r["partition_cost"] == brute["best_cost"] and r["status"] == "OPTIMAL" for r in runs)
summary = {key: mt.summarize([r[key] for r in runs]) for key in ("solve_seconds", "gurobi_runtime_seconds", "setup_seconds", "nodes")}

print(f"Gurobi {'.'.join(map(str, gp.gurobi.version()))}: {len(runs)} runs, all OPTIMAL and equal to brute force ({brute['best_cost']:.0f}): {matches}")
print(f"optimal partition: {sc.describe(runs[0]['index'], K)}")
print(f"solve time median {summary['solve_seconds']['median'] * 1e3:.2f} ms, setup (license check and model build) median {summary['setup_seconds']['median'] * 1e3:.1f} ms, "
      f"nodes median {summary['nodes']['median']:.0f}, MIP gap {runs[0]['mip_gap']}")

out = ROOT / "results" / "02b_gurobi.json"
out.write_text(json.dumps({
    "experiment": "02b_gurobi", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "K": K,
    "gurobi": ".".join(map(str, gp.gurobi.version())), "threads": 1, "matches_brute_force": matches,
    "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()},
    "summary": summary, "runs": runs}, indent=2))
print(f"\nwrote {out.relative_to(ROOT)}")
