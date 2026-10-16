import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import baselines as bl, config as cfg, gurobi_solver as gs, metrics as mt, scenario as sc  # noqa: E402

K = 3
METHODS = {"random": bl.random_search, "greedy": bl.greedy, "annealing": bl.simulated_annealing}

brute_runs = [bl.brute_force(K) for _ in cfg.SEEDS3]
brute = brute_runs[0] | {"seconds": [r["seconds"] for r in brute_runs]}
target = brute["best_cost"]

runs = {}
for name, method in METHODS.items():
    runs[name] = []
    for seed in cfg.SEEDS3:
        result = method(K, cfg.BUDGET, target, np.random.default_rng(seed))
        trace = result.pop("trace")
        result["seed"] = seed
        result["best_within"] = mt.best_within(trace, cfg.BUDGETS)
        runs[name].append(result)

gurobi = [gs.solve_3zone() for _ in cfg.SEEDS3]
for r in gurobi:
    r["partition_cost"] = sc.cost(sc.decode(r["index"], K))
gurobi_ok = all(r["status"] == "OPTIMAL" and r["partition_cost"] == target and r["index"] in brute["optimal_indices"] for r in gurobi)

print(f"3 zones: optimum {target:.0f} ({len(brute['optimal_indices'])} optimal partitions), worst {brute['worst_cost']:.0f}, mean {brute['mean_cost']:.1f}; "
      f"brute force over {brute['evals']} partitions: {[f'{s * 1e3:.1f} ms' for s in brute['seconds']]}")
print("per seed (seeds " + ", ".join(map(str, cfg.SEEDS3)) + "):")
for name, rs in runs.items():
    print(f"  {name:<10} hit {sum(r['hit'] for r in rs)}/3 | evaluations to first hit {[r['first_hit_eval'] for r in rs]} | "
          f"seconds to first hit {[None if r['first_hit_seconds'] is None else round(r['first_hit_seconds'] * 1e3, 2) for r in rs]} ms | "
          f"best cost within 10/100/1000/4096: {[list(r['best_within'].values()) for r in rs]}")
print(f"  gurobi     all optimal and equal to brute force: {gurobi_ok} | solve {[round(r['solve_seconds'] * 1e3, 1) for r in gurobi]} ms | "
      f"setup {[round(r['setup_seconds'] * 1e3, 1) for r in gurobi]} ms | nodes {[r['nodes'] for r in gurobi]} | status {[r['status'] for r in gurobi]}")

out = ROOT / "results" / "09_baselines_3zone.json"
out.write_text(json.dumps({
    "experiment": "09_baselines_3zone", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "K": K, "budget": cfg.BUDGET,
    "seeds": cfg.SEEDS3, "settings": {"annealing": {"steps": 200, "t_start": 10.0, "t_end": 0.2}},
    "machine": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
    "brute_force": brute, "gurobi": {"matches_brute_force": gurobi_ok, "runs": gurobi}, "runs": runs}, indent=2))
print(f"\nwrote {out.relative_to(ROOT)}")
