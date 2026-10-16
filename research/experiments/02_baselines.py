import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import baselines as bl, config as cfg, metrics as mt, scenario as sc  # noqa: E402

K = 2
METHODS = {"random": bl.random_search, "greedy": bl.greedy, "annealing": bl.simulated_annealing}

brute_runs = [bl.brute_force(K) for _ in cfg.SEEDS]
brute = brute_runs[0] | {"seconds": mt.summarize([r["seconds"] for r in brute_runs])}
target = brute["best_cost"]

runs, summary = {}, {}
for name, method in METHODS.items():
    runs[name] = []
    for seed in cfg.SEEDS:
        result = method(K, cfg.BUDGET, target, np.random.default_rng(seed))
        trace = result.pop("trace")
        result["seed"] = seed
        result["best_within"] = mt.best_within(trace, cfg.BUDGETS)
        runs[name].append(result)
    hits = [r for r in runs[name] if r["hit"]]
    summary[name] = {
        "success_rate": len(hits) / len(runs[name]),
        "evals_to_first_hit": mt.summarize([r["first_hit_eval"] for r in hits]) if hits else None,
        "seconds_to_first_hit": mt.summarize([r["first_hit_seconds"] for r in hits]) if hits else None,
        "seconds_total": mt.summarize([r["seconds"] for r in runs[name]]),
        "best_within": {str(n): mt.summarize([r["best_within"][n] for r in runs[name]]) for n in cfg.BUDGETS},
    }

print(f"optimum {target:.0f}, worst {brute['worst_cost']:.0f}, mean {brute['mean_cost']:.1f}, optimal partitions {len(brute['optimal_indices'])}, "
      f"brute force {brute['seconds']['median'] * 1e3:.2f} ms ({brute['evals']} evaluations)")
print(f"random: chance of a hit per sample exactly 1/{2 ** len(sc.FREE)}")
header = "".join(f"best@{n:<5}" for n in cfg.BUDGETS)
print(f"{'method':<10}{'hit rate':<10}{'evals to hit':<14}{'sec to hit':<12}{'sec total':<11}{header}")
for name, s in summary.items():
    e = s["evals_to_first_hit"]["median"] if s["evals_to_first_hit"] else float("nan")
    t = s["seconds_to_first_hit"]["median"] if s["seconds_to_first_hit"] else float("nan")
    best = "".join(f"{s['best_within'][str(n)]['mean']:<10.1f}" for n in cfg.BUDGETS)
    print(f"{name:<10}{s['success_rate']:<10.2f}{e:<14.0f}{t:<12.4f}{s['seconds_total']['median']:<11.3f}{best}")

out = ROOT / "results" / "02_baselines.json"
out.write_text(json.dumps({
    "experiment": "02_baselines", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "K": K, "budget": cfg.BUDGET, "seeds": cfg.SEEDS,
    "settings": {"annealing": {"steps": 200, "t_start": 10.0, "t_end": 0.2}, "greedy": "steepest descent over all one-system moves, random restarts"},
    "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(), "numpy": np.__version__},
    "brute_force": brute, "random_hit_probability_per_sample": 1 / 2 ** len(sc.FREE),
    "summary": summary, "runs": runs}, indent=2))
print(f"\nwrote {out.relative_to(ROOT)}")
