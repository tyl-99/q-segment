import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import config as cfg, scaling  # noqa: E402

SIZES = (20, 30, 40, 50, 60, 70, 80, 100)
DEGREES = (4, 6)
TIME_LIMIT = 30


env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
rows = []
for degree in DEGREES:
    for n in SIZES:
        timed_out = 0
        for seed in cfg.SEEDS3:
            pairs, weights = scaling.random_instance(n, n * degree // 2, seed)
            result = scaling.solve_exact(n, pairs, weights, env, TIME_LIMIT) | {"n": n, "degree": degree, "pairs": len(pairs), "seed": seed}
            if result["status"] == "OPTIMAL":
                sa = scaling.annealing_time_to_target(n, pairs, weights, result["objective"], seed)
                result["annealing_seconds_to_optimum"] = sa
            else:
                timed_out += 1
            rows.append(result)
            print(f"N={n:>3} degree {degree} pairs {len(pairs):>3} seed {seed}: Gurobi {result['status']} in {result['solve_seconds']:.2f} s (nodes {result['nodes']:.0f}, gap {result['mip_gap']:.3f})"
                  + (f" | annealing reaches the optimum in {result['annealing_seconds_to_optimum']:.2f} s" if result.get("annealing_seconds_to_optimum") is not None else ""), flush=True)
        if timed_out == len(cfg.SEEDS3):
            break

out = ROOT / "results" / "13_scaling_classical.json"
out.write_text(json.dumps({"experiment": "13_scaling_classical", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seeds": cfg.SEEDS3,
                           "time_limit_seconds": TIME_LIMIT, "gurobi_threads": 1, "rows": rows}, indent=2))
print(f"wrote {out.relative_to(ROOT)}")
