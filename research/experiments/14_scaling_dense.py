import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import config as cfg, scaling  # noqa: E402

SIZES = (30, 40, 50)
PAIRS = (150, 250, 350, 450, 600)
TIME_LIMIT = 30

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
rows = []
for n in SIZES:
    for n_pairs in PAIRS:
        if n_pairs > n * (n - 1) // 2:
            continue
        for seed in cfg.SEEDS3:
            pairs, weights = scaling.random_instance(n, n_pairs, seed)
            result = scaling.solve_exact(n, pairs, weights, env, TIME_LIMIT) | {"n": n, "pairs": n_pairs, "seed": seed}
            rows.append(result)
            print(f"N={n:>3} pairs {n_pairs:>3} seed {seed}: Gurobi {result['status']} in {result['solve_seconds']:.2f} s (nodes {result['nodes']:.0f}, gap {result['mip_gap']:.3f})", flush=True)

out = ROOT / "results" / "14_scaling_dense.json"
out.write_text(json.dumps({"experiment": "14_scaling_dense", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seeds": cfg.SEEDS3,
                           "time_limit_seconds": TIME_LIMIT, "gurobi_threads": 1, "rows": rows}, indent=2))
print(f"wrote {out.relative_to(ROOT)}")
