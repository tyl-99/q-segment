import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import config as cfg, scaling  # noqa: E402

SIZES = (60, 80, 100, 150)
DENSITY = 0.5
TIME_LIMIT = 60
K = 3

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
rows = []
for n in SIZES:
    n_pairs = int(DENSITY * n * (n - 1) / 2)
    for seed in cfg.SEEDS3:
        pairs, weights = scaling.random_instance(n, n_pairs, seed)
        result = scaling.solve_exact_zones(n, pairs, weights, env, TIME_LIMIT, K) | {"n": n, "pairs": n_pairs, "seed": seed}
        rows.append(result)
        print(f"3 zones N={n:>3} pairs {n_pairs:>4} seed {seed}: Gurobi {result['status']} after {result['solve_seconds']:.1f} s | best found {result['objective']:.0f}, bound {result['bound']:.0f}, gap {100 * result['mip_gap']:.1f}% | nodes {result['nodes']:.0f}", flush=True)

out = ROOT / "results" / "16_classical_wall_3zone.json"
out.write_text(json.dumps({"experiment": "16_classical_wall_3zone", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "K": K, "seeds": cfg.SEEDS3,
                           "density": DENSITY, "time_limit_seconds": TIME_LIMIT, "gurobi_threads": 1, "rows": rows}, indent=2))
print(f"wrote {out.relative_to(ROOT)}")
