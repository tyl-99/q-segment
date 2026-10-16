import math
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def random_instance(n, n_pairs, seed):
    """Random 2-zone problem on n systems with n_pairs pairs. Pair weight w = risk - traffic, an integer in -7..7 without 0."""
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = sorted(rng.choice(n, size=2, replace=False))
        pairs.add((int(i), int(j)))
    weights = rng.integers(1, 8, size=len(pairs)) * rng.choice([-1, 1], size=len(pairs))
    return sorted(pairs), [int(w) for w in weights]


def solve_exact(n, pairs, weights, env, time_limit):
    """Minimize the sum of w_ij [x_i == x_j] over x in {0,1}^n (x_0 fixed by symmetry) with Gurobi, one thread."""
    model = gp.Model(env=env)
    x = [model.addVar(vtype=GRB.BINARY) for _ in range(n)]
    model.addConstr(x[0] == 0)
    model.setObjective(gp.quicksum(w * (1 - x[i] - x[j] + 2 * x[i] * x[j]) for (i, j), w in zip(pairs, weights)), GRB.MINIMIZE)
    model.Params.NonConvex = 2
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    start = time.perf_counter()
    model.optimize()
    seconds = time.perf_counter() - start
    return {"status": "OPTIMAL" if model.Status == GRB.OPTIMAL else int(model.Status), "objective": model.ObjVal, "bound": model.ObjBound,
            "mip_gap": model.MIPGap, "nodes": model.NodeCount, "solve_seconds": seconds}


def annealing_time_to_target(n, pairs, weights, target, seed, max_steps=400_000, t_start=8.0, t_end=0.3):
    """Single-system flips with geometric cooling and restarts; seconds until the cost first equals `target` (None if not reached)."""
    rng = np.random.default_rng(seed)
    nbrs = [[] for _ in range(n)]
    for (i, j), w in zip(pairs, weights):
        nbrs[i].append((j, w))
        nbrs[j].append((i, w))
    start = time.perf_counter()
    steps_per_run = 20_000
    cooling = (t_end / t_start) ** (1 / (steps_per_run - 1))
    done = 0
    while done < max_steps:
        x = rng.integers(2, size=n)
        cost = sum(w * (x[i] == x[j]) for (i, j), w in zip(pairs, weights))
        temperature = t_start
        for _ in range(steps_per_run):
            k = int(rng.integers(n))
            delta = sum(w * ((1 - x[k]) == x[j]) - w * (x[k] == x[j]) for j, w in nbrs[k])
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                x[k] = 1 - x[k]
                cost += delta
                if cost <= target + 1e-9:
                    return time.perf_counter() - start
            temperature *= cooling
        done += steps_per_run
    return None


def solve_exact_zones(n, pairs, weights, env, time_limit, K=3):
    """Minimize the sum of w_ij [zone_i == zone_j] over K zones with one-hot binaries and Gurobi, one thread.
    System 0 sits in zone 0 and system 1 in zone 0 or 1, which removes zone-relabeling symmetry without losing any split."""
    model = gp.Model(env=env)
    x = model.addVars(n, K, vtype=GRB.BINARY)
    model.addConstrs(gp.quicksum(x[i, k] for k in range(K)) == 1 for i in range(n))
    model.addConstr(x[0, 0] == 1)
    for k in range(2, K):
        model.addConstr(x[1, k] == 0)
    model.setObjective(gp.quicksum(w * gp.quicksum(x[i, k] * x[j, k] for k in range(K)) for (i, j), w in zip(pairs, weights)), GRB.MINIMIZE)
    model.Params.NonConvex = 2
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    start = time.perf_counter()
    model.optimize()
    seconds = time.perf_counter() - start
    return {"status": "OPTIMAL" if model.Status == GRB.OPTIMAL else int(model.Status), "objective": model.ObjVal, "bound": model.ObjBound,
            "mip_gap": model.MIPGap, "nodes": model.NodeCount, "solve_seconds": seconds}
