import time

import gurobipy as gp
from gurobipy import GRB

from qsegment import formulation as fm, scenario as sc


def solve():
    """Solve the 2-zone QUBO exactly with Gurobi. Returns the objective, the partition index and timings."""
    const, linear, quad = fm.qubo_2zone()
    t0 = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("qsegment_2zone", env=env)
    x = {j: model.addVar(vtype=GRB.BINARY, name=sc.SYSTEMS[j]) for j in sc.FREE}
    model.setObjective(const + gp.quicksum(c * x[j] for j, c in linear.items())
                       + gp.quicksum(c * x[i] * x[j] for (i, j), c in quad.items()), GRB.MINIMIZE)
    model.Params.NonConvex = 2
    model.Params.Threads = 1
    setup_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    model.optimize()
    solve_seconds = time.perf_counter() - t1
    result = {"status": "OPTIMAL" if model.Status == GRB.OPTIMAL else int(model.Status), "objective": model.ObjVal,
              "index": sum(int(round(x[j].X)) << (j - 1) for j in sc.FREE), "mip_gap": model.MIPGap,
              "nodes": model.NodeCount, "gurobi_runtime_seconds": model.Runtime,
              "setup_seconds": setup_seconds, "solve_seconds": solve_seconds}
    model.dispose()
    env.dispose()
    return result


def solve_3zone():
    """Solve the 3-zone problem exactly with one-hot binary variables. Returns the objective, the partition index and timings."""
    t0 = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("qsegment_3zone", env=env)
    y = {(i, k): model.addVar(vtype=GRB.BINARY, name=f"{sc.SYSTEMS[i]}_{k}") for i in sc.FREE for k in range(3)}
    for i in sc.FREE:
        model.addConstr(gp.quicksum(y[i, k] for k in range(3)) == 1)
    objective = gp.LinExpr(float(sc.total_traffic()))
    for i, j in sc.PAIRS:
        w = sc.net_weight(i, j)
        if i == sc.FIXED:
            objective += w * y[j, 0]
        else:
            objective += gp.quicksum(w * y[i, k] * y[j, k] for k in range(3))
    model.setObjective(objective, GRB.MINIMIZE)
    model.Params.NonConvex = 2
    model.Params.Threads = 1
    setup_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    model.optimize()
    solve_seconds = time.perf_counter() - t1
    zones = {i: next(k for k in range(3) if y[i, k].X > 0.5) for i in sc.FREE}
    result = {"status": "OPTIMAL" if model.Status == GRB.OPTIMAL else int(model.Status), "objective": model.ObjVal,
              "index": sum(zones[i] * 3 ** (i - 1) for i in sc.FREE), "mip_gap": model.MIPGap, "nodes": model.NodeCount,
              "gurobi_runtime_seconds": model.Runtime, "setup_seconds": setup_seconds, "solve_seconds": solve_seconds}
    model.dispose()
    env.dispose()
    return result
