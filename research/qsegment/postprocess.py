import numpy as np

from qsegment import baselines as bl, scenario as sc


def refine(index, K, table):
    """Steepest descent over one-system moves from `index`. Returns (final partition, cost lookups, moves).
    The start costs one lookup and every move costs one lookup per neighbor."""
    evals, moves, value = 1, 0, table[index]
    while True:
        best, best_value = None, value
        for candidate in bl.neighbors(index, K):
            evals += 1
            if table[candidate] < best_value:
                best, best_value = candidate, table[candidate]
        if best is None:
            return index, evals, moves
        index, value, moves = best, best_value, moves + 1


def refine_all(K):
    """Refinement of every partition as a start, so any distribution of starts can be scored by weighting."""
    table = sc.cost_table(K)
    return table, [refine(i, K, table) for i in range(K ** len(sc.FREE))]


def summarize(weights, refined, table):
    """Outcome of refining starts drawn with probability `weights` (one weight per partition)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    best = min(table)
    final_cost = np.array([table[r[0]] for r in refined])
    evals = np.array([r[1] for r in refined], dtype=float)
    moves = np.array([r[2] for r in refined], dtype=float)
    success = float(w @ (final_cost == best))
    mean_evals = float(w @ evals)
    return {"start_cost": float(w @ np.array(table, dtype=float)), "success_rate": success, "mean_final_cost": float(w @ final_cost),
            "mean_evals_per_start": mean_evals, "mean_moves": float(w @ moves),
            "evals_per_success": mean_evals / success if success else float("inf")}
