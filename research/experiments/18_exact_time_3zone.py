import argparse
import json
import os
import resource
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model, quicksum

K = 3
DENSITY = 0.5
MARGIN_SECONDS = 1800
PROGRESS_EVERY_SECONDS = 60


def random_instance(n, n_pairs, seed):
    """Copy of qsegment.scaling.random_instance (that module imports gurobipy, which Athene does not have)."""
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = sorted(rng.choice(n, size=2, replace=False))
        pairs.add((int(i), int(j)))
    weights = rng.integers(1, 8, size=len(pairs)) * rng.choice([-1, 1], size=len(pairs))
    return sorted(pairs), [int(w) for w in weights]


def write_json(path, data):
    """Write to a temp file first, then rename, so a kill never leaves a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


LOCK = threading.Lock()


def append_line(path, line):
    """Append one JSON line and force it to disk."""
    with LOCK, path.open("a") as f:
        f.write(json.dumps(line) + "\n")
        f.flush()
        os.fsync(f.fileno())


def heartbeat(path, t0, phase, stop):
    """Every minute, record that the job is still alive, even while SCIP is inside one long step."""
    while not stop.wait(PROGRESS_EVERY_SECONDS):
        append_line(path, {"wall_seconds": round(time.perf_counter() - t0, 3), "event": "alive", "phase": phase[0]})


class Progress(Eventhdlr):
    """Append one line per new best split, and one line per minute, to a JSONL file that survives a kill."""

    def __init__(self, path, t0):
        super().__init__()
        self.path, self.t0, self.last = path, t0, 0.0

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexec(self, event):
        now = time.perf_counter() - self.t0
        new_best = event.getType() == SCIP_EVENTTYPE.BESTSOLFOUND
        if not new_best and now - self.last < PROGRESS_EVERY_SECONDS:
            return
        self.last = now
        m = self.model
        best = m.getSolObjVal(m.getBestSol()) if m.getNSols() > 0 else None
        append_line(self.path, {"wall_seconds": round(now, 3), "solve_seconds": round(m.getSolvingTime(), 3), "event": "best" if new_best else "tick",
                                "best_found": best, "bound": m.getDualbound(), "nodes": m.getNNodes()})


def build(n, pairs, weights):
    """One-hot x[i,k]; z[i,j,k] stands for x[i,k]*x[j,k], linearised by the sign of w (exact for binary x)."""
    model = Model()
    x = {(i, k): model.addVar(vtype="B", name=f"x{i}_{k}") for i in range(n) for k in range(K)}
    for i in range(n):
        model.addCons(quicksum(x[i, k] for k in range(K)) == 1)
    model.addCons(x[0, 0] == 1)
    for k in range(2, K):
        model.addCons(x[1, k] == 0)
    terms = []
    for (i, j), w in zip(pairs, weights):
        for k in range(K):
            z = model.addVar(vtype="C", lb=0.0, ub=1.0)
            if w > 0:
                model.addCons(z >= x[i, k] + x[j, k] - 1)
            else:
                model.addCons(z <= x[i, k])
                model.addCons(z <= x[j, k])
            terms.append(w * z)
    model.setObjective(quicksum(terms), "minimize")
    return model, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cap-seconds", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"n{args.n}_s{args.seed}"
    result_path, progress_path = args.out / f"{tag}.json", args.out / f"{tag}_progress.jsonl"
    n_pairs = int(DENSITY * args.n * (args.n - 1) / 2)
    meta = {"experiment": "18_exact_time_3zone", "solver": "SCIP", "n": args.n, "seed": args.seed, "K": K, "density": DENSITY, "pairs": n_pairs,
            "threads": 1, "cap_seconds": args.cap_seconds, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(result_path, meta | {"status": "started"})
    phase, stop = ["build"], threading.Event()
    threading.Thread(target=heartbeat, args=(progress_path, t0, phase, stop), daemon=True).start()
    try:
        pairs, weights = random_instance(args.n, n_pairs, args.seed)
        model, x = build(args.n, pairs, weights)
        build_seconds = time.perf_counter() - t0
        write_json(result_path, meta | {"status": "built", "build_seconds": build_seconds, "variables": model.getNVars(), "constraints": model.getNConss()})

        phase[0] = "solve"
        model.setLogfile(str(args.out / f"{tag}_scip.log"))
        model.setParam("display/freq", 1000)
        model.setParam("limits/time", max(60.0, args.cap_seconds - MARGIN_SECONDS - build_seconds))
        model.setParam("parallel/maxnthreads", 1)
        model.includeEventhdlr(Progress(progress_path, t0), "progress", "logs best splits and a tick per minute")

        start = time.perf_counter()
        model.optimize()
        solve_wall = time.perf_counter() - start

        status = model.getStatus()
        out = meta | {"status": status, "proved_optimal": status == "optimal", "build_seconds": build_seconds, "solve_seconds": solve_wall,
                      "scip_solving_time": model.getSolvingTime(), "time_limit_seconds": model.getParam("limits/time"),
                      "best_found": None, "bound": model.getDualbound(), "gap": model.getGap(), "nodes": model.getNNodes(),
                      "variables": model.getNVars(), "constraints": model.getNConss(), "scip_version": model.version(),
                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if model.getNSols() > 0:
            sol = model.getBestSol()
            zones = [max(range(K), key=lambda k: model.getSolVal(sol, x[i, k])) for i in range(args.n)]
            recomputed = sum(w for (i, j), w in zip(pairs, weights) if zones[i] == zones[j])
            out |= {"best_found": model.getSolObjVal(sol), "recomputed_cost": recomputed, "zones": zones}
        write_json(result_path, out)
        print(json.dumps({k: out[k] for k in ("n", "seed", "status", "solve_seconds", "best_found", "bound", "gap", "nodes")}), flush=True)
    except BaseException as e:
        write_json(result_path, meta | {"status": "crashed", "phase": phase[0], "error": f"{type(e).__name__}: {e}", "wall_seconds": time.perf_counter() - t0,
                                        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
