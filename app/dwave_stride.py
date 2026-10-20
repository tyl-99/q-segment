import os
import threading
import time
import uuid

import numpy as np

from app import ibm, solver

JOBS, LOCK, SUBMITTED = {}, threading.Lock(), []


def token():
    """Leap token from the DWAVE_API_TOKEN variable, else from the local dwave config file."""
    if os.environ.get("DWAVE_API_TOKEN"):
        return os.environ["DWAVE_API_TOKEN"]
    try:
        from dwave.cloud.config import load_config
        return load_config().get("token")
    except Exception:
        return None


def available():
    return bool(token()) and bool(ibm.settings()["code"])


def take_slot():
    """At most DWAVE_MAX_RUNS_PER_HOUR Stride runs per rolling hour, so a leaked code cannot drain the Leap allowance."""
    now, limit = time.time(), int(os.environ.get("DWAVE_MAX_RUNS_PER_HOUR", "20"))
    with LOCK:
        SUBMITTED[:] = [t for t in SUBMITTED if now - t < 3600]
        if len(SUBMITTED) >= limit:
            return False
        SUBMITTED.append(now)
        return True


def split_cost(zones, links):
    """Traffic on links that cross zones plus risk on links kept inside a zone."""
    return sum(t if zones[i] != zones[j] else r for i, j, t, r in links)


def relabel(zones):
    """Zone letters in order of first appearance, so system 0 is always in zone A."""
    seen = {}
    return [seen.setdefault(z, len(seen)) for z in zones]


def build_model(n, links, K):
    """One zone number (0..K-1) per system; minimize the sum of w = risk - traffic over pairs in the same zone."""
    from dwave.optimization import Model
    keep = [(i, j, r - t) for i, j, t, r in links if r != t]
    m = Model()
    z = m.integer(n, lower_bound=0, upper_bound=K - 1)
    if keep:
        I, J, W = (np.array(c) for c in zip(*keep))
        m.minimize((m.constant(W.astype(float)) * (z[m.constant(I)] == z[m.constant(J)])).sum())
    else:
        m.minimize(m.constant(0.0) * z.sum())
    return m, z


def solve(names, links, K, label="qsegment-web", time_limit=None):
    from dwave.system import LeapHybridNLSampler
    n = len(names)
    m, z = build_model(n, links, K)
    sampler = LeapHybridNLSampler(token=token())
    tmin = sampler.estimated_min_time_limit(m)
    limit = max(tmin, min(float(time_limit or 0), 60.0))
    t0 = time.time()
    res = sampler.sample(m, time_limit=limit, label=label).result()
    wall = time.time() - t0
    mine = not m.is_locked()
    if mine:
        m.lock()
    try:
        best = min(range(m.states.size()), key=lambda s: m.objective.state(s))
        zones = relabel([int(v) for v in z.state(best)])
    finally:
        if mine:
            m.unlock()
    timing = res.info.get("timing", {})
    return {"zones": zones, "cost": split_cost(zones, links), "compute_ms": round(wall * 1000),
            "solver_seconds": timing.get("charge_time", 0) / 1e6, "qpu_seconds": timing.get("qpu_access_time", 0) / 1e6,
            "solver": sampler.solver.name, "engine": "dwave"}


def set_job(job_id, **fields):
    with LOCK:
        JOBS[job_id].update(fields)


def get_job(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def work(job_id, payload):
    try:
        names, links = solver.parse_network(payload)
        K = int(payload.get("zones", 3))
        set_job(job_id, status="running")
        set_job(job_id, status="done", result=solve(names, links, K, time_limit=payload.get("time_limit")))
    except Exception as err:
        set_job(job_id, status="error", error=f"The D-Wave run failed: {err}")


def start(payload):
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"status": "preparing", "created": time.time(), "backend": "D-Wave Stride"}
    threading.Thread(target=work, args=(job_id, payload), daemon=True).start()
    return job_id
