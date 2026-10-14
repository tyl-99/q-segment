import hmac
import os
import threading
import time
import uuid

import numpy as np

from app import solver

JOBS, LOCK, SUBMITTED = {}, threading.Lock(), []
SHOTS = 4096


def settings():
    return {
        "token": os.environ.get("IBM_QUANTUM_TOKEN", ""), "instance": os.environ.get("IBM_QUANTUM_INSTANCE") or None,
        "backend": os.environ.get("IBM_BACKEND", "ibm_kingston"), "code": os.environ.get("QSEGMENT_ACCESS_CODE", ""),
        "max_seconds": int(os.environ.get("IBM_MAX_SECONDS", "30")), "per_hour": int(os.environ.get("IBM_MAX_RUNS_PER_HOUR", "3")),
        "min_remaining": int(os.environ.get("IBM_MIN_REMAINING_SECONDS", "60")), "dry_run": os.environ.get("IBM_DRY_RUN") == "1",
    }


def available():
    s = settings()
    return bool(s["code"]) and (bool(s["token"]) or s["dry_run"])


def code_ok(code):
    expected = settings()["code"]
    return bool(expected) and hmac.compare_digest(str(code or ""), expected)


def take_slot():
    """At most IBM_MAX_RUNS_PER_HOUR chip runs per rolling hour, so a leaked code cannot drain the allowance."""
    now, limit = time.time(), settings()["per_hour"]
    with LOCK:
        SUBMITTED[:] = [t for t in SUBMITTED if now - t < 3600]
        if len(SUBMITTED) >= limit:
            return False
        SUBMITTED.append(now)
        return True


def set_job(job_id, **fields):
    with LOCK:
        JOBS[job_id].update(fields)


def get_job(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def start(payload):
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"status": "preparing", "created": time.time()}
    threading.Thread(target=_work, args=(job_id, payload), daemon=True).start()
    return job_id


def _fake_counts(prep):
    """Local testing only (IBM_DRY_RUN=1): readouts drawn from the ideal circuit with random bit flips."""
    rng = np.random.default_rng(7)
    probs = solver.evolve(prep["norm"], prep["n"], prep["K"], prep["gamma"], prep["beta"])
    n, K, nv = prep["n"], prep["K"], prep["op"].num_qubits
    counts = {}
    for idx in rng.choice(len(probs), size=SHOTS, p=probs / probs.sum()):
        bits = ["0"] * nv
        for i in range(1, n):
            z = prep["zones"][idx][i]
            if K == 2:
                bits[solver.variable(i, 0, K)] = "1" if z == 1 else "0"
            else:
                bits[solver.variable(i, z, K)] = "1"
        bits = ["1" if (b == "0") == (rng.random() < 0.03) else "0" for b in bits]
        key = "".join(reversed(bits))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _work(job_id, payload):
    s = settings()
    try:
        prep = solver.prepare(payload)
        if s["dry_run"]:
            set_job(job_id, status="running", backend="dry-run")
            time.sleep(2)
            counts, ibm_id, chip_seconds, two_q = _fake_counts(prep), "dry-run", 0, None
        else:
            from qiskit import transpile
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=s["token"], instance=s["instance"])
            remaining = service.usage().get("usage_remaining_seconds")
            if remaining is not None and remaining < s["min_remaining"]:
                raise RuntimeError("The IBM time allowance is almost used up, so chip runs are paused.")
            backend = service.backend(s["backend"])
            isa = transpile(prep["bound"], backend, optimization_level=3, seed_transpiler=1)
            two_q = int(sum(v for k, v in isa.count_ops().items() if k in ("cz", "ecr", "cx")))
            job = SamplerV2(mode=backend, options={"max_execution_time": s["max_seconds"]}).run([isa], shots=SHOTS)
            ibm_id = job.job_id()
            set_job(job_id, status="queued", backend=s["backend"], ibm_job_id=ibm_id)
            while True:
                st = str(job.status()).upper()
                if "DONE" in st:
                    break
                if "ERROR" in st or "CANCEL" in st:
                    raise RuntimeError(f"IBM reported the job as {st.lower()}.")
                set_job(job_id, status="running" if "RUNNING" in st else "queued")
                time.sleep(3)
            counts = job.result()[0].data.meas.get_counts()
            usage = job.usage()
            chip_seconds = float(usage) if usage is not None else None
        set_job(job_id, status="reading")
        indices, valid_share = solver.counts_to_indices(counts, prep)
        if not indices:
            raise RuntimeError("None of the chip's readouts was a valid split. Try again.")
        best_idx = solver.best_after_cleanup(prep, list(indices))
        result = solver.summary(prep, best_idx, {"engine": "ibm-quantum", "backend": s["backend"] if not s["dry_run"] else "dry-run",
                                                 "ibm_job_id": ibm_id, "chip_seconds": chip_seconds, "valid_share": round(valid_share, 4),
                                                 "two_qubit_gates_routed": two_q, "shots": SHOTS})
        set_job(job_id, status="done", result=result)
    except solver.NetworkError as err:
        set_job(job_id, status="error", error=str(err))
    except Exception as err:
        set_job(job_id, status="error", error=f"The chip run failed: {err}")
