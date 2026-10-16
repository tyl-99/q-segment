import json

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qsegment import circuits as cx, config as cfg, metrics as mt, mitigation as mg, noise, qaoa

N = qaoa.N_QUBITS


def get_service():
    return QiskitRuntimeService(name="qsegment")


def calibration_summary(backend):
    target = backend.target
    errors = lambda op: [p.error for p in target[op].values() if p is not None and p.error is not None]
    props = [q for q in target.qubit_properties if q is not None]
    return {"backend": backend.name, "calibration_time": backend.properties().last_update_date.isoformat(),
            "pending_jobs": backend.status().pending_jobs, "cz_error_median": float(np.median(errors("cz"))),
            "readout_error_median": float(np.median(errors("measure"))),
            "t1_us_median": float(np.median([q.t1 for q in props if q.t1]) * 1e6),
            "t2_us_median": float(np.median([q.t2 for q in props if q.t2]) * 1e6)}


def noise_simulator(backend):
    """Noise model built from the backend's current calibration."""
    return AerSimulator.from_backend(backend, method="density_matrix")


def transpile_for(backend, p):
    return cx.pick_transpiled(cx.build_ansatz(p), backend, cfg.TRANSPILE_LEVELS, cfg.TRANSPILE_SEEDS)


def pick_parameter_sets(runs, k):
    """The k runs with the lowest average cost on their own fresh sample, so the choice never uses exact values."""
    return sorted(runs, key=lambda r: r["final_sample"]["avg_cost"])[:k]


def submit(backend, transpiled, parameter_sets, shots, options=None):
    """One job with a pub per parameter set plus the two readout calibration circuits. Returns (job, physical qubits)."""
    physical = noise.measured_qubits(transpiled)
    pubs = [(transpiled, cx.bind_values(transpiled, r["gammas"], r["betas"])) for r in parameter_sets]
    pubs += [mg.calibration_circuit(backend.num_qubits, physical, prepared) for prepared in (0, 1)]
    return SamplerV2(mode=backend, options=options).run(pubs, shots=shots), physical


def read(job, n_sets):
    """Wait for the job. Returns raw counts, probabilities per parameter set and the readout confusion matrices."""
    counts = [pub.data.meas.get_counts() for pub in job.result()]
    return {"counts": counts, "probs": [mt.counts_to_probs(c, N) for c in counts[:n_sets]],
            "matrices": mg.matrices_from_counts({0: counts[n_sets], 1: counts[n_sets + 1]}, N)}


def job_record(job):
    return {"job_id": job.job_id(), "usage_seconds": job.usage(), "metrics": json.loads(json.dumps(job.metrics(), default=str))}
