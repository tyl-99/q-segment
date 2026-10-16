import time

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer.primitives import SamplerV2
from scipy.optimize import minimize

from qsegment import circuits as cx, config as cfg, metrics as mt, scenario as sc, subspace

COSTS = np.array(sc.cost_table(2), dtype=float)
COSTS3 = subspace.COSTS
N_QUBITS = len(sc.FREE)


def ideal_probs(p, gammas, betas):
    """Exact output distribution of the noiseless circuit."""
    circuit = cx.build_ansatz(p, measure=False)
    values = cx.bind_values(circuit, gammas, betas)
    return Statevector(circuit.assign_parameters(dict(zip(circuit.parameters, values)))).probabilities()


def sampled_probs(sampler, circuit, values, shots):
    counts = sampler.run([(circuit, values)], shots=shots).result()[0].data.meas.get_counts()
    return mt.counts_to_probs(counts, N_QUBITS)


def run_qaoa(p, seed, shots_opt=cfg.SHOTS_OPT, shots_final=cfg.SHOTS, maxiter=cfg.MAXITER, rhobeg=cfg.RHOBEG, tol=cfg.TOL,
             restarts=cfg.RESTARTS):
    """Tune gamma and beta with COBYLA from `restarts` random starts, using the average cost of shots_opt shots.
    The winner is the restart with the lowest average cost on a fresh sample of shots_final shots, so the choice
    never uses the exact probabilities. The winner is then sampled once more for reporting."""
    rng = np.random.default_rng(seed)
    circuit = cx.build_ansatz(p)
    sampler = SamplerV2(seed=seed)
    candidates = []
    start = time.perf_counter()
    for _ in range(restarts):
        x0 = np.r_[rng.uniform(-0.6, 0.6, p), rng.uniform(-0.8, 0.8, p)]

        def objective(x):
            probs = sampled_probs(sampler, circuit, cx.bind_values(circuit, x[:p], x[p:]), shots_opt)
            return float(probs @ COSTS)

        fit = minimize(objective, x0, method="COBYLA", tol=tol, options={"maxiter": maxiter, "rhobeg": rhobeg})
        gammas, betas = fit.x[:p], fit.x[p:]
        select = sampled_probs(sampler, circuit, cx.bind_values(circuit, gammas, betas), shots_final)
        candidates.append({"x0": x0.tolist(), "gammas": gammas.tolist(), "betas": betas.tolist(), "iterations": int(fit.nfev),
                           "hit_maxiter": bool(fit.nfev >= maxiter), "select_cost": float(select @ COSTS),
                           "ideal_p_best": mt.prob_best(ideal_probs(p, gammas, betas), COSTS)})
    tuning_seconds = time.perf_counter() - start

    winner = min(candidates, key=lambda c: c["select_cost"])
    gammas, betas = winner["gammas"], winner["betas"]
    ideal = ideal_probs(p, gammas, betas)
    final = sampled_probs(sampler, circuit, cx.bind_values(circuit, gammas, betas), shots_final)
    p_best = mt.prob_best(ideal, COSTS)
    avg = mt.average_cost(ideal, COSTS)
    evaluations = sum(c["iterations"] for c in candidates)
    tuning_shots = evaluations * shots_opt + restarts * shots_final
    return {
        "p": p, "seed": seed, "restarts": restarts, "gammas": gammas, "betas": betas, "candidates": candidates,
        "iterations": evaluations, "hit_maxiter": any(c["hit_maxiter"] for c in candidates), "tuning_shots": tuning_shots,
        "tuning_seconds": tuning_seconds,
        "ideal": {"p_best": p_best, "avg_cost": avg, "approx_ratio": mt.approx_ratio(avg, COSTS),
                  "shots_to_hit": mt.shots_to_hit(p_best), "total_shots_to_hit": tuning_shots + mt.shots_to_hit(p_best)},
        "final_sample": {"shots": shots_final, "p_best": mt.prob_best(final, COSTS), "hit": bool(mt.prob_best(final, COSTS) > 0),
                         "avg_cost": mt.average_cost(final, COSTS), "hellinger_vs_ideal": mt.hellinger_fidelity(ideal, final)},
    }


def run_qaoa_3zone(p, seed, shots_opt=cfg.SHOTS_OPT, shots_final=cfg.SHOTS, maxiter=cfg.MAXITER, rhobeg=cfg.RHOBEG, tol=cfg.TOL,
                   restarts=cfg.RESTARTS):
    """Same tuning protocol as run_qaoa, for 3 zones. The shots are drawn from the exact distribution inside the
    valid partitions (subspace.py, checked against the Qiskit circuit in 08_verify_3zone.py), which is much faster
    than simulating 21 qubits for every evaluation."""
    rng = np.random.default_rng(seed)
    sample = lambda probs, shots: rng.multinomial(shots, probs / probs.sum()) / shots
    candidates = []
    start = time.perf_counter()
    for _ in range(restarts):
        x0 = np.r_[rng.uniform(-0.6, 0.6, p), rng.uniform(-0.8, 0.8, p)]

        def objective(x):
            return float(sample(subspace.probs(x[:p], x[p:]), shots_opt) @ COSTS3)

        fit = minimize(objective, x0, method="COBYLA", tol=tol, options={"maxiter": maxiter, "rhobeg": rhobeg})
        gammas, betas = fit.x[:p], fit.x[p:]
        probs = subspace.probs(gammas, betas)
        candidates.append({"x0": x0.tolist(), "gammas": gammas.tolist(), "betas": betas.tolist(), "iterations": int(fit.nfev),
                           "hit_maxiter": bool(fit.nfev >= maxiter), "select_cost": float(sample(probs, shots_final) @ COSTS3),
                           "ideal_p_best": mt.prob_best(probs, COSTS3)})
    tuning_seconds = time.perf_counter() - start

    winner = min(candidates, key=lambda c: c["select_cost"])
    ideal = subspace.probs(winner["gammas"], winner["betas"])
    counts = rng.multinomial(shots_final, ideal / ideal.sum())
    final = counts / shots_final
    p_best = mt.prob_best(ideal, COSTS3)
    avg = mt.average_cost(ideal, COSTS3)
    evaluations = sum(c["iterations"] for c in candidates)
    tuning_shots = evaluations * shots_opt + restarts * shots_final
    return {
        "p": p, "seed": seed, "restarts": restarts, "gammas": winner["gammas"], "betas": winner["betas"], "candidates": candidates,
        "iterations": evaluations, "hit_maxiter": any(c["hit_maxiter"] for c in candidates), "tuning_shots": tuning_shots,
        "tuning_seconds": tuning_seconds,
        "ideal": {"p_best": p_best, "avg_cost": avg, "approx_ratio": mt.approx_ratio(avg, COSTS3),
                  "shots_to_hit": mt.shots_to_hit(p_best), "total_shots_to_hit": tuning_shots + mt.shots_to_hit(p_best)},
        "final_sample": {"shots": shots_final, "p_best": mt.prob_best(final, COSTS3), "hit": bool(mt.prob_best(final, COSTS3) > 0),
                         "avg_cost": mt.average_cost(final, COSTS3), "hellinger_vs_ideal": mt.hellinger_fidelity(ideal, final)},
        "final_counts": {int(i): int(c) for i, c in enumerate(counts) if c},
    }
