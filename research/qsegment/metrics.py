import math
import time

import numpy as np


def prob_best(probs, costs):
    """Total probability of every partition that has the lowest cost."""
    costs = np.asarray(costs)
    return float(np.asarray(probs)[costs == costs.min()].sum())


def average_cost(probs, costs):
    return float(np.dot(probs, costs))


def approx_ratio(avg_cost, costs):
    """(worst - average) / (worst - best): 1 is the optimum, 0 is the worst partition."""
    costs = np.asarray(costs)
    return float((costs.max() - avg_cost) / (costs.max() - costs.min()))


def shots_to_hit(p):
    return 1.0 / p if p > 0 else math.inf


def hellinger_fidelity(p, q):
    """(sum_b sqrt(p_b q_b))^2 for two probability arrays in the same partition-index order."""
    return float(np.sum(np.sqrt(np.asarray(p) * np.asarray(q))) ** 2)


def counts_to_probs(counts, n_qubits):
    """Sampler counts {bitstring: count} to an array indexed by partition index (qubit k = bit k)."""
    probs = np.zeros(2 ** n_qubits)
    for bits, count in counts.items():
        probs[int(bits.replace(" ", ""), 2)] += count
    return probs / probs.sum()


def score(probs, ideal, costs):
    """Hellinger fidelity to the ideal distribution plus the solution-quality numbers of one distribution."""
    avg = average_cost(probs, costs)
    return {"hellinger": hellinger_fidelity(probs, ideal), "p_best": prob_best(probs, costs),
            "avg_cost": avg, "approx_ratio": approx_ratio(avg, costs)}


def counts_to_partition_probs(counts, K=3, n_systems=7):
    """Counts of K-zone one-hot bitstrings to (probabilities over valid partitions, share of invalid shots).
    Free system k uses bits K*k .. K*k+K-1; it is valid when exactly one of them is 1. The probabilities add up to 1 - invalid."""
    probs = np.zeros(K ** n_systems)
    total = invalid = 0
    for bits, count in counts.items():
        bits = bits.replace(" ", "")
        total += count
        index = 0
        for k in range(n_systems):
            hot = [z for z in range(K) if bits[len(bits) - 1 - (K * k + z)] == "1"]
            if len(hot) != 1:
                invalid += count
                break
            index += hot[0] * K ** k
        else:
            probs[index] += count
    return probs / total, invalid / total


def best_within(trace, budgets):
    """Best cost seen after the first N evaluations, for each N. `trace` is the best-so-far after every evaluation."""
    return {n: trace[min(n, len(trace)) - 1] for n in budgets}


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(values.size), "mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


def timed(fn, *args, repeat=1, **kwargs):
    """Run fn `repeat` times and return (last result, seconds per call)."""
    start = time.perf_counter()
    for _ in range(repeat):
        result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) / repeat
