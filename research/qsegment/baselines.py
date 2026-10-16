import math
import time

from qsegment import scenario as sc


class BudgetExhausted(Exception):
    pass


class Oracle:
    """Computes the cost of one partition per call and records the best so far after every call."""

    def __init__(self, K, budget, target):
        self.K, self.budget, self.target = K, budget, target
        self.evals = 0
        self.best = math.inf
        self.best_index = None
        self.trace = []
        self.first_hit_eval = None
        self.first_hit_seconds = None
        self.start = time.perf_counter()

    def __call__(self, index):
        if self.evals >= self.budget:
            raise BudgetExhausted
        self.evals += 1
        value = sc.cost(sc.decode(index, self.K))
        if value < self.best:
            self.best, self.best_index = value, index
        self.trace.append(self.best)
        if self.first_hit_eval is None and value == self.target:
            self.first_hit_eval = self.evals
            self.first_hit_seconds = time.perf_counter() - self.start
        return value

    def result(self):
        return {"best_cost": self.best, "best_index": self.best_index, "evals": self.evals,
                "hit": self.first_hit_eval is not None, "first_hit_eval": self.first_hit_eval,
                "first_hit_seconds": self.first_hit_seconds, "seconds": time.perf_counter() - self.start,
                "trace": self.trace}


def neighbors(index, K):
    """Partitions that differ from `index` in the zone of exactly one free system."""
    for k in range(len(sc.FREE)):
        zone = (index // K ** k) % K
        for z in range(K):
            if z != zone:
                yield index + (z - zone) * K ** k


def brute_force(K):
    start = time.perf_counter()
    costs = [sc.cost(sc.decode(index, K)) for index in range(K ** len(sc.FREE))]
    seconds = time.perf_counter() - start
    best = min(costs)
    return {"best_cost": best, "optimal_indices": [i for i, c in enumerate(costs) if c == best],
            "worst_cost": max(costs), "mean_cost": sum(costs) / len(costs), "evals": len(costs), "seconds": seconds}


def random_search(K, budget, target, rng):
    oracle = Oracle(K, budget, target)
    for index in rng.integers(K ** len(sc.FREE), size=budget):
        oracle(int(index))
    return oracle.result()


def greedy(K, budget, target, rng):
    """Steepest descent over all one-system moves, restarted from a random partition until the budget is used."""
    oracle = Oracle(K, budget, target)
    size = K ** len(sc.FREE)
    try:
        while True:
            current = int(rng.integers(size))
            value = oracle(current)
            while True:
                best_neighbor, best_value = None, value
                for candidate in neighbors(current, K):
                    v = oracle(candidate)
                    if v < best_value:
                        best_neighbor, best_value = candidate, v
                if best_neighbor is None:
                    break
                current, value = best_neighbor, best_value
    except BudgetExhausted:
        pass
    return oracle.result()


def simulated_annealing(K, budget, target, rng, steps=200, t_start=10.0, t_end=0.2):
    """Single-system moves, geometric cooling from t_start to t_end over `steps`, restarted until the budget is used."""
    oracle = Oracle(K, budget, target)
    size = K ** len(sc.FREE)
    cooling = (t_end / t_start) ** (1 / (steps - 1))
    try:
        while True:
            current = int(rng.integers(size))
            value = oracle(current)
            temperature = t_start
            for _ in range(steps):
                k = int(rng.integers(len(sc.FREE)))
                zone = (current // K ** k) % K
                z = (zone + 1 + int(rng.integers(K - 1))) % K
                candidate = current + (z - zone) * K ** k
                v = oracle(candidate)
                if v <= value or rng.random() < math.exp((value - v) / temperature):
                    current, value = candidate, v
                temperature *= cooling
    except BudgetExhausted:
        pass
    return oracle.result()
