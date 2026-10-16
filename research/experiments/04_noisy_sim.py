import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import circuits as cx, config as cfg, metrics as mt, mitigation as mg, noise, qaoa  # noqa: E402

source = json.loads((ROOT / "results" / "03_qaoa_repeats.json").read_text())
N = qaoa.N_QUBITS
MODES = ("shot_floor", "raw", "mitigated")


def score(probs, ideal):
    avg = mt.average_cost(probs, qaoa.COSTS)
    return {"hellinger": mt.hellinger_fidelity(probs, ideal), "p_best": mt.prob_best(probs, qaoa.COSTS),
            "avg_cost": avg, "approx_ratio": mt.approx_ratio(avg, qaoa.COSTS)}


results = []
for name in cfg.BACKENDS:
    backend = noise.get_backend(name)
    sim = noise.simulator(backend)
    for p in cfg.P_VALUES:
        transpiled, chosen, options = cx.pick_transpiled(cx.build_ansatz(p), backend, cfg.TRANSPILE_LEVELS, cfg.TRANSPILE_SEEDS)
        physical = noise.measured_qubits(transpiled)
        runs = []
        for run in source["runs"][str(p)]:
            seed = run["seed"]
            ideal = qaoa.ideal_probs(p, run["gammas"], run["betas"])
            empirical = np.bincount(np.random.default_rng(seed).choice(2 ** N, size=cfg.SHOTS, p=ideal), minlength=2 ** N) / cfg.SHOTS
            raw = noise.noisy_probs(sim, transpiled, run["gammas"], run["betas"], cfg.SHOTS, seed, N)
            matrices = mg.calibration_matrices(sim, backend.num_qubits, physical, cfg.SHOTS, seed)
            mitigated = mg.mitigate(raw, matrices)
            runs.append({"seed": seed, "ideal": score(ideal, ideal), "shot_floor": score(empirical, ideal),
                         "raw": score(raw, ideal), "mitigated": score(mitigated, ideal)})
        summary = {mode: {key: mt.summarize([r[mode][key] for r in runs]) for key in ("hellinger", "p_best", "avg_cost", "approx_ratio")}
                   for mode in MODES}
        by_level = {str(level): {"two_qubit_gates": mt.summarize([o["two_qubit_gates"] for o in options if o["level"] == level]),
                                 "depth": mt.summarize([o["depth"] for o in options if o["level"] == level])} for level in cfg.TRANSPILE_LEVELS}
        results.append({"backend": name, "p": p, "chosen_transpile": chosen, "transpile_by_level": by_level, "transpile_options": options,
                        "physical_qubits": physical, "summary": summary, "runs": runs})
        s = summary
        print(f"{name:<11}p={p}  chosen level {chosen['level']} seed {chosen['seed']}: {chosen['two_qubit_gates']} two-qubit gates, depth {chosen['depth']}"
              f" | F_H floor {s['shot_floor']['hellinger']['mean']:.3f} raw {s['raw']['hellinger']['mean']:.3f} mitigated {s['mitigated']['hellinger']['mean']:.3f}"
              f" | P(best) ideal {np.mean([r['ideal']['p_best'] for r in runs]):.3f} raw {s['raw']['p_best']['mean']:.3f} mitigated {s['mitigated']['p_best']['mean']:.3f}", flush=True)

out = ROOT / "results" / "04_noisy_sim.json"
out.write_text(json.dumps({
    "experiment": "04_noisy_sim", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "note": "Fake backends are stored snapshots of IBM devices, not today's calibration. Parameters come from results/03_qaoa_repeats.json.",
    "settings": {"shots": cfg.SHOTS, "backends": cfg.BACKENDS, "transpile_levels": cfg.TRANSPILE_LEVELS, "transpile_seeds": cfg.TRANSPILE_SEEDS,
                 "simulator_method": "density_matrix", "mitigation": "tensored readout-error inversion with projection to a valid distribution"},
    "machine": {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
                "qiskit_aer": qiskit_aer.__version__, "numpy": np.__version__},
    "results": results}, indent=2))
print(f"\nwrote {out.relative_to(ROOT)}")
