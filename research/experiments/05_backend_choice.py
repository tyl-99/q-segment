import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qsegment import config as cfg, hardware as hw, metrics as mt, noise, qaoa  # noqa: E402

NAMES = ("ibm_fez", "ibm_kingston", "ibm_marrakesh")
PS = (1, 2)
source = json.loads((ROOT / "results" / "03_qaoa_repeats.json").read_text())
service = hw.get_service()

report = {}
for name in NAMES:
    backend = service.backend(name)
    sim = hw.noise_simulator(backend)
    report[name] = {"calibration": hw.calibration_summary(backend), "p": {}}
    for p in PS:
        transpiled, chosen, _ = hw.transpile_for(backend, p)
        runs = []
        for run in source["runs"][str(p)]:
            ideal = qaoa.ideal_probs(p, run["gammas"], run["betas"])
            raw = noise.noisy_probs(sim, transpiled, run["gammas"], run["betas"], cfg.SHOTS, run["seed"], qaoa.N_QUBITS)
            runs.append({"seed": run["seed"], "ideal_p_best": mt.prob_best(ideal, qaoa.COSTS)} | mt.score(raw, ideal, qaoa.COSTS))
        hell = [r["hellinger"] for r in runs]
        report[name]["p"][p] = {
            "chosen_transpile": chosen, "physical_qubits": noise.measured_qubits(transpiled),
            "hellinger": mt.summarize(hell), "p_best": mt.summarize([r["p_best"] for r in runs]),
            "retention": float(np.mean([r["p_best"] for r in runs]) / np.mean([r["ideal_p_best"] for r in runs])),
            "hellinger_by_seed": hell}
        s = report[name]["p"][p]
        print(f"{name:<14}p={p} {chosen['two_qubit_gates']:>3} two-qubit gates, depth {chosen['depth']:>3} | F_H {s['hellinger']['mean']:.4f} "
              f"(se {s['hellinger']['std'] / np.sqrt(len(hell)):.4f}) | P(best) {s['p_best']['mean']:.4f} | share of ideal {s['retention']:.0%}", flush=True)

order = sorted(NAMES, key=lambda n: -report[n]["p"][2]["hellinger"]["mean"])
best, second = order[0], order[1]
diff = np.array(report[best]["p"][2]["hellinger_by_seed"]) - np.array(report[second]["p"][2]["hellinger_by_seed"])
se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
clear = bool(diff.mean() > 2 * se)
chosen = best if clear else min((best, second), key=lambda n: report[n]["calibration"]["cz_error_median"])
print(f"\nranking by F_H at p=2: {order}")
print(f"{best} minus {second}: paired difference {diff.mean():+.4f}, standard error {se:.4f} -> {'clear win' if clear else 'tie, broken by lower median CZ error'}")
print(f"chosen backend: {chosen}")

out = ROOT / "results" / "05_backend_choice.json"
out.write_text(json.dumps({
    "experiment": "05_backend_choice", "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "rule": "highest mean Hellinger fidelity at p=2 under each chip's current noise model; a lead below 2 paired standard errors counts as a tie, broken by lower median CZ error",
    "ranking_p2": order, "top_two_paired_difference": {"mean": float(diff.mean()), "standard_error": se, "clear": clear},
    "chosen": chosen, "backends": report}, indent=2))
print(f"wrote {out.relative_to(ROOT)}")
