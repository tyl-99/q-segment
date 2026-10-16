# Q-Segment

Network zoning that balances security and business. Q-Segment splits a network into zones so that one hacked machine cannot reach everything, while the traffic the business needs keeps flowing.

- **Live demo:** https://q-segment.up.railway.app (draw a network, press **Segment my network**, then try **Simulate breach** in the zoned view)
- **Built for:** Qiskit Fall Fest hackathon, FAU, 17-18 Oct 2026

## The problem in one paragraph

Every link between two systems has a **traffic** value (how much the business needs them to talk) and a **risk** value (how much damage spreads if an attacker on one reaches the other). A split into zones costs the traffic on links that cross zones (each needs a firewall rule) plus the risk on links kept inside a zone. With `w = risk - traffic` per link, finding the cheapest split is a **signed Max-k-Cut** problem. We solve it for **3 zones** with QAOA in Qiskit.

## What we found (3 zones, 8 systems, 21 qubits)

| | Result | Source |
|---|---|---|
| Possible splits | 2,187 (3^7, first system fixed in zone A); best cost 25 | `research/results/08_verify_3zone.json`, `09_baselines_3zone.json` |
| Circuit size | 21 qubits, 148 two-qubit gates before routing, 436 after routing to ibm_kingston (p = 1) | `08_verify_3zone.json`, `12_hardware_3zone.json` |
| Valid answers on ibm_kingston | 12.2% of readouts (seeds 2026-2028: 11.3%, 12.9%, 12.4%) | `12_hardware_3zone.json` |
| Chance a valid readout is the best split | 0.94% on the chip vs 1.87% in ideal simulation (mean of 3 seeds) | `12_hardware_3zone.json` |
| Average cost of a valid readout | 37.1 on the chip vs 34.6 in ideal simulation | `12_hardware_3zone.json` |
| After post-processing (local search from each valid chip readout) | average cost drops to 26.3; 30-39% of starting readouts reach the best split (cost 25) | `12_hardware_3zone.json` |
| Readout correction (our own, from 2 calibration circuits in the same job) | valid answers 12.7% → 13.4% | `17_hardware_3zone_suppression.json` |
| IBM dynamical decoupling + twirling | valid answers 12.7% → 1.4% (made it worse) | `17_hardware_3zone_suppression.json` |
| Classical solvers at this size | find the best split in about a millisecond | `09_baselines_3zone.json` |
| Classical solvers at 60-150 systems | Gurobi (60 s limit, 1 thread) proved the best answer in 0 of 12 runs; gaps 40-117% | `16_classical_wall_3zone.json` |

**Honest conclusion:** at 8 systems a normal computer is far faster, and the chip only finds the best split with classical post-processing. We make no speedup claim. Whether quantum hardware helps at the sizes where exact classical solvers stall (60+ systems) is an open question; at 100 systems the 3-zone encoding needs 297 qubits (today's ibm_kingston has 156), with much lower noise than today.

**IBM job IDs** (all on `ibm_kingston`, look them up at quantum.cloud.ibm.com → Workloads): `db6e7do4qg6s73c4kuv0`, `db7cqjiraffc73fa926g`, `db7cqjmil99s73fnr8sg`.

## What we built vs what we reused

**Reused:** Qiskit (`QAOAAnsatz`, `StatePreparation`, `XXPlusYYGate`, `SparsePauliOp`, `transpile`), Qiskit Aer and IBM Runtime `SamplerV2`, SciPy's COBYLA optimizer, Gurobi for the classical comparison, and the standard QAOA and Max-Cut ideas (Farhi et al. 2014).

**Built by us:**
- The formulation of network microsegmentation as signed Max-3-Cut with traffic and risk per link, and its QUBO and Ising forms (`research/qsegment/formulation.py`).
- A 3-zone QAOA circuit with one-hot encoding: each system starts in a W state over its 3 zone qubits, and a ring mixer of `XXPlusYY` gates keeps every system in exactly one zone (`research/qsegment/circuits.py`).
- An exact simulator restricted to valid splits, used to tune angles (`research/qsegment/subspace.py`).
- Post-processing: steepest-descent local search that starts from each chip readout (`research/qsegment/postprocess.py`).
- Readout correction from two calibration circuits in the same IBM job (`research/qsegment/mitigation.py`).
- Classical baselines (exhaustive search, greedy, simulated annealing) and the experiments and figures below.
- The web app and its FastAPI + Qiskit backend (`app/`).

## Repository layout

```
app/                  web app (FastAPI server, Qiskit solver, IBM runs, page)
research/
  qsegment/           core code: formulation, circuits, QAOA, mitigation, baselines
  experiments/        numbered scripts; 08-17 are the 3-zone results, 01-07 the earlier 2-zone version
  results/            JSON output of every experiment (including IBM job IDs and raw counts)
  notebooks/          01_circuit (the circuit), 02_all_results (tables), 03_hackathon_figures (figures)
```

## Reproduce the research results

```bash
cd research
pip install -r requirements.txt
python experiments/08_verify_3zone.py      # checks QUBO, Ising and circuit against brute force
python experiments/09_baselines_3zone.py   # classical baselines (needs gurobipy)
python experiments/10_qaoa_3zone.py        # QAOA in simulation, 3 seeds
jupyter notebook notebooks/                # open 02_all_results.ipynb for every table
```

- Seeds are fixed in `research/qsegment/config.py` (base seed 2026).
- Gurobi is optional and only used by experiments 02b, 09 and 13-16; it needs a Gurobi licence.
- Hardware scripts (06, 07, 11, 12, 17) spend real IBM time. They need an IBM Quantum account saved with `notebooks/00_ibm_account.ipynb` (the key is typed into a hidden prompt and is never stored in the repo). Their saved results are already in `research/results/`.

## Run the web app locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000. The app needs only the root `requirements.txt`, not the research one.

### App API

- `GET /api/health` returns the server status, the Qiskit version and the available backends.
- `POST /api/run` takes `{zones, backend, nodes, links}`. With `backend: "simulator"` it returns the zones, the cost, the best possible cost and the circuit size. With `backend: "ibm"`, a `chip` and a valid `access_code` it starts a real IBM job and returns a `job_id` (HTTP 202).
- `GET /api/job/{job_id}` reports `preparing`, `queued`, `running`, `reading`, then `done` with the result or `error`.

### Real IBM runs from the app

Switched on only when these server variables are set (never commit them):

- `QSEGMENT_ACCESS_CODE`: the code people type on the page.
- `IBM_QUANTUM_TOKEN`: IBM Quantum API key. `IBM_QUANTUM_INSTANCE` is optional.
- Optional limits: `IBM_BACKEND` (default `ibm_kingston`), `IBM_BACKENDS` (chips offered), `IBM_MAX_SECONDS` (30 per job), `IBM_MAX_RUNS_PER_HOUR` (3), `IBM_MIN_REMAINING_SECONDS` (60).

## Limits we know about

- Risk is a hand-set 0-10 number per link in this demo. A real deployment would compute it from vulnerability scores and asset value.
- The cost counts risk only on linked pairs inside a zone; the breach animation treats every system in a zone as reachable.
- 8 systems is a demo size. Real networks have thousands of systems.
