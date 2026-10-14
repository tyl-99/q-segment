# Q-Segment

Network zoning that balances security and business. Draw a network, set traffic and risk on each link, and press **Run quantum**: the server turns the network into a QUBO, converts it to an Ising Hamiltonian with Qiskit, builds the QAOA circuit, simulates it, and returns the zones.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000.

## API

- `GET /api/health` returns the server status and Qiskit version.
- `POST /api/run` takes `{zones, backend, nodes, links}`. With `backend: "simulator"` it returns the zones, the cost, the best possible cost and the circuit size. With `backend: "ibm"` and a valid `access_code` it starts a real IBM job and returns a `job_id` (HTTP 202).
- `GET /api/job/{job_id}` reports `preparing`, `queued`, `running`, `reading`, then `done` with the result or `error`.

## Real IBM runs

Switched on only when these Railway variables are set (never commit them):

- `QSEGMENT_ACCESS_CODE`: the code people type on the page.
- `IBM_QUANTUM_TOKEN`: IBM Quantum API key. `IBM_QUANTUM_INSTANCE` is optional.
- Optional limits: `IBM_BACKEND` (default `ibm_kingston`), `IBM_MAX_SECONDS` (30 per job), `IBM_MAX_RUNS_PER_HOUR` (3), `IBM_MIN_REMAINING_SECONDS` (60).

## Layout

- `app/main.py`: FastAPI server (page + API).
- `app/solver.py`: QUBO, Ising (Qiskit `SparsePauliOp`), QAOA circuit, simulation and local search.
- `app/ibm.py`: real IBM jobs (access code, hourly limit, allowance check, readout processing).
- `app/static/index.html`: the web app.
