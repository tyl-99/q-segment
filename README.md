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
- `POST /api/run` takes `{zones, backend, nodes, links}` and returns the zones, the cost, the best possible cost, and the circuit size. Only the `simulator` backend is enabled; real IBM runs are switched off on this server.

## Layout

- `app/main.py`: FastAPI server (page + API).
- `app/solver.py`: QUBO, Ising (Qiskit `SparsePauliOp`), QAOA circuit, simulation and local search.
- `app/static/index.html`: the web app.
