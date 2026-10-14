from pathlib import Path

import qiskit
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app import ibm, solver

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Q-Segment", docs_url=None, redoc_url=None)


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    backends = ["simulator"] + (["ibm"] if ibm.available() else [])
    return {"ok": True, "qiskit": qiskit.__version__, "backends": backends}


@app.post("/api/run")
def run(payload: dict):
    backend = payload.get("backend", "simulator")
    if backend == "simulator":
        try:
            return solver.run(payload)
        except solver.NetworkError as err:
            raise HTTPException(status_code=400, detail=str(err))
    if backend != "ibm":
        raise HTTPException(status_code=400, detail="Unknown backend.")
    if not ibm.available():
        raise HTTPException(status_code=503, detail="Real IBM runs are not set up on this server.")
    if not ibm.code_ok(payload.get("access_code")):
        raise HTTPException(status_code=401, detail="That access code is not valid.")
    try:
        solver.parse_network(payload)
    except solver.NetworkError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if not ibm.take_slot():
        raise HTTPException(status_code=429, detail="The chip run limit for this hour is reached. Try again later.")
    return JSONResponse(status_code=202, content={"job_id": ibm.start(payload), "status": "preparing"})


@app.get("/api/job/{job_id}")
def job(job_id: str):
    j = ibm.get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="No such job.")
    return {k: v for k, v in j.items() if k != "created"}
