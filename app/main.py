from pathlib import Path

import qiskit
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import solver

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Q-Segment", docs_url=None, redoc_url=None)


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "qiskit": qiskit.__version__, "backends": ["simulator"]}


@app.post("/api/run")
def run(payload: dict):
    if payload.get("backend", "simulator") != "simulator":
        raise HTTPException(status_code=403, detail="Real IBM runs are not enabled on this server.")
    try:
        return solver.run(payload)
    except solver.NetworkError as err:
        raise HTTPException(status_code=400, detail=str(err))
