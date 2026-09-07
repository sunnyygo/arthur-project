"""API FastAPI — Confidence Trace untuk Agent Riset Otonom.

v1.2 (UI redesign):
- Hasil riset TIDAK disimpan di server. Frontend menyimpan riwayat di
  localStorage browser, sehingga memori VPS tidak termakan riset pengguna.
- Riwayat "Riset Terbaru" ditampilkan sebagai sidebar terpisah (scrollable),
  tidak bercampur dengan area riset pengguna.
- Light mode.
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from app.engine import ResearchEngine, resolve_backend

app = FastAPI(title="UQ Agent — Confidence Trace", version="1.2.0")

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent


class ScenarioIn(BaseModel):
    scenario: str

    @field_validator("scenario")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("skenario tidak boleh kosong")
        return v.strip()


@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/api/run")
def run_scenario(payload: ScenarioIn):
    """Jalankan skenario & kembalikan trace. Tidak ada penyimpanan di server."""
    engine = ResearchEngine(backend=resolve_backend())
    return engine.run_scenario(payload.scenario)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "app" / "static" / "index.html")
