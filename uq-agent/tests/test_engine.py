"""Tests engine: pipeline 4 tahap + self-consistency (mocked LLM, tanpa API asli)."""
import pytest
from app.engine import ResearchEngine, MockBackend, resolve_backend
from app.uq import THRESHOLD_OK, THRESHOLD_WARN


# ---------- Mock backend ----------
def test_mock_backend_deterministic_at_temperature_zero():
    b = MockBackend(temperature=0)
    r1 = b.chat([{"role": "user", "content": "halo"}])
    r2 = b.chat([{"role": "user", "content": "halo"}])
    assert r1 == r2  # temperature 0 = deterministik, seperti LLM asli

def test_mock_backend_varies_at_high_temperature():
    b = MockBackend(temperature=0.9)
    runs = {b.chat([{"role": "user", "content": "tes"}]) for _ in range(20)}
    assert len(runs) > 1  # temperature > 0 wajib menghasilkan variasi, kalau tidak self-consistency tidak bermakna


# ---------- Pipeline ----------
def test_engine_runs_four_stages():
    eng = ResearchEngine(backend=MockBackend())
    trace = eng.run_scenario("Apakah verbalized confidence bisa diandalkan?")
    stages = [s["stage"] for s in trace["steps"]]
    assert stages == ["pencarian", "ekstraksi", "sintesis", "penyimpulan"]

def test_engine_each_step_has_scores_and_alert():
    eng = ResearchEngine(backend=MockBackend())
    trace = eng.run_scenario("tes skenario")
    for s in trace["steps"]:
        assert 0.0 <= s["verbalized_confidence"] <= 1.0
        assert 0.0 <= s["consistency"] <= 1.0
        assert 0.0 <= s["score"] <= 1.0
        assert s["alert"] in ("OK", "PERLU CEK MANUAL", "TIDAK PASTI")
        assert s["n_calls"] == 5  # 4 pipeline (1 asli + 3 pengulangan) + 1 tanya keyakinan (satu pemanggilan tambahan)

def test_engine_records_repetition_outputs():
    eng = ResearchEngine(backend=MockBackend())
    trace = eng.run_scenario("tes")
    step = trace["steps"][0]
    assert len(step["repetitions"]) == 3
    assert step["output"]  # keluaran asli tersimpan

def test_engine_trace_metadata():
    eng = ResearchEngine(backend=MockBackend())
    trace = eng.run_scenario("tes")
    assert trace["scenario"] == "tes"
    assert trace["model"] == "mock"
    assert "started_at" in trace and "finished_at" in trace
    assert trace["run_id"]


# ---------- Resolusi backend ----------
def test_resolve_backend_mock_by_default(monkeypatch):
    monkeypatch.delenv("UQ_LLM_API_KEY", raising=False)
    assert isinstance(resolve_backend(), MockBackend)

def test_resolve_backend_http_when_key_present(monkeypatch):
    monkeypatch.setenv("UQ_LLM_API_KEY", "test-key")
    monkeypatch.setenv("UQ_LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("UQ_LLM_MODEL", "test-model")
    backend = resolve_backend()
    from app.engine import OpenAICompatBackend
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.model == "test-model"


# ---------- Ambang dibekukan ----------
def test_thresholds_match_frozen_design():
    assert THRESHOLD_OK == 0.75
    assert THRESHOLD_WARN == 0.50
