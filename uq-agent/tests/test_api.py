"""Tests API: jalankan skenario, list riwayat, detail run, health."""
import json
import pytest
from fastapi.testclient import TestClient
from app import main
from app.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_run_scenario_returns_trace_without_persistence(client):
    """v1.2: hasil riset disimpan di browser (localStorage), BUKAN di VPS."""
    r = client.post("/api/run", json={"scenario": "Apakah verbalized confidence bisa diandalkan?"})
    assert r.status_code == 200
    body = r.json()
    assert body["scenario"].startswith("Apakah")
    assert len(body["steps"]) == 4
    assert body["steps"][0]["stage"] == "pencarian"
    # v1.2: server tidak lagi punya penyimpanan (DATA_DIR dihapus dari main)
    assert not hasattr(main, "DATA_DIR")


def test_list_runs_endpoint_removed(client):
    """v1.2: endpoint riwayat server dihapus — riwayat pindah ke localStorage."""
    client.post("/api/run", json={"scenario": "skenario A"})
    assert client.get("/api/runs").status_code == 404
    assert client.get("/api/runs/abc123").status_code == 404


def test_get_run_404(client):
    assert client.get("/api/runs/tidakada").status_code == 404


def test_empty_scenario_rejected(client):
    assert client.post("/api/run", json={"scenario": "  "}).status_code == 422


def test_index_serves_light_chat_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    html = r.text
    # light mode (bukan dark)
    assert 'data-theme="light"' in html
    # alur chat: bubble user + kartu agent + composer terpisah
    assert "msg-user" in html and "msg-agent" in html
    # sidebar riwayat terpisah dari area ketik
    assert "Riset Terbaru" in html
    # riwayat disimpan lokal di browser
    assert "localStorage" in html
    # v1.3: sidebar posisinya di KIRI (disebut eksplisit di komentar CSS)
    assert "sidebar kiri" in html
    # v1.3: UX interaktif — animasi kecil + penghormatan reduced-motion
    assert "@keyframes" in html
    assert "prefers-reduced-motion" in html
    # skeleton loader menggantikan spinner teks polos
    assert "skeleton" in html
    # transisi halus di elemen interaktif + feedback tekan
    assert ":active" in html
    assert "transition" in html
    # v1.3.1: tombol clear chat — konfirmasi via modal custom (v1.3.3, bukan confirm() bawaan)
    assert "hapusRiwayat" in html
    # v1.3.2: tombol tutup thread chat yang terbuka
    assert "tutupThread" in html
    # v1.3.2: jam pada bubble chat (user & agent)
    assert "msg-time" in html
    assert "jamSekarang" in html
    # v1.3.3: modal persetujuan custom di tengah halaman (bukan confirm() bawaan)
    assert "modal-overlay" in html
    assert "mintaPersetujuan" in html
    # v1.3.3: dedup — klik riset sama langsung fokus ke chat terbuka
    assert "klikRiset" in html
    assert "fokusThread" in html
    # v1.3.3: heading hero kembali saat tidak ada riset terbuka
    assert "pulihkanHero" in html
    # v1.3.4: keluaran agent user-friendly — JSON diparse jadi kartu sumber & daftar klaim
    assert "formatOutput" in html
    assert "parseJsonBlock" in html
    assert "src-card" in html
    # v1.3.5: ringkasan sumber literatur selalu tampil di kartu agent, lebih jelas
    assert "lit-box" in html and "Sumber literatur" in html and "src-num" in html
    # v1.3.6: scenario terpollusi grounding dibersihkan saat render + klaim tanpa nama sumber
    assert "bersihkanScenario" in html
    assert "claim-src" not in html
    # v1.3.7: parseJsonBlock nangkep JSON tanpa fence (raw array) — kasus query non-teknologi
    assert "JSON.parse(raw)" in html
    assert "claim-item" in html
