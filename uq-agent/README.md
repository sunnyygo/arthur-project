# UQ Agent — Sistem Kuantifikasi Ketidakpastian Langkah-demi-Langkah

Prototipe penelitian TA: *"Rancang Bangun Sistem Kuantifikasi Ketidakpastian
Langkah-demi-Langkah pada Agent Riset Otonom"* (FASTIKOM UNSIQ).

## Apa ini
Agent riset 4 tahap (pencarian → ekstraksi → sintesis → penyimpulan) yang setiap
langkahnya diukur ketidakpastiannya:
- **Verbalized confidence** — model ditanya seberapa yakin (Tian dkk., 2023)
- **Self-consistency** — langkah sama diulang 3× dengan konteks identik
- **Skor gabungan** = 0.4·VC + 0.6·konsistensi (rumus dibekukan sebelum eksperimen)
- **Alert**: ≥0.75 OK · 0.50–0.74 PERLU CEK MANUAL · <0.50 TIDAK PASTI

## Struktur
```
app/uq.py        # modul UQ (inti penelitian) — 19 unit test
app/engine.py    # pipeline 4 tahap + backend OpenAI-compatible
app/main.py      # API FastAPI (v1.2: tanpa penyimpanan server)
app/static/      # UI web publik — light mode, alur chat, sidebar riwayat
tests/           # 48 unit test (pytest)
e2e_test.py      # uji end-to-end dengan API asli
.env             # UQ_LLM_API_KEY / BASE_URL / MODEL (chmod 600)
```

## Penyimpanan hasil (sejak v1.2)
Hasil riset **tidak disimpan di server** — memori VPS hemat. Setiap browser
menyimpan riwayat risetnya sendiri di `localStorage` (kunci
`uq_agent_history_v1`, maks 50 run). Endpoint `/api/runs` dihapus; server hanya
menyediakan `POST /api/run` dan `GET /api/health`.

## Jalankan
```bash
.venv/bin/pytest tests/ -q                                  # unit test
./.venv/bin/python -u e2e_test.py                           # uji API asli
systemctl --user start uq-agent                             # service (port 8000)
curl http://localhost:8000/api/health
```

## Ganti model
Edit `.env` → `UQ_LLM_MODEL` (mis. `hermes-4-70b` via Nous Portal), lalu
`systemctl --user restart uq-agent`.

## Catatan desain (bahan sidang)
- Temperature 0.7 (wajib >0, kalau tidak self-consistency selalu 100%)
- 1 langkah = 5 pemanggilan (4 pipeline + 1 tanya keyakinan)
- Modul UQ = penyadap pasif: mengukur tanpa mengubah keluaran agent
- Konteks tiap pengulangan identik & tercatat → pengukuran reproducible
