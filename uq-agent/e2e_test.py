"""Pengujian end-to-end dengan panggilan API model asli (bukan mock).

Berdasarkan tumpukan-teknologi-dan-coding.md bagian 6:
mulai dari SATU pemanggilan API -> satu tahap -> baru lengkap.
"""
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Muat key dari file .env project (format KEY=value, dibuat terpisah, chmod 600)
env_path = BASE_DIR / ".env"
if env_path.is_file():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("UQ_LLM_API_KEY"):
    sys.exit("UQ_LLM_API_KEY kosong — isi file .env dulu.")

from app.engine import OpenAICompatBackend, ResearchEngine  # noqa: E402


def main():
    backend = OpenAICompatBackend(
        base_url=os.environ["UQ_LLM_BASE_URL"],
        api_key=os.environ["UQ_LLM_API_KEY"],
        model=os.environ["UQ_LLM_MODEL"],
    )
    eng = ResearchEngine(backend=backend)

    # LANGKAH 1: satu pemanggilan API
    print("[1/3] Tes satu pemanggilan API…", flush=True)
    t0 = time.time()
    out = backend.chat([{"role": "user", "content": "Jawab satu kata: apa ibu kota Indonesia?"}])
    print(f"      OK ({time.time()-t0:.1f}s): {out[:80]!r}", flush=True)

    # LANGKAH 2: satu langkah pipeline penuh dengan UQ (5 pemanggilan)
    print("[2/3] Tes satu langkah UQ (pencarian, 5 pemanggilan)…", flush=True)
    t0 = time.time()
    step = eng._measure_step("pencarian", "Apakah verbalized confidence bisa diandalkan?")
    print(f"      OK ({time.time()-t0:.1f}s) skor={step['score']:.2f} vc={step['verbalized_confidence']:.2f} "
          f"cons={step['consistency']:.2f} alert={step['alert']}", flush=True)
    for i, r in enumerate(step["repetitions"]):
        print(f"      percobaan {i+1}: {r[:100]!r}", flush=True)

    # LANGKAH 3: skenario lengkap 4 tahap (20 pemanggilan)
    print("[3/3] Skenario lengkap 4 tahap…", flush=True)
    t0 = time.time()
    trace = eng.run_scenario("Apakah verbalized confidence bisa diandalkan untuk mengukur keyakinan model bahasa?")
    print(f"      OK ({time.time()-t0:.1f}s)", flush=True)
    for s in trace["steps"]:
        print(f"      {s['stage']:<12} skor={s['score']:.2f} vc={s['verbalized_confidence']:.2f} "
              f"cons={s['consistency']:.2f} alert={s['alert']}", flush=True)
    print("--- potongan keluaran pencarian ---")
    print(trace["steps"][0]["output"][:400])

    # simpan sebagai run pertama
    import json
    from pathlib import Path
    data_dir = Path(__file__).parent / "data" / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{trace['run_id']}.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tersimpan: data/runs/{trace['run_id']}.json")


if __name__ == "__main__":
    main()
