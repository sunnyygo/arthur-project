"""Engine pipeline riset 4 tahap + self-consistency sampling.

Desain (dari tumpukan-teknologi-dan-coding.md):
- Pipeline didefinisikan eksplisit oleh peneliti agar konteks masukan tiap
  pengulangan IDENTIK -> pengukuran self-consistency valid & reproducible.
- Modul UQ adalah penyadap pasif: mengukur, tidak mengubah keluaran agent.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import uuid
from typing import Any

from app.uq import (
    ALPHA,
    THRESHOLD_OK,
    THRESHOLD_WARN,
    alert_level,
    combine_score,
    consistency_for_stage,
    verbalized_confidence,
)

N_REPEAT = 3        # 1 asli + 3 pengulangan = 4 pemanggilan per langkah (sesuai proposal)
TEMPERATURE = 0.7   # wajib > 0; nilai ini variabel eksperimen, dicatat di Bab III


# =================================================================
# Backend model — kompatibel OpenAI (Nous Portal / gateway apa pun)
# =================================================================
class OpenAICompatBackend:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = TEMPERATURE):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(timeout=180)
        self.max_retries = int(os.environ.get("UQ_LLM_MAX_RETRIES", "4"))
        self.retry_delay = float(os.environ.get("UQ_LLM_RETRY_DELAY", "8"))

    def chat(self, messages: list[dict]) -> str:
        import time as _time
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages, "temperature": self.temperature},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:  # 5xx gateway, timeout, connection error -> retry
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                # 4xx selain 429 = kesalahan permintaan, bukan sementara -> jangan diulang
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                if attempt < self.max_retries:
                    delay = self.retry_delay * attempt
                    print(f"[retry {attempt}/{self.max_retries}] {type(e).__name__} ({status or 'network'}) tunggu {delay:.0f}s", flush=True)
                    _time.sleep(delay)
        raise last_err


class MockBackend:
    """Backend deterministik-per-prompt (untuk test & demo tanpa API)."""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self.model = "mock"

    def chat(self, messages: list[dict]) -> str:
        prompt = messages[-1]["content"]
        h = hashlib.sha256(prompt.encode()).hexdigest()
        # perilaku meniru LLM asli: temperature 0 -> deterministik; > 0 -> sampling bervariasi
        if self.temperature <= 0:
            return f"jawaban[{h[:8]}]"
        rng = random.Random(f"{h}|{time.time_ns()}")
        variant = rng.choice(["", "var1", "var2", "var3"])
        return f"jawaban[{h[:8]}]{variant}"


def resolve_backend() -> Any:
    """Mock kalau tidak ada key; HTTP backend kalau ada (konfigurasi via env)."""
    api_key = os.environ.get("UQ_LLM_API_KEY")
    if not api_key:
        return MockBackend()
    return OpenAICompatBackend(
        base_url=os.environ.get("UQ_LLM_BASE_URL", "https://portal.nousresearch.com/v1"),
        api_key=api_key,
        model=os.environ.get("UQ_LLM_MODEL", "hermes-4-70b"),
    )


# =================================================================
# Pipeline 4 tahap — setiap tahap punya masukan eksplisit & tercatat
# =================================================================
VC_PROMPT = (
    "Setelah menghasilkan jawaban di atas, sebutkan tingkat keyakinanmu pada jawaban itu. "
    "Tulis dulu ALASAN singkat keyakinanmu (1-2 kalimat: misalnya dukungan sumber, konsistensi antar klaim, "
    "atau ada bagian yang tidak terjawab). Baris TERAKHIR harus HANYA angka 0-100 "
    "(persen keyakinan), tanpa penjelasan."
)


class ResearchEngine:
    def __init__(self, backend: Any, use_literature_search: bool | None = None):
        self.backend = backend
        self.model = getattr(backend, "model", "unknown")
        if use_literature_search is None:
            # default: aktif hanya untuk backend asli; MockBackend (unit test,
            # demo offline) tidak boleh nembak jaringan agar tetap deterministik
            use_literature_search = not isinstance(backend, MockBackend)
        self.use_literature_search = use_literature_search

    # ----- satu pemanggilan tahap (fungsi murni dari masukan -> keluaran) -----
    def _call_stage(self, stage: str, context: str) -> str:
        prompts = {
            "pencarian": (
                "Kamu agent riset akademik. Untuk pertanyaan riset berikut, kembalikan 3 sumber "
                "(paper/artikel) relevan dalam format JSON array of objects dengan field: "
                "title, authors, year, url, abstract. Field abstract WAJIB disalin VERBATIM "
                "dari baris \"ABSTRAK:\" milik sumber terpilih (salin persis, tanpa mengubah "
                "kata); jika sumber tidak punya baris ABSTRAK, isi string kosong.\n\n"
                "Pertanyaan riset: " + context
            ),
            "ekstraksi": (
                "Dari sumber-sumber berikut, ekstraksi 3-5 klaim faktual utama yang relevan "
                "dengan pertanyaan riset.\n\n"
                "ATURAN GROUNDING (wajib jika sumber memuat teks abstrak): setiap klaim yang "
                "bersandar pada sumber ber-abstrak harus disertai field \"quote\" berisi KUTIPAN "
                "VERBATIM — salin persis minimal 8 kata berturut-turut dari abstrak sumber itu, "
                "TANPA mengubah kata. Klaim tanpa kutipan verbatim yang bisa ditemukan di abstrak "
                "dianggap TIDAK VALID.\n"
                "JIKA tidak ada satu pun sumber yang memuat abstrak, kembalikan array JSON dengan "
                "field: claim, source_title (tanpa quote) — jangan menolak tugas.\n\n"
                "Format JSON array of objects.\n\n"
                "Sumber (dengan abstrak asli bila tersedia):\n" + context
            ),
            "sintesis": (
                "Rangkum klaim-klaim berikut menjadi satu paragraf sintesis koheren (Bahasa Indonesia). "
                "JANGAN menambahkan informasi baru yang tidak ada di klaim.\n\nKlaim:\n" + context
            ),
            "penyimpulan": (
                "Berdasarkan sintesis berikut, tarik kesimpulan (2-3 kalimat) yang menjawab "
                "pertanyaan riset, tanpa melampaui bukti yang ada.\n\nSintesis:\n" + context
            ),
        }
        messages = [{"role": "user", "content": prompts[stage]}]
        return self.backend.chat(messages)

    def _ask_confidence(self, stage_output: str) -> str:
        return self.backend.chat([
            {"role": "user", "content": stage_output + "\n\n" + VC_PROMPT}
        ])

    # ----- pengukuran UQ pada satu tahap: penyadap pasif -----
    def _measure_step(self, stage: str, context: str) -> dict:
        output = self._call_stage(stage, context)          # 1 pemanggilan asli
        reps = [self._call_stage(stage, context) for _ in range(N_REPEAT)]  # 3 pengulangan konteks identik
        vc_raw = self._ask_confidence(output)
        vc = verbalized_confidence(vc_raw)
        cons = consistency_for_stage(stage, self._normalize_runs(stage, [output] + reps))
        score = combine_score(vc=vc, cons=cons, alpha=ALPHA)
        return {
            "stage": stage,
            "input_context": context,
            "output": output,
            "repetitions": reps,
            "verbalized_raw": vc_raw[:200],
            "verbalized_confidence": round(vc, 4),
            "consistency": round(cons, 4),
            "score": round(score, 4),
            "alert": alert_level(score),
            "n_calls": 1 + N_REPEAT + 1,  # asli + pengulangan + tanya keyakinan
            "temperature": getattr(self.backend, "temperature", TEMPERATURE),
        }

    @staticmethod
    def _normalize_runs(stage: str, runs: list[str]) -> Any:
        """Ubah keluaran mentah jadi bentuk yang bisa dibandingkan per tahap."""
        if stage == "pencarian":
            out = []
            for r in runs:
                titles = re.findall(r'"title"\s*:\s*"([^"]+)"', r, re.IGNORECASE)
                out.append(titles if titles else [r[:80]])
            return out
        if stage == "ekstraksi":
            out = []
            for r in runs:
                claims = re.findall(r'"claim"\s*:\s*"([^"]+)"', r, re.IGNORECASE)
                out.append({"claims": "|".join(claims) if claims else r[:120]})
            return out
        return runs  # sintesis & penyimpulan: teks langsung

    # ----- alur skenario lengkap: pencarian -> ekstraksi -> sintesis -> penyimpulan -----
    def run_scenario(self, question: str) -> dict:
        run_id = uuid.uuid4().hex[:12]
        t0 = time.time()
        steps: list[dict] = []

        # ---- pencarian real-time (opsi A): API akademik dipanggil SEKALI per run,
        # hasilnya di-inject ke konteks SEMUA pengulangan tahap pencarian agar
        # konteks tetap identik & self-consistency tetap valid.
        lit: dict | None = None
        if self.use_literature_search:
            try:
                from app.search import format_for_prompt, search_literature
                lit = search_literature(question)
                if not lit["fallback_llm"]:
                    question = (
                        question + "\n\n" + format_for_prompt(lit["merged"])
                    )
            except Exception:
                lit = None  # fallback ke mode LLM murni, sistem tetap jalan

        s1 = self._measure_step("pencarian", question)
        s1["literature_search"] = {
            "used": bool(lit and not lit["fallback_llm"]),
            "providers_found": {k: len(v) for k, v in (lit or {}).get("providers", {}).items()},
            "errors": (lit or {}).get("errors", {}),
            "n_papers_injected": len((lit or {}).get("merged", [])),
        }
        steps.append(s1)
        s2 = self._measure_step("ekstraksi", s1["output"])
        steps.append(s2)
        s3 = self._measure_step("sintesis", s2["output"])
        steps.append(s3)
        s4 = self._measure_step("penyimpulan", s3["output"])
        steps.append(s4)

        return {
            "run_id": run_id,
            "scenario": question,
            "model": self.model,
            "temperature": getattr(self.backend, "temperature", TEMPERATURE),
            "alpha": ALPHA,
            "thresholds": {"ok": THRESHOLD_OK, "warn": THRESHOLD_WARN},
            "started_at": t0,
            "finished_at": time.time(),
            "duration_s": round(time.time() - t0, 2),
            "steps": steps,
        }
