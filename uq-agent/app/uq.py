"""Modul UQ — Kuantifikasi Ketidakpastian langkah-demi-langkah.

Implementasi dari desain riset (proposal TA, lihat /home/ubuntu/shared/):
- Pengukuran A: verbalized confidence (Tian dkk., 2023)
- Pengukuran B: self-consistency 3x pengulangan
- Penggabungan skor: dibekukan SEBELUM eksperimen (prinsip P4 rubrik)
- Prinsif: modul UQ adalah penyadap pasif — tidak mengubah keluaran agent.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------
# Konstanta desain yang DIBEKUKAN (dicatat juga di catatan-desain.md)
# ---------------------------------------------------------------
ALPHA = 0.4            # bobot verbalized confidence; konsistensi dapat (1 - ALPHA)
THRESHOLD_OK = 0.75    # skor >= ambang ini -> OK
THRESHOLD_WARN = 0.50  # skor >= ambang ini (dan < OK) -> PERLU CEK MANUAL; di bawah -> TIDAK PASTI
STAGES = ("pencarian", "ekstraksi", "sintesis", "penyimpulan")


# ---------------------------------------------------------------
# Pengukuran A — verbalized confidence
# ---------------------------------------------------------------
def verbalized_confidence(text: str) -> float:
    """Ambil angka keyakinan dari jawaban model (0.0–1.0).

    Tian dkk. (2023): model cenderung overconfident & hanya mengeluarkan
    segelintir nilai berulang. Kalau parsing gagal, netral 0.5 —
    tidak menguntungkan sistem yang diuji.
    """
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:%|persen|percent)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1).replace(",", "."))
        if val > 1.0:
            val = val / 100.0
        return min(val, 1.0)
    m = re.search(r"0\.\d+|1\.0", text)
    if m:
        return min(float(m.group(0)), 1.0)
    # jawaban polos berupa satu angka saja (prompt UQ meminta "HANYA angka 0-100")
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", text)
    if m:
        val = float(m.group(1).replace(",", "."))
        if val > 1.0:
            val = val / 100.0
        return min(val, 1.0)
    return 0.5


# ---------------------------------------------------------------
# Pengukuran B — pembanding konsistensi per tahap
# ---------------------------------------------------------------
def _jaccard(sets: list[set]) -> float:
    union = set().union(*sets) if sets else set()
    if not union:
        return 0.0
    inter = set.intersection(*sets)
    return len(inter) / len(union)


def consistency_search(runs: list[list[str]]) -> float:
    """Pencarian: irisan himpunan sumber (Jaccard) antar percobaan."""
    return _jaccard([set(r) for r in runs])


def consistency_exact(runs: list[dict]) -> float:
    """Ekstraksi: pecahan percobaan yang identik dengan modus."""
    if not runs:
        return 0.0
    counts: dict[tuple, int] = {}
    for r in runs:
        key = tuple(sorted((k, str(v)) for k, v in r.items()))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / len(runs)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def consistency_semantic(runs: list[str]) -> float:
    """Sintesis: kemiripan makna antar rangkuman.

    Versi ringan memakai rata-rata Jaccard token (tanpa dependency
    sentence-transformers — cocok untuk VPS 2GB). Identik = 1.0.
    """
    token_sets = [_tokenize(t) for t in runs]
    if not token_sets:
        return 0.0
    n = len(token_sets)
    if n == 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            union = token_sets[i] | token_sets[j]
            if not union:
                continue
            total += len(token_sets[i] & token_sets[j]) / len(union)
            pairs += 1
    return total / pairs if pairs else 0.0


def consistency_direction(runs: list[str]) -> float:
    """Penyimpulan: kesamaan arah kesimpulan (pola kata arah)."""
    POS = {"lebih", "meningkat", "naik", "efektif", "positif", "baik", "mendukung",
           "setuju", "dapat", "valid", "andal", "signifikan"}
    NEG = {"menurun", "turun", "buruk", "negatif", "tidak", "lemah", "gagal",
           "menolak", "overconfident", "meragukan", "rendah"}
    dirs: list[str] = []
    for t in runs:
        toks = set(re.findall(r"[a-z]+", t.lower()))
        score = len(toks & POS) - len(toks & NEG)
        dirs.append("pos" if score > 0 else "neg" if score < 0 else "net")
    if not dirs:
        return 0.0
    from collections import Counter
    return max(Counter(dirs).values()) / len(dirs)


def consistency_for_stage(stage: str, runs: Any) -> float:
    """Dispatcher: cara membandingkan konsistensi berbeda per jenis tahap."""
    if stage == "pencarian":
        return consistency_search(runs)
    if stage == "ekstraksi":
        return consistency_exact(runs)
    if stage == "sintesis":
        return consistency_semantic(runs)
    if stage == "penyimpulan":
        return consistency_direction(runs)
    raise ValueError(f"tahap tidak dikenal: {stage}")


# ---------------------------------------------------------------
# Penggabungan skor & alert
# ---------------------------------------------------------------
def combine_score(vc: float, cons: float, alpha: float = ALPHA) -> float:
    """Rumus penggabungan: skor = alpha*VC + (1-alpha)*konsistensi.

    Rumus ditetapkan sebelum eksperimen (keputusan desain, bukan hasil ukur).
    Konsistensi diberi bobot lebih besar karena verbalized confidence terbukti
    overconfident (Tian dkk., 2023).
    """
    if not (0.0 <= vc <= 1.0) or not (0.0 <= cons <= 1.0):
        raise ValueError("vc dan cons harus dalam rentang 0..1")
    return alpha * vc + (1 - alpha) * cons


def alert_level(score: float) -> str:
    """Ambang alert dibekukan sebelum eksperimen."""
    if not (0.0 <= score <= 1.0):
        raise ValueError("skor harus dalam rentang 0..1")
    if score >= THRESHOLD_OK:
        return "OK"
    if score >= THRESHOLD_WARN:
        return "PERLU CEK MANUAL"
    return "TIDAK PASTI"
