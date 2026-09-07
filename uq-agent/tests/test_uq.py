"""Tests untuk Modul UQ — sesuai desain riset scholar (verbalized confidence + self-consistency)."""
import math
import pytest
from app.uq import (
    verbalized_confidence,
    consistency_search,
    consistency_exact,
    consistency_semantic,
    consistency_direction,
    consistency_for_stage,
    combine_score,
    alert_level,
)


# ---------- Verbalized confidence ----------
def test_vc_parses_number():
    assert verbalized_confidence("Keyakinan saya sekitar 85 persen.") == 0.85

def test_vc_parses_decimal():
    assert verbalized_confidence("confidence: 0.42") == pytest.approx(0.42)

def test_vc_caps_at_one():
    assert verbalized_confidence("100% yakin") == 1.0

def test_vc_neutral_when_unparseable():
    # Tian dkk. 2023: model overconfident — kalau gagal parsing, anggap 0.5 (netral, tidak menguntungkan sistem)
    assert verbalized_confidence("tidak tahu") == 0.5

def test_vc_parses_bare_number_answer():
    # prompt UQ minta "jawab HANYA angka 0-100" — model asli menjawab "95" tanpa simbol
    assert verbalized_confidence("95") == 0.95
    assert verbalized_confidence("  90 \n") == pytest.approx(0.90)
    assert verbalized_confidence("40") == pytest.approx(0.40)

def test_vc_bare_decimal_between_0_and_1():
    assert verbalized_confidence("0.85") == pytest.approx(0.85)


# ---------- Consistency per tahap ----------
def test_consistency_search_jaccard_perfect():
    runs = [["A", "B", "C"], ["A", "B", "C"], ["A", "B", "C"]]
    assert consistency_search(runs) == 1.0

def test_consistency_search_jaccard_partial():
    # irisan {A,B}, union {A,B,C,D} -> 0.5 (contoh di penjelasan-alur: dua sumber mantap, satu ragu)
    runs = [["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]]
    assert consistency_search(runs) == pytest.approx(0.5)

def test_consistency_search_empty():
    assert consistency_search([[], [], []]) == 0.0

def test_consistency_search_no_source_in_all_runs():
    # metode terdokumentasi: irisan antar KETIGA percobaan (Jaccard).
    # Tidak ada sumber yang muncul di ketiga percobaan -> 0.0
    assert consistency_search([["A"], ["A"], ["B"]]) == 0.0

def test_consistency_exact_identical():
    assert consistency_exact([{"a": 1}, {"a": 1}, {"a": 1}]) == 1.0

def test_consistency_exact_differ():
    assert consistency_exact([{"a": 1}, {"a": 2}, {"a": 1}]) == pytest.approx(2 / 3)

def test_consistency_semantic_similar_texts():
    a = "Model menunjukkan kalibrasi yang baik pada benchmark TriviaQA"
    b = "Kalibrasi model tergolong baik pada benchmark TriviaQA"
    c = "Cuaca hari ini cerah sekali di Wonosobo"
    hi = consistency_semantic([a, b, a])
    lo = consistency_semantic([a, b, c])
    assert hi > lo

def test_consistency_semantic_needs_token_overlap():
    # versi ringan tanpa sentence-transformers: overlap token, identik = 1.0
    assert consistency_semantic(["x y z", "x y z", "x y z"]) == 1.0

def test_consistency_direction_same_sign():
    assert consistency_direction(["lebih baik", "lebih baik", "lebih buruk"]) == pytest.approx(2 / 3)


# ---------- Dispatcher per tahap ----------
def test_dispatch():
    assert consistency_for_stage("pencarian", [["A", "B"], ["A", "B"], ["A", "B"]]) == 1.0
    assert consistency_for_stage("ekstraksi", [{"f": 1}, {"f": 1}, {"f": 1}]) == 1.0
    assert 0.0 <= consistency_for_stage("sintesis", ["aa bb", "aa bb", "cc"]) <= 1.0
    assert 0.0 <= consistency_for_stage("penyimpulan", ["naik", "turun", "naik"]) <= 1.0

def test_dispatch_invalid_stage():
    with pytest.raises(ValueError):
        consistency_for_stage("ngawur", [1, 2, 3])


# ---------- Penggabungan skor (rumus dibekukan sesuai catatan riset) ----------
def test_combine_score():
    # alpha=0.4 (verbalized), 1-alpha=0.6 (konsistensi) — konsistensi diberi bobot lebih besar
    assert combine_score(vc=0.9, cons=1.0, alpha=0.4) == pytest.approx(0.4 * 0.9 + 0.6 * 1.0)

def test_combine_score_default_alpha():
    assert combine_score(vc=0.5, cons=0.5) == 0.5

def test_alert_levels():
    # ambang dibekukan: >=0.75 OK, 0.50-0.74 PERLU CEK, <0.50 TIDAK PASTI
    assert alert_level(0.84) == "OK"
    assert alert_level(0.75) == "OK"
    assert alert_level(0.74) == "PERLU CEK MANUAL"
    assert alert_level(0.50) == "PERLU CEK MANUAL"
    assert alert_level(0.49) == "TIDAK PASTI"

def test_alert_boundaries_invalid():
    with pytest.raises(ValueError):
        alert_level(1.2)
    with pytest.raises(ValueError):
        alert_level(-0.1)
