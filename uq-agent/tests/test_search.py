"""Unit test untuk app/search.py (pencarian literatur real-time).

Provider eksternal di-mock (tidak ada network) — yang diuji logika:
parsing, dedup, filter sitasi, fallback, dan ketahanan terhadap error.
Pengujian API asli ada di test_search_live.py (ditandai, jalan manual).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search import (  # noqa: E402
    _dedup,
    format_for_prompt,
    search_literature,
)


# -----------------------------------------------------------------_dedup
class TestDedup:
    def test_hapus_duplicat_judul_sama(self):
        papers = [
            {"title": "A Survey on LLM Agents", "year": "2024", "authors": [], "url": "u1"},
            {"title": "A Survey on LLM Agents", "year": "2024", "authors": [], "url": "u2"},
        ]
        assert len(_dedup(papers)) == 1

    def test_judul_beda_tetap_dua(self):
        papers = [
            {"title": "Paper A", "year": "2024", "authors": [], "url": "u1"},
            {"title": "Paper B", "year": "2023", "authors": [], "url": "u2"},
        ]
        assert len(_dedup(papers)) == 2

    def test_judul_kosong_dibuang(self):
        papers = [{"title": "", "year": "", "authors": [], "url": "u"}]
        assert _dedup(papers) == []


# ------------------------------------------------------------format prompt
class TestFormatForPrompt:
    def test_kosong_menghasilkan_string_kosong(self):
        assert format_for_prompt([]) == ""

    def test_format_memuat_judul_url_dan_sitasi(self):
        papers = [{
            "title": "Survey Agen", "year": "2024", "authors": ["Ani", "Budi"],
            "url": "https://doi.org/10.1/x", "venue": "Jurnal AI", "cited_by": 42,
        }]
        teks = format_for_prompt(papers)
        assert "Survey Agen" in teks
        assert "https://doi.org/10.1/x" in teks
        assert "42" in teks
        assert "Ani" in teks

    def test_limit_bekerja(self):
        papers = [
            {"title": f"P{i}", "year": "2024", "authors": [], "url": f"u{i}"}
            for i in range(10)
        ]
        teks = format_for_prompt(papers, limit=3)
        baris_entri = [ln for ln in teks.splitlines() if ln[:2].rstrip(".").isdigit()]
        assert len(baris_entri) == 3

    def test_abstrak_disertakan_kalau_ada(self):
        papers = [{
            "title": "Grounded Extraction", "year": "2024", "authors": ["A B"],
            "url": "u", "abstract": "Agents plan before they act in unknown domains.",
        }]
        teks = format_for_prompt(papers)
        assert "ABSTRAK:" in teks
        assert "Agents plan before they act" in teks

    def test_tanpa_abstrak_tidak_muncul_baris_kosong(self):
        papers = [{"title": "No Abs", "year": "2024", "authors": [], "url": "u"}]
        teks = format_for_prompt(papers)
        assert "ABSTRAK:" not in teks

    def test_openalex_inverted_index_direkonstruksi(self):
        from app.search import _abstract_openalex
        inv = {"Agents": [0], "plan": [1], "ahead": [2]}
        assert _abstract_openalex(inv) == "Agents plan ahead"
        assert _abstract_openalex(None) == ""
        assert _abstract_openalex({}) == ""


# -------------------------------------------------------search_literature
class TestSearchLiterature:
    def test_semua_provider_gagal_tetap_return_tanpa_raise(self, monkeypatch):
        import app.search as s

        def boom(*a, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(s, "search_arxiv", boom)
        monkeypatch.setattr(s, "search_crossref", boom)
        monkeypatch.setattr(s, "search_openalex", boom)
        r = search_literature("test query")
        assert r["total"] == 0
        assert r["fallback_llm"] is True
        assert set(r["errors"]) == {"arxiv", "crossref", "openalex"}

    def test_hasil_gabungan_dan_dedup(self, monkeypatch):
        import app.search as s

        monkeypatch.setattr(s, "search_arxiv", lambda q, m: [
            {"title": "Paper Sama", "year": "2024", "authors": ["X"], "url": "a1"}])
        monkeypatch.setattr(s, "search_crossref", lambda q, m: [
            {"title": "Paper Sama", "year": "2024", "authors": ["X"], "url": "c1",
             "venue": "J", "cited_by": 9}])
        monkeypatch.setattr(s, "search_openalex", lambda q, m: [
            {"title": "Paper Lain", "year": "2023", "authors": ["Y"], "url": "o1",
             "venue": "K", "cited_by": 3}])
        r = search_literature("q")
        assert r["total"] == 3          # sebelum dedup
        assert len(r["merged"]) == 2    # sesudah dedup
        assert r["fallback_llm"] is False

    def test_satu_provider_gagal_lainnya_jalan(self, monkeypatch):
        import app.search as s

        def boom(*a, **kw):
            raise RuntimeError("x")

        monkeypatch.setattr(s, "search_arxiv", boom)
        monkeypatch.setattr(s, "search_crossref", lambda q, m: [])
        monkeypatch.setattr(s, "search_openalex", lambda q, m: [
            {"title": "P", "year": "2024", "authors": [], "url": "u", "cited_by": 7}])
        r = search_literature("q")
        assert r["fallback_llm"] is False
        assert "arxiv" in r["errors"]
        assert r["merged"][0]["title"] == "P"
