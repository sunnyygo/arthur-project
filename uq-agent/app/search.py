"""Pencarian literatur real-time untuk tahap PENCARIAN (tambahan opsi A).

Multi-provider, semua API publik & gratis tanpa API key:
- arXiv      : preprint CS/ML/AI (Atom XML)
- Crossref   : DOI registrar — jurnal internasional bereputasi
- OpenAlex   : indeks scholarly terbuka (jaringan Crossref, cakupan terluas,
               termasuk jurnal Indonesia yang terindeks internasional)

Catatan desain (penting untuk validitas eksperimen):
- Modul pencarian = ALAT yang dipanggil tahap pencarian, bukan bagian modul UQ.
  Self-consistency tetap mengukur KELUARAN MODEL; provider pencarian dipanggil
  SEKALI per run (bukan per pengulangan) supaya perbedaan antar pengulangan
  murni berasal dari sampling model, bukan fluktuasi hasil API pencarian.
- Semua provider gagal -> fallback: tahap pencarian kembali ke mode LLM
  (perilaku lama), sistem tetap jalan.
- TANPA key & network sniffing; timeout pendek; tidak pernah melempar exception
  ke pemanggil (selalu return list, kosong bila gagal).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

TIMEOUT = 12.0
UA = "uq-agent-research/1.0 (thesis prototype; FASTIKOM UNSIQ)"

_ATOM = "{http://www.w3.org/2005/Atom}"


def _client() -> "httpx.Client":
    return httpx.Client(timeout=TIMEOUT, headers={"User-Agent": UA}, follow_redirects=True)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# =====================================================================
# arXiv — preprint CS/ML/AI
# =====================================================================
def search_arxiv(query: str, max_results: int = 4) -> list[dict]:
    """API resmi arXiv (Atom XML). Referensi: export.arxiv.org/api/query."""
    q = re.sub(r'[&"+]', " ", query).strip()
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{q}&start=0&max_results={max_results}"
        "&sortBy=relevance"
    )
    with _client() as c:
        resp = c.get(url)
        resp.raise_for_status()
    root = ET.fromstring(resp.text)
    out = []
    for entry in root.findall(f"{_ATOM}entry"):
        arxiv_id = _clean(entry.findtext(f"{_ATOM}id", ""))
        summary = _clean(entry.findtext(f"{_ATOM}summary", ""))
        out.append({
            "title": _clean(entry.findtext(f"{_ATOM}title", "")),
            "authors": [
                _clean(a.findtext(f"{_ATOM}name", ""))
                for a in entry.findall(f"{_ATOM}author")
            ][:6],
            "year": (entry.findtext(f"{_ATOM}published", "") or "")[:4],
            "url": arxiv_id or "",
            "venue": "arXiv",
            "abstract": summary[:600] if summary else "",
        })
    return [p for p in out if p["title"]]


# =====================================================================
# Crossref — jurnal internasional bereputasi (ber-DOI)
# =====================================================================
def search_crossref(query: str, max_results: int = 4) -> list[dict]:
    """API publik Crossref. Filter type=journal-article + minimal 5 sitasi
    agar hanya artikel jurnal yang sudah diakui (is-referenced-by-count)."""
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": max_results * 3,          # ambil lebih banyak, sift manual
        "select": "title,author,issued,DOI,container-title,is-referenced-by-count,URL,abstract",
        "filter": "type:journal-article",
        "sort": "relevance",
    }
    with _client() as c:
        resp = c.get(url, params=params)
        resp.raise_for_status()
    items = (resp.json().get("message") or {}).get("items") or []
    out = []
    for it in items:
        cited = int(it.get("is-referenced-by-count") or 0)
        if cited < 5:                      # belum terbukti diakui -> buang
            continue
        year = ((it.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        out.append({
            "title": _clean((it.get("title") or [""])[0]),
            "authors": [
                _clean(f"{a.get('given','')} {a.get('family','')}").strip()
                for a in (it.get("author") or [])[:6]
            ],
            "year": str(year) if year else "",
            "url": it.get("URL") or f"https://doi.org/{it.get('DOI','')}",
            "venue": _clean((it.get("container-title") or [""])[0]) or "Crossref",
            "cited_by": cited,
            "abstract": _clean(re.sub(r"<[^>]+>", " ", it.get("abstract") or ""))[:600],
        })
        if len(out) >= max_results:
            break
    return out


# =====================================================================
# OpenAlex — indeks terbuka terluas (termasuk artikel Indonesia)
# =====================================================================
def _abstract_openalex(inv: dict | None, max_words: int = 100) -> str:
    """Rekonstruksi abstrak dari inverted index OpenAlex
    (format {token: [posisi]}). Return "" bila kosong/gagal."""
    if not inv:
        return ""
    try:
        positions: list[tuple[int, str]] = []
        for token, idxs in inv.items():
            for i in idxs:
                positions.append((i, token))
        positions.sort()
        return _clean(" ".join(t for _, t in positions[:max_words]))
    except Exception:
        return ""


def search_openalex(query: str, max_results: int = 4) -> list[dict]:
    """API publik OpenAlex. Cakupan paling luas; cover jurnal Indonesia yang
    terindeks internasional (pengganti GARUDA/SINTA yang API publiknya tidak
    stabil dan tidak resmi untuk akses programatik)."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": max_results * 3,
        "select": "title,publication_year,doi,cited_by_count,primary_location,authorships,abstract_inverted_index",
        "sort": "relevance_score:desc",
    }
    with _client() as c:
        resp = c.get(url, params=params)
        resp.raise_for_status()
    results = (resp.json() or {}).get("results") or []
    out = []
    for it in results:
        if int(it.get("cited_by_count") or 0) < 5:
            continue
        loc = it.get("primary_location") or {}
        src = (loc.get("source") or {}).get("display_name") or "OpenAlex"
        out.append({
            "title": _clean(it.get("title") or ""),
            "authors": [
                _clean(a.get("author", {}).get("display_name", ""))
                for a in (it.get("authorships") or [])[:6]
            ],
            "year": str(it.get("publication_year") or ""),
            "url": it.get("doi") or (loc.get("landing_page_url") or ""),
            "venue": _clean(src),
            "cited_by": int(it.get("cited_by_count") or 0),
            "abstract": _abstract_openalex(it.get("abstract_inverted_index")),
        })
        if len(out) >= max_results:
            break
    return out


# =====================================================================
# Agregator multi-provider
# =====================================================================
def _dedup(papers: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in papers:
        key = re.sub(r"[^a-z0-9]", "", p["title"].lower())[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def search_literature(query: str, max_per_provider: int = 4) -> dict:
    """Gabung hasil semua provider. Return dict per provider + flag fallback.

    Tidak pernah raise — error per provider ditelan dan dilaporkan di 'errors'.
    """
    result: dict[str, Any] = {"providers": {}, "errors": {}, "total": 0}
    for name, fn in (
        ("arxiv", search_arxiv),
        ("crossref", search_crossref),
        ("openalex", search_openalex),
    ):
        try:
            papers = fn(query, max_per_provider)
            result["providers"][name] = papers
            result["total"] += len(papers)
        except Exception as e:  # jangan biarkan satu provider menggagalkan run
            result["errors"][name] = f"{type(e).__name__}: {e}"[:150]
            result["providers"][name] = []
    merged = _dedup(
        result["providers"]["arxiv"]
        + result["providers"]["crossref"]
        + result["providers"]["openalex"]
    )
    result["merged"] = merged
    result["fallback_llm"] = result["total"] == 0   # semua gagal -> mode lama
    return result


def format_for_prompt(papers: list[dict], limit: int = 8) -> str:
    """Format hasil pencarian jadi teks untuk prompt tahap PENCARIAN.

    v1.1: menyertakan abstrak asli (kalau ada) sebagai bahan grounding
    untuk tahap EKSTRAKSI — klaim wajib dikutip verbatim dari abstrak.
    """
    if not papers:
        return ""
    lines = []
    for i, p in enumerate(papers[:limit], 1):
        authors = ", ".join(a for a in p["authors"] if a) or "unknown"
        cited = f" · disitasi {p['cited_by']}x" if p.get("cited_by") else ""
        lines.append(
            f"{i}. {p['title']} ({p.get('year','n.d.')}) — {authors}. "
            f"{p.get('venue','')}{cited}. {p['url']}"
        )
        abstrak = (p.get("abstract") or "").strip()
        if abstrak:
            lines.append(f"   ABSTRAK: {abstrak}")
    return (
        "HASIL PENCARIAN LITERATUR (API akademik real-time):\n"
        + "\n".join(lines)
        + "\n\nGunakan sumber di atas sebagai kandidat utama. Kamu boleh "
        "menyaring/menambahkan maksimal 2 sumber lain yang kamu ketahui pasti ada."
    )
