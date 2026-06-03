#!/usr/bin/env python3
"""
Cite-gap audit: for the top-N items by combined_score in the Zotero ranking
CSV, look up DOI / arXiv ID and check whether they are cited in
`ai-models-ensembles-paper/paper_refs.bib`.

Output: markdown table at
    /iopsstor/scratch/cscs/sadamov/zotero_cite_gap_audit.md

Read-only against the paper repo (no writes).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from pyzotero import zotero

USER_ID = 9098647
API_KEY = "21Lw2AW5pAVWL3btSuIHSykn"
COLLECTION_KEY = "C8CXQJFN"

RANKING_CSV = Path("/iopsstor/scratch/cscs/sadamov/zotero_quality_ranking.csv")
BIB_PATH = Path(
    "/users/sadamov/pyprojects/ai-models-ensembles/ai-models-ensembles-paper/paper_refs.bib"
)
OUT_MD = Path("/iopsstor/scratch/cscs/sadamov/zotero_cite_gap_audit.md")

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,6})(?:v\d+)?", re.IGNORECASE)


def parse_bib(path: Path) -> list[dict]:
    """Parse a small bibtex file into a list of dicts."""
    text = path.read_text()
    entries: list[dict] = []
    # Greedy match of @type{key, ... } blocks. Bib entries here don't nest braces.
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.DOTALL):
        typ, key, body = m.group(1), m.group(2).strip(), m.group(3)
        doi = ""
        arxiv = ""
        author = ""
        year = ""
        title = ""
        m2 = re.search(r"doi\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if m2:
            doi = m2.group(1).strip().lower().rstrip(".")
        m2 = re.search(r"eprint\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if m2:
            arxiv = m2.group(1).strip().lower()
        m2 = re.search(r"author\s*=\s*\{([^}]+)\}", body, re.IGNORECASE | re.DOTALL)
        if m2:
            author = m2.group(1).strip()
        m2 = re.search(r"year\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if m2:
            year = m2.group(1).strip()
        m2 = re.search(r"title\s*=\s*\{([^}]+)\}", body, re.IGNORECASE | re.DOTALL)
        if m2:
            title = m2.group(1).strip()
        entries.append(
            {
                "type": typ,
                "key": key,
                "doi": doi,
                "arxiv": arxiv,
                "author": author,
                "year": year,
                "title": title,
            }
        )
    return entries


def normalise_arxiv(s: str) -> str:
    if not s:
        return ""
    m = ARXIV_RE.search(s)
    return m.group(1) if m else s.strip().lower()


def author_lastname(author: str) -> str:
    """Pull the first surname out of 'Lam, Remi and Sanchez-Gonzalez ...' or
    'Remi Lam and Alvaro Sanchez-Gonzalez ...'."""
    if not author:
        return ""
    first = author.split(" and ")[0].strip()
    if "," in first:
        return first.split(",")[0].strip().lower()
    parts = first.split()
    return parts[-1].lower() if parts else ""


def build_bib_index(entries: list[dict]) -> tuple[dict, dict, dict]:
    """Return (doi -> cite_key, arxiv -> cite_key, (lastname, year) -> cite_key)."""
    by_doi: dict[str, str] = {}
    by_arxiv: dict[str, str] = {}
    by_author_year: dict[tuple[str, str], str] = {}
    for e in entries:
        if e["doi"]:
            by_doi[e["doi"]] = e["key"]
        if e["arxiv"]:
            by_arxiv[normalise_arxiv(e["arxiv"])] = e["key"]
        ln = author_lastname(e["author"])
        if ln and e["year"]:
            by_author_year[(ln, e["year"])] = e["key"]
    return by_doi, by_arxiv, by_author_year


def extract_item_ids(z_item_data: dict) -> tuple[str, str]:
    """Return (doi, arxiv) from a Zotero item dict."""
    doi = (z_item_data.get("DOI") or "").strip().lower().rstrip(".")
    arxiv = ""
    archive_id = z_item_data.get("archiveID") or ""
    if archive_id.lower().startswith("arxiv:"):
        arxiv = archive_id.split(":", 1)[1].strip()
    if not arxiv:
        # Try url / extra
        for blob in [
            z_item_data.get("url") or "",
            z_item_data.get("extra") or "",
            z_item_data.get("libraryCatalog") or "",
        ]:
            m = ARXIV_RE.search(blob)
            if m:
                arxiv = m.group(1)
                break
    return doi, normalise_arxiv(arxiv)


def first_author_lastname_from_creators(creators: list[dict]) -> str:
    for c in creators or []:
        if c.get("creatorType") in ("author", None) or "creatorType" not in c:
            ln = (c.get("lastName") or "").strip()
            if ln:
                return ln.lower()
            name = (c.get("name") or "").strip()
            if name:
                parts = name.split()
                return parts[-1].lower() if parts else ""
    return ""


def short_note(title: str, abstract: str) -> str:
    """Generate a one-line 'why might we want this' note based on title+abstract."""
    t = (title or "").lower()
    a = (abstract or "").lower()[:500]
    blob = t + " " + a

    rules = [
        ("dropout|bayesian|epistemic", "Bayesian / epistemic uncertainty primer"),
        ("conformal", "conformal prediction for distribution-free UQ"),
        ("evidential", "evidential deep learning UQ baseline"),
        ("deep ensemble", "deep ensemble baseline for UQ"),
        ("scoringrules|scoring rule", "scoring-rules toolkit / methodology"),
        ("crps|continuous ranked probability", "CRPS calibration / scoring reference"),
        ("variogram", "variogram-score multivariate verification"),
        ("energy score", "energy-score multivariate verification"),
        ("ssim|structural similarity", "spatial-pattern verification metric"),
        ("ssr|spread.{0,3}skill", "spread-skill ratio / calibration"),
        ("model soup|weight averag|stochastic weight", "weight-averaging / SWA / ensembling"),
        ("singular vector", "singular-vector IC perturbation reference"),
        ("eda|ensemble of data assimilation", "EDA IC perturbation reference"),
        (
            "stochastic.{0,30}physics|stochastic param|sppt|spp",
            "stochastic-physics parametrisation reference",
        ),
        ("sfno|spherical fourier", "SFNO architecture detail"),
        ("graphcast|icosahedral", "GraphCast architecture / mesh detail"),
        ("aurora", "Aurora architecture / weights"),
        ("aifs", "AIFS architecture / training"),
        ("fourcastnet|fcn3", "FourCastNet baseline architecture"),
        ("gencast", "GenCast probabilistic baseline"),
        ("pangu", "Pangu-Weather deterministic baseline"),
        ("nowcast|sub.{0,3}seasonal", "nowcasting / subseasonal-range ML weather"),
        ("era5|reanalysis", "ERA5 / reanalysis training data reference"),
        ("foundation model|pretrain", "foundation-model / pretraining perspective"),
        ("interpretab|explain|attribution", "interpretability / XAI for weather ML"),
        ("calibrat", "calibration / post-processing reference"),
        ("hurricane|tropical cyclone", "TC-specific ML verification"),
        ("precipitation", "precipitation forecasting reference"),
        ("downscal", "downscaling / super-resolution"),
        ("transformer|attention", "transformer / attention architecture component"),
        ("graph neural|gnn|message passing", "graph-NN architecture component"),
        ("neural operator", "neural-operator architecture component"),
        ("normaliz.*flow|diffusion model", "generative-model probabilistic baseline"),
        ("verification|benchmark", "verification / benchmark methodology"),
        ("ensemble forecast|probabilistic forecast", "general probabilistic-forecasting reference"),
        ("limited area|regional|km.scale", "regional / km-scale weather ML"),
    ]
    matched = []
    for pat, note in rules:
        if re.search(pat, blob):
            matched.append(note)
        if len(matched) >= 2:
            break
    if not matched:
        return "general topical relevance (no specific tag matched)"
    return "; ".join(matched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    print(f"[1/4] parsing {BIB_PATH.name} ...", flush=True)
    bib_entries = parse_bib(BIB_PATH)
    by_doi, by_arxiv, by_ay = build_bib_index(bib_entries)
    print(f"      {len(bib_entries)} entries | doi={len(by_doi)} arxiv={len(by_arxiv)}", flush=True)

    print(f"[2/4] loading ranking from {RANKING_CSV.name} ...", flush=True)
    with RANKING_CSV.open() as f:
        rows = list(csv.DictReader(f))
    # Filter to items with combined_score, sort desc by it
    rows = [r for r in rows if r["combined_score"] not in ("", None)]
    rows.sort(key=lambda r: -float(r["combined_score"]))
    top = rows[: args.top_n]
    print(f"      top {len(top)} items", flush=True)

    print("[3/4] fetching DOIs / arXiv IDs from Zotero ...", flush=True)
    z = zotero.Zotero(USER_ID, "user", API_KEY)
    enriched = []
    for r in top:
        it = z.item(r["item_key"])
        d = it["data"]
        doi, arxiv = extract_item_ids(d)
        creators = d.get("creators") or []
        ln = first_author_lastname_from_creators(creators)
        year = ""
        date = d.get("date") or ""
        m = re.search(r"\b(19\d{2}|20\d{2})\b", date)
        if m:
            year = m.group(1)
        # Check citation status
        status = "NOT CITED"
        match_key = ""
        if doi and doi in by_doi:
            status = "CITED"
            match_key = by_doi[doi]
        elif arxiv and arxiv in by_arxiv:
            status = "CITED"
            match_key = by_arxiv[arxiv]
        elif ln and year and (ln, year) in by_ay:
            status = "CITED"
            match_key = by_ay[(ln, year)]
        note = ""
        if status == "NOT CITED":
            note = short_note(d.get("title") or "", d.get("abstractNote") or "")
        enriched.append(
            {
                "rank": len(enriched) + 1,
                "key": r["item_key"],
                "combined": r["combined_score"],
                "match": r["match_score"],
                "quality": r["quality_score"],
                "year": year or r["year"],
                "venue": (r["venue"] or "")[:50],
                "title": (r["title"] or "")[:140],
                "doi": doi,
                "arxiv": arxiv,
                "status": status,
                "bib_key": match_key,
                "note": note,
            }
        )

    n_cited = sum(1 for e in enriched if e["status"] == "CITED")
    n_not = len(enriched) - n_cited
    print(f"      cited: {n_cited} | not cited: {n_not}", flush=True)

    print(f"[4/4] writing {OUT_MD} ...", flush=True)
    with OUT_MD.open("w") as f:
        f.write(f"# Cite-gap audit: top {len(enriched)} Zotero items by combined score\n\n")
        f.write(
            f"Collection `{COLLECTION_KEY}` -- generated against `{BIB_PATH.name}` ({len(bib_entries)} entries).\n\n"
        )
        f.write(
            f"Of the top {len(enriched)} items: **{n_cited} CITED** | **{n_not} NOT CITED**.\n\n"
        )

        f.write("## Top 50 ranking\n\n")
        f.write("| # | comb | match | qual | yr | venue | title | status | bib key | note |\n")
        f.write("|--:|--:|--:|--:|--:|---|---|---|---|---|\n")
        for e in enriched:
            f.write(
                f"| {e['rank']} | {e['combined']} | {e['match']} | {e['quality']} | "
                f"{e['year']} | {e['venue'].replace('|','/')} | "
                f"{e['title'].replace('|','/')} | {e['status']} | "
                f"{e['bib_key'] or '-'} | {e['note'].replace('|','/')} |\n"
            )

        not_cited = [e for e in enriched if e["status"] == "NOT CITED"]
        f.write(f"\n## Priority list -- {len(not_cited)} NOT-CITED items sorted by combined\n\n")
        f.write("Zotero item key | combined | year | title | suggested-use note\n")
        f.write("--- | --: | --: | --- | ---\n")
        for e in not_cited:
            f.write(
                f"`{e['key']}` | {e['combined']} | {e['year']} | "
                f"{e['title'].replace('|','/')} | {e['note'].replace('|','/')}\n"
            )

        f.write("\n## How to look up a Zotero key\n\n")
        f.write(
            "https://www.zotero.org/groups/_/library/items/<KEY> "
            "or for personal library: open Zotero -> Edit -> Find -> paste key into search.\n"
        )

    # Also write a CSV for easy sorting
    out_csv = OUT_MD.with_suffix(".csv")
    with out_csv.open("w", newline="") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=[
                "rank",
                "key",
                "combined",
                "match",
                "quality",
                "year",
                "venue",
                "title",
                "doi",
                "arxiv",
                "status",
                "bib_key",
                "note",
            ],
        )
        w.writeheader()
        for e in enriched:
            w.writerow(e)
    print(f"      wrote {OUT_MD}", flush=True)
    print(f"      wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
