#!/usr/bin/env python3
"""
Rescue quality scores for Zotero items tagged `qmeta:s2-miss`.

================================================================================
PURPOSE
================================================================================

A prior run of `tools/zotero_quality_score.py` left ~17 Zotero items in
collection `C8CXQJFN` with `quality:NA` + `qmeta:s2-miss` because the Semantic
Scholar batch endpoint returned `null` for their IDs (typically very recent
arXiv preprints; preprints with delayed S2 indexing; mis-shaped DOIs;
non-arXiv DOIs that S2 cannot resolve). This script retries those items
against three alternative metadata sources, in order:

  (a) arXiv API           (title / abstract / authors / category; no citations)
  (b) Semantic Scholar    by `arXiv:<id>` prefix (in case the prior batch
                          posted a different identifier shape)
  (c) OpenAlex            (citation count, host venue, concepts; broader
                          preprint coverage than S2)

When OpenAlex provides only `cited_by_count`, we approximate the S2
`influentialCitationCount` as `ceil(0.2 * cited_by_count)` -- the S2 field
generally tracks a ~15-25% subset of total citations for highly-cited papers.

For each successfully rescued item we write back:
  - `quality:NN` (replacing `quality:NA`)
  - `qmeta:rescued-<source>` (replacing `qmeta:s2-miss`)
  - `qmeta:venue-<class>` (same convention as the main scorer)

Items where ALL three sources fail keep `quality:NA` but the `qmeta:s2-miss`
tag is replaced with `qmeta:no-source`.

The quality formula is imported verbatim from `tools/zotero_quality_score.py`
so the two pipelines stay numerically consistent.

CSV output: /iopsstor/scratch/cscs/sadamov/zotero_quality_rescue.csv

Author: generated for sadamov, 2026-06-03.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from pyzotero import zotero

# Reuse the canonical scorer to keep formulas in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zotero_quality_score import (  # noqa: E402
    USER_ID,
    API_KEY,
    COLLECTION_KEY,
    quality_score,
    strip_quality_tags,
)

# -------------------------------------------------------------------- config

OUT_CSV = Path("/iopsstor/scratch/cscs/sadamov/zotero_quality_rescue.csv")

# Politeness budgets
ARXIV_SLEEP_S = 3.1  # arXiv API: 1 req per 3 s
S2_SLEEP_S = 1.1
OPENALEX_SLEEP_S = 0.11  # OpenAlex tolerates ~10 req/s
HTTP_TIMEOUT = 45

OPENALEX_MAILTO = "kode@mailbox.org"  # polite-pool

# Only treat 10.48550/arXiv.* DOIs as arXiv identifiers; ignore false-positive
# YYYY.NNNNNN substrings from unrelated DOIs.
ARXIV_DOI_RE = re.compile(r"10\.48550/ar[xX]iv\.(\d{4}\.\d{4,6})")
ARXIV_ID_IN_URL_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,6})", re.IGNORECASE)
PURE_ARXIV_RE = re.compile(r"^\s*(\d{4}\.\d{4,6})\s*$")

# Explicit bibkey -> arxiv-id supplement. Most items are found via their stored
# DOI/URL, but the user provided this list to make the mapping auditable.
EXTRA_HINTS = {
    "Pathak2022": "2202.11214",
    "Bi2023": "2211.02556",
    "Lang2024aifs": "2406.01465",
    "Lang2024aifsens": "2412.15832",
    "Bodnar2025": "2405.13063",
    "Bonev2025fcn3": "2507.12144",
    "Andrychowicz2025fgn": "2506.10772",
    "Cohen2025wm3": "2503.22235",
    "Mahesh2024huge": "2408.03100",
    "Sonderby2020": "2003.12140",
    "Selz2023": "2308.07398",
    "Bulte2024": "2410.04483",
    "Hu2022swinvrnn": "2205.13158",
    "Oskarsson2023": "2305.06745",
    "Chen2026atlas": "2601.18111",
    "AlbergoVE2023": "2209.15571",
}

# -------------------------------------------------------------- ID extraction


def extract_arxiv_id(item_data: dict) -> str | None:
    """Return a clean arXiv id (YYMM.NNNNN) if the item has one."""
    doi = (item_data.get("DOI") or "").strip()
    m = ARXIV_DOI_RE.search(doi)
    if m:
        return m.group(1)
    url = item_data.get("url") or ""
    m = ARXIV_ID_IN_URL_RE.search(url)
    if m:
        return m.group(1)
    extra = item_data.get("extra") or ""
    for line in extra.splitlines():
        m = ARXIV_ID_IN_URL_RE.search(line)
        if m:
            return m.group(1)
        m = PURE_ARXIV_RE.match(line)
        if m:
            return m.group(1)
    archid = item_data.get("archiveID") or ""
    m = ARXIV_ID_IN_URL_RE.search(archid)
    if m:
        return m.group(1)
    m = PURE_ARXIV_RE.match(archid)
    if m:
        return m.group(1)
    return None


def extract_clean_doi(item_data: dict) -> str | None:
    """Return a non-arxiv DOI suitable for OpenAlex /works/doi: lookup."""
    doi = (item_data.get("DOI") or "").strip()
    if not doi:
        return None
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.strip().rstrip("/")
    if ARXIV_DOI_RE.search(doi):
        return None  # arXiv DOI; arXiv path will handle this
    if not doi.startswith("10."):
        return None
    return doi


# ---------------------------------------------------- arXiv API (rescue source a)


ARXIV_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def arxiv_lookup(arxiv_id: str, session: requests.Session) -> dict | None:
    """Return {title,year,venue=preprint,category,publishedYear} or None."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        r = session.get(url, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        print(f"  arxiv: exception for {arxiv_id}: {e!r}", flush=True)
        return None
    if r.status_code != 200:
        print(f"  arxiv: HTTP {r.status_code} for {arxiv_id}", flush=True)
        return None
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        print(f"  arxiv: parse error for {arxiv_id}: {e!r}", flush=True)
        return None
    entry = root.find("a:entry", ARXIV_NS)
    if entry is None:
        return None
    title_el = entry.find("a:title", ARXIV_NS)
    published_el = entry.find("a:published", ARXIV_NS)
    cat_el = entry.find("arxiv:primary_category", ARXIV_NS)
    summary_el = entry.find("a:summary", ARXIV_NS)
    year = None
    if published_el is not None and published_el.text:
        m = re.match(r"(\d{4})", published_el.text)
        if m:
            year = int(m.group(1))
    return {
        "title": (title_el.text or "").strip() if title_el is not None else "",
        "year": year,
        "category": cat_el.get("term") if cat_el is not None else None,
        "summary": (summary_el.text or "").strip()[:400] if summary_el is not None else "",
        "venue": "arXiv",
    }


# ------------------------------- S2 retry by arXiv ID (rescue source b)


S2_BY_ID_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}"
S2_FIELDS = "citationCount,influentialCitationCount,venue,publicationVenue,publicationTypes,journal,year,title"


def s2_arxiv_lookup(arxiv_id: str, session: requests.Session) -> dict | None:
    """Best-effort single-shot retry: S2 free tier per-paper endpoint is
    aggressively rate-limited, so we don't burn time on long backoffs --
    OpenAlex is our primary citation source for rescue. One quick retry on
    429 to ride out brief throttle bursts."""
    url = S2_BY_ID_URL.format(aid=arxiv_id)
    for attempt in range(2):
        try:
            r = session.get(url, params={"fields": S2_FIELDS}, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            print(f"  s2-retry: exception for {arxiv_id}: {e!r}", flush=True)
            return None
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        if r.status_code in (429, 502, 503, 504):
            if attempt == 0:
                time.sleep(3)
                continue
            print(f"  s2-retry: HTTP {r.status_code} for {arxiv_id}, giving up", flush=True)
            return None
        return None
    return None


# ------------------------------- OpenAlex (rescue source c)


OPENALEX_DOI_URL = "https://api.openalex.org/works/doi:{doi}"
OPENALEX_ARXIV_URL = "https://api.openalex.org/works"


def _openalex_to_s2_shape(work: dict) -> dict:
    """Translate an OpenAlex work payload to the same dict shape the
    quality_score() function expects from Semantic Scholar."""
    cc = int(work.get("cited_by_count") or 0)
    ic = math.ceil(cc * 0.2)  # rough proxy for S2 influential_citationCount
    year = work.get("publication_year")
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    pv_type = (source.get("type") or "").lower()  # 'journal' / 'repository' / ...
    pv_name = source.get("display_name") or ""
    # Map OpenAlex 'repository' onto S2 'preprint' venue class.
    if pv_type == "repository":
        pv_type_s2 = "Repository"
        # tag venue name with arxiv/preprint where applicable so classify_venue
        # picks up 'preprint'
        if "arxiv" not in (pv_name or "").lower():
            pv_name = pv_name + " preprint"
    elif pv_type == "journal":
        pv_type_s2 = "journal"
    elif pv_type == "conference":
        pv_type_s2 = "conference"
    elif pv_type == "book series" or pv_type == "book":
        pv_type_s2 = "book"
    else:
        pv_type_s2 = pv_type or ""
    return {
        "citationCount": cc,
        "influentialCitationCount": ic,
        "year": year,
        "venue": pv_name,
        "publicationVenue": {"type": pv_type_s2, "name": pv_name},
        "publicationTypes": [],
        "journal": {"name": pv_name} if pv_type == "journal" else {},
    }


def openalex_doi_lookup(doi: str, session: requests.Session) -> dict | None:
    url = OPENALEX_DOI_URL.format(doi=doi)
    try:
        r = session.get(url, params={"mailto": OPENALEX_MAILTO}, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        print(f"  openalex: exception for doi={doi}: {e!r}", flush=True)
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        print(f"  openalex: HTTP {r.status_code} for doi={doi}", flush=True)
        return None
    try:
        return r.json()
    except ValueError:
        return None


def openalex_arxiv_lookup(arxiv_id: str, session: requests.Session) -> dict | None:
    # OpenAlex indexes most arXiv preprints under DOI 10.48550/arXiv.<id>
    return openalex_doi_lookup(f"10.48550/arXiv.{arxiv_id}", session)


def openalex_title_lookup(title: str, year: int | None, session: requests.Session) -> dict | None:
    if not title:
        return None
    # `search` is broader than `filter=title.search`: it does cross-field full-text
    # match with relevance ranking. Empirically required for older arXiv papers
    # whose OpenAlex record uses the journal DOI (not the arXiv DOI).
    params = {
        "search": title[:200],
        "per-page": 5,
        "mailto": OPENALEX_MAILTO,
    }
    try:
        r = session.get(OPENALEX_ARXIV_URL, params=params, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        print(f"  openalex-title: exception: {e!r}", flush=True)
        return None
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except ValueError:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    # Require a non-trivial title overlap to avoid keyword false positives.
    tlow = title.lower()
    candidates = []
    for w in results:
        nm = (w.get("display_name") or "").lower()
        # crude overlap: shared 6+ char tokens, both directions
        toks_a = {t for t in re.split(r"\W+", tlow) if len(t) >= 5}
        toks_b = {t for t in re.split(r"\W+", nm) if len(t) >= 5}
        if not toks_a or not toks_b:
            continue
        overlap = len(toks_a & toks_b) / max(1, min(len(toks_a), len(toks_b)))
        if overlap >= 0.4:
            candidates.append((overlap, w))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    if year:
        for ov, w in candidates:
            if w.get("publication_year") == year:
                return w
    return candidates[0][1]


# -------------------------------------------------------------- tag bookkeeping


def update_item_tags(z: zotero.Zotero, item: dict, new_tag_names: list[str], dry_run: bool) -> bool:
    data = item["data"]
    base_tags = strip_quality_tags(data.get("tags") or [])
    merged = base_tags + [{"tag": t} for t in new_tag_names]
    seen = set()
    dedup = []
    for t in merged:
        if t["tag"] in seen:
            continue
        seen.add(t["tag"])
        dedup.append(t)
    if dry_run:
        return True
    payload = {"key": data["key"], "version": data["version"], "tags": dedup}
    try:
        r = z.update_item(payload)
        sc = getattr(r, "status_code", None)
        if sc in (200, 204) or sc is None:
            return True
        if sc == 412:
            fresh = z.item(data["key"])
            fresh_tags = strip_quality_tags(fresh["data"].get("tags") or [])
            merged2 = fresh_tags + [{"tag": t} for t in new_tag_names]
            seen2 = set()
            dedup2 = []
            for t in merged2:
                if t["tag"] in seen2:
                    continue
                seen2.add(t["tag"])
                dedup2.append(t)
            payload2 = {"key": data["key"], "version": fresh["data"]["version"], "tags": dedup2}
            r2 = z.update_item(payload2)
            sc2 = getattr(r2, "status_code", None)
            if sc2 in (200, 204) or sc2 is None:
                return True
            print(f"  ! retry update HTTP {sc2} for {data['key']}", flush=True)
            return False
        print(f"  ! update HTTP {sc} for {data['key']}", flush=True)
        return False
    except Exception as e:
        print(f"  ! update failed for {data['key']}: {e}", flush=True)
        return False


# ---------------------------------------------------------- per-item rescue


@dataclass
class RescueResult:
    item_key: str
    bibkey: str | None
    arxiv_id: str | None
    doi: str | None
    title: str
    old_quality: str
    source: str  # arxiv / s2-retry / openalex / openalex-title / none
    citation_count: int | None = None
    influential_count: int | None = None
    venue: str = ""
    venue_class: str = ""
    new_quality: int | None = None
    sources_tried: list[str] = field(default_factory=list)
    multi_source_disagreement: dict = field(default_factory=dict)


def rescue_one(rec: dict, session: requests.Session) -> RescueResult:
    """Try sources in order, return RescueResult with the first hit that
    yields a citation count. arXiv-only hits are kept as a fallback for venue
    classification when no citation source resolves."""
    data = rec["data"]
    arxiv_id = extract_arxiv_id(data)
    doi = extract_clean_doi(data)
    title = (data.get("title") or "").strip()
    year_str = (data.get("date") or "")[:4]
    year_int = int(year_str) if year_str.isdigit() else None

    result = RescueResult(
        item_key=data["key"],
        bibkey=None,
        arxiv_id=arxiv_id,
        doi=doi,
        title=title[:200],
        old_quality="quality:NA",
        source="none",
    )

    # Source (a): arXiv API for metadata (no citations)
    arxiv_meta = None
    if arxiv_id:
        result.sources_tried.append("arxiv")
        arxiv_meta = arxiv_lookup(arxiv_id, session)
        time.sleep(ARXIV_SLEEP_S)

    # Source (b): S2 retry by arXiv ID
    s2_payload = None
    if arxiv_id:
        result.sources_tried.append("s2-retry")
        s2_payload = s2_arxiv_lookup(arxiv_id, session)
        time.sleep(S2_SLEEP_S)

    # Source (c): OpenAlex by arXiv-DOI, then by clean DOI, then title
    oa_work = None
    if arxiv_id:
        result.sources_tried.append("openalex-arxiv")
        oa_work = openalex_arxiv_lookup(arxiv_id, session)
        time.sleep(OPENALEX_SLEEP_S)
    if oa_work is None and doi:
        result.sources_tried.append("openalex-doi")
        oa_work = openalex_doi_lookup(doi, session)
        time.sleep(OPENALEX_SLEEP_S)
    if oa_work is None and title:
        result.sources_tried.append("openalex-title")
        oa_work = openalex_title_lookup(title, year_int, session)
        time.sleep(OPENALEX_SLEEP_S)

    # Pick winner: prefer S2 (familiar venue classifier), else OpenAlex shape.
    winner_payload = None
    winner_label = "none"

    if s2_payload and (s2_payload.get("citationCount") is not None):
        winner_payload = s2_payload
        winner_label = "s2-retry"
    elif oa_work:
        winner_payload = _openalex_to_s2_shape(oa_work)
        winner_label = "openalex"
    elif arxiv_meta:
        # arXiv-only: synthesize a payload with 0 citations + preprint venue.
        # Quality will be low (recency * preprint weight only) but score is
        # better-defined than 'NA'.
        winner_payload = {
            "citationCount": 0,
            "influentialCitationCount": 0,
            "year": arxiv_meta.get("year") or year_int,
            "venue": "arXiv",
            "publicationVenue": {"type": "preprint", "name": "arXiv"},
            "publicationTypes": [],
            "journal": {},
        }
        winner_label = "arxiv"

    if winner_payload is None:
        return result

    if not winner_payload.get("year") and year_int:
        winner_payload["year"] = year_int

    q, sub = quality_score(winner_payload)
    result.source = winner_label
    result.citation_count = int(winner_payload.get("citationCount") or 0)
    result.influential_count = int(winner_payload.get("influentialCitationCount") or 0)
    result.venue = winner_payload.get("venue") or ""
    result.venue_class = sub["venue_class"]
    result.new_quality = int(round(q))

    # Cross-source citation sanity check (s2 vs openalex)
    if s2_payload and oa_work:
        s2_cc = int(s2_payload.get("citationCount") or 0)
        oa_cc = int(oa_work.get("cited_by_count") or 0)
        if max(s2_cc, oa_cc) >= 50 and abs(s2_cc - oa_cc) / max(1, max(s2_cc, oa_cc)) > 0.5:
            result.multi_source_disagreement = {"s2": s2_cc, "openalex": oa_cc}
    return result


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="compute scores + write CSV but do NOT push tags"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--include-extra-bibkeys",
        action="store_true",
        help="also try the explicit-bibkey list (not just qmeta:s2-miss tag)",
    )
    args = parser.parse_args()

    print(f"[1/4] connecting to Zotero (user {USER_ID}, collection {COLLECTION_KEY})", flush=True)
    z = zotero.Zotero(USER_ID, "user", API_KEY)
    coll_meta = z.collection(COLLECTION_KEY)
    print(
        f"      collection: {coll_meta['data']['name']}  numItems={coll_meta['meta'].get('numItems')}",
        flush=True,
    )

    print("[2/4] fetching qmeta:s2-miss items ...", flush=True)
    items = z.everything(z.collection_items(COLLECTION_KEY, tag="qmeta:s2-miss"))
    print(f"      pulled {len(items)} s2-miss items", flush=True)

    if args.include_extra_bibkeys:
        # Augment with explicit bibkey hints: search collection for items whose
        # arXiv ID matches one in EXTRA_HINTS but lack a quality score.
        print("      augmenting with EXTRA_HINTS bibkey list ...", flush=True)
        already = {it["data"]["key"] for it in items}
        all_items = z.everything(z.collection_items(COLLECTION_KEY, itemType="-attachment || note"))
        hint_axids = set(EXTRA_HINTS.values())
        for it in all_items:
            d = it["data"]
            if d["key"] in already:
                continue
            tag_names = {t.get("tag", "") for t in d.get("tags") or []}
            has_quality = any(t.startswith("quality:") and t != "quality:NA" for t in tag_names)
            if has_quality:
                continue
            aid = extract_arxiv_id(d)
            if aid and aid in hint_axids:
                items.append(it)
                already.add(d["key"])
                print(f"        + added {d['key']} (arxiv {aid})", flush=True)

    if args.limit:
        items = items[: args.limit]

    print(f"[3/4] running rescue cascade on {len(items)} items ...", flush=True)
    # bibkey assignment from EXTRA_HINTS reverse-lookup (best-effort)
    axid_to_bibkey = {v: k for k, v in EXTRA_HINTS.items()}
    session = requests.Session()
    session.headers["User-Agent"] = "zotero-quality-rescue/1.0 (kode@mailbox.org)"
    results: list[RescueResult] = []
    by_key = {it["data"]["key"]: it for it in items}
    for i, it in enumerate(items, 1):
        d = it["data"]
        print(f"  [{i}/{len(items)}] {d['key']}  | {(d.get('title') or '')[:70]}", flush=True)
        res = rescue_one(it, session)
        if res.arxiv_id and res.arxiv_id in axid_to_bibkey:
            res.bibkey = axid_to_bibkey[res.arxiv_id]
        # decide tag set
        if res.new_quality is None:
            res.source = "none"
            tags = ["quality:NA", "qmeta:no-source"]
        else:
            tags = [
                f"quality:{res.new_quality}",
                f"qmeta:rescued-{res.source}",
                f"qmeta:venue-{res.venue_class}",
            ]
        ok = update_item_tags(z, by_key[d["key"]], tags, dry_run=args.dry_run)
        if not ok:
            print(f"  ! tag write failed for {d['key']}", flush=True)
        results.append(res)
        print(
            f"      -> source={res.source} cit={res.citation_count} venue={res.venue_class} quality={res.new_quality}",
            flush=True,
        )

    print(f"[4/4] writing CSV to {OUT_CSV}", flush=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bibkey",
                "item_key",
                "arxiv_id",
                "doi",
                "source",
                "citationCount",
                "influentialCount",
                "venue",
                "venue_class",
                "old_quality",
                "new_quality",
                "sources_tried",
                "multi_source_disagreement",
                "title",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.bibkey or "",
                    r.item_key,
                    r.arxiv_id or "",
                    r.doi or "",
                    r.source,
                    r.citation_count if r.citation_count is not None else "",
                    r.influential_count if r.influential_count is not None else "",
                    (r.venue or "").replace("\n", " ")[:120],
                    r.venue_class,
                    r.old_quality,
                    r.new_quality if r.new_quality is not None else "",
                    "|".join(r.sources_tried),
                    ";".join(f"{k}={v}" for k, v in r.multi_source_disagreement.items()),
                    (r.title or "").replace("\n", " ")[:200],
                ]
            )

    # Summary
    n_rescued = sum(1 for r in results if r.new_quality is not None)
    n_failed = len(results) - n_rescued
    scores = sorted(r.new_quality for r in results if r.new_quality is not None)
    if scores:
        p50 = scores[len(scores) // 2]
        p75 = scores[int(len(scores) * 0.75)]
        print("\n== Summary ==")
        print(f"  total tried:     {len(results)}")
        print(f"  rescued:         {n_rescued}")
        print(f"  still NA:        {n_failed}")
        print(f"  quality min/med/p75/max: {scores[0]}/{p50}/{p75}/{scores[-1]}")
    else:
        print("\n== Summary ==")
        print(f"  total tried: {len(results)}  rescued: 0")
    disag = [r for r in results if r.multi_source_disagreement]
    if disag:
        print(f"  multi-source disagreement (>50% gap, >=50 cit): {len(disag)} items")
        for r in disag:
            print(f"    {r.item_key} {r.bibkey or '-'}: {r.multi_source_disagreement}")
    print(f"  CSV: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
