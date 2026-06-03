#!/usr/bin/env python3
"""
Quality-score a Zotero collection by cross-referencing Semantic Scholar.

================================================================================
QUALITY SCORE FORMULA  (0 -- 100; integer rounded for tag use)
================================================================================

  quality = 100 * clip( w_cit  * f_cit
                       + w_inf  * f_inf
                       + w_ven  * f_ven
                       + w_rec  * f_rec , 0, 1 )

where each f_x in [0, 1] and the weights default to:

    w_cit = 0.30   citation count, log-scaled
    w_inf = 0.30   influential citation count, log-scaled (S2's filter)
    w_ven = 0.25   venue tier (journal > conference > preprint > unknown)
    w_rec = 0.15   recency / longevity blend

Sub-scores
----------
  f_cit  = log1p(citationCount)        / log1p(CIT_SAT)        clipped to 1
           with CIT_SAT = 1000 (a paper at >=1000 citations saturates this axis)
  f_inf  = log1p(influentialCitationCount) / log1p(INF_SAT)    clipped to 1
           with INF_SAT = 100
  f_ven  = VENUE_TIER[venue_class]
             journal           -> 1.00
             conference        -> 0.80   (peer-reviewed conference)
             workshop          -> 0.55
             preprint (arXiv)  -> 0.40
             book / chapter    -> 0.70
             unknown / other   -> 0.30
  f_rec  = mild trapezoid:
             age <= 1y : 0.70   (too fresh, may not yet have citations)
             1y < age <= 8y : 1.00 (sweet spot)
             8y < age <= 20y : linear decay from 1.00 -> 0.65
             age > 20y AND citationCount < 100 : 0.30 (old + uncited)
             age > 20y AND citationCount >= 100 : 0.75 (genuine classic)

Combined score
--------------
If a matching score (`match:NN` tag, NN in [0,100]) is already present on the
item, we also emit `combined:NN` where combined = round(match * quality / 100).
The matching-score pass on this collection (2026-06-03 audit) had NOT yet
written tags at the time this script was first run -- in that case we still
emit quality scores, leaving combined undefined.

Re-running
----------
The script is idempotent. It clears any existing `quality:*` / `combined:*` /
`qmeta:*` tags on each item before writing the new pair. Items without a DOI
or arXiv ID get `quality:NA` and reason `no-identifier`.

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

# -------------------------------------------------------------------- config

USER_ID = 9098647
API_KEY = "21Lw2AW5pAVWL3btSuIHSykn"
COLLECTION_KEY = "C8CXQJFN"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = (
    "title,year,citationCount,influentialCitationCount,"
    "venue,publicationVenue,publicationTypes,journal,externalIds"
)
S2_BATCH_SIZE = 100
S2_SLEEP_S = 1.1
S2_MAX_RETRIES = 5

OUT_CSV = Path("/iopsstor/scratch/cscs/sadamov/zotero_quality_ranking.csv")
OUT_MD = Path("/iopsstor/scratch/cscs/sadamov/zotero_quality_top50.md")

CIT_SAT = 1000.0
INF_SAT = 100.0

WEIGHTS = {"cit": 0.30, "inf": 0.30, "ven": 0.25, "rec": 0.15}

VENUE_TIER = {
    "journal": 1.00,
    "conference": 0.80,
    "workshop": 0.55,
    "preprint": 0.40,
    "book": 0.70,
    "unknown": 0.30,
}

CURRENT_YEAR = 2026

# -------------------------------------------------------------------- helpers


def log_norm(x: float, sat: float) -> float:
    if x is None or x < 0:
        x = 0.0
    return min(1.0, math.log1p(x) / math.log1p(sat))


def classify_venue(s2: dict) -> str:
    """Return one of journal / conference / workshop / preprint / book / unknown."""
    pv = s2.get("publicationVenue") or {}
    pv_type = (pv.get("type") or "").lower()
    pv_name = (pv.get("name") or "").lower()
    types = [t.lower() for t in (s2.get("publicationTypes") or [])]
    journal = (s2.get("journal") or {}).get("name", "")
    venue_str = (s2.get("venue") or "").lower()

    text = " ".join([pv_type, pv_name, venue_str, " ".join(types), (journal or "").lower()])

    if "arxiv" in text or "preprint" in text or "ssrn" in text or "biorxiv" in text:
        return "preprint"
    if "workshop" in text:
        return "workshop"
    if pv_type == "journal" or "journalarticle" in types or "journal" in pv_type:
        return "journal"
    if pv_type == "conference" or "conference" in pv_type or "conferencepaper" in types:
        return "conference"
    if "book" in types or "bookchapter" in types or "book" in pv_type:
        return "book"
    return "unknown"


def recency_factor(year: int | None, citation_count: int) -> float:
    if not year or year < 1900:
        return 0.5
    age = max(0, CURRENT_YEAR - int(year))
    if age <= 1:
        return 0.70
    if age <= 8:
        return 1.00
    if age <= 20:
        # linear 1.00 -> 0.65 between 8y and 20y
        return 1.00 - (age - 8) * (1.00 - 0.65) / (20 - 8)
    if citation_count >= 100:
        return 0.75
    return 0.30


def quality_score(s2: dict) -> tuple[float, dict[str, float]]:
    cc = int(s2.get("citationCount") or 0)
    ic = int(s2.get("influentialCitationCount") or 0)
    year = s2.get("year")
    venue_class = classify_venue(s2)
    f_cit = log_norm(cc, CIT_SAT)
    f_inf = log_norm(ic, INF_SAT)
    f_ven = VENUE_TIER.get(venue_class, VENUE_TIER["unknown"])
    f_rec = recency_factor(year, cc)
    raw = (
        WEIGHTS["cit"] * f_cit
        + WEIGHTS["inf"] * f_inf
        + WEIGHTS["ven"] * f_ven
        + WEIGHTS["rec"] * f_rec
    )
    score = 100.0 * max(0.0, min(1.0, raw))
    return score, {
        "f_cit": f_cit,
        "f_inf": f_inf,
        "f_ven": f_ven,
        "f_rec": f_rec,
        "venue_class": venue_class,
    }


# -------------------------------------------------------------- ID extraction

ARXIV_RE = re.compile(r"(?:arxiv[:/])?(\d{4}\.\d{4,6})(?:v\d+)?", re.IGNORECASE)
ARXIV_OLD_RE = re.compile(r"([a-z\-]+/\d{7})", re.IGNORECASE)


def extract_arxiv(item_data: dict) -> str | None:
    blobs = [
        item_data.get("url") or "",
        item_data.get("extra") or "",
        item_data.get("DOI") or "",
        item_data.get("archive") or "",
        item_data.get("archiveID") or "",
        item_data.get("publicationTitle") or "",
        item_data.get("libraryCatalog") or "",
    ]
    text = " ".join(blobs)
    m = ARXIV_RE.search(text)
    if m:
        return m.group(1)
    m = ARXIV_OLD_RE.search(text)
    if m:
        return m.group(1)
    return None


def s2_id_for_item(item_data: dict) -> str | None:
    doi = (item_data.get("DOI") or "").strip()
    if doi:
        return f"DOI:{doi}"
    arxiv = extract_arxiv(item_data)
    if arxiv:
        return f"ARXIV:{arxiv}"
    return None


# -------------------------------------------------------- tag bookkeeping


def existing_match_score(tags: list[dict]) -> int | None:
    for t in tags or []:
        m = re.match(r"^match[:\-](\d+)$", t.get("tag", ""), re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            except ValueError:
                pass
    return None


def strip_quality_tags(tags: list[dict]) -> list[dict]:
    keep = []
    for t in tags or []:
        name = t.get("tag", "")
        if re.match(r"^(quality|combined|qmeta)[:\-]", name, re.IGNORECASE):
            continue
        keep.append(t)
    return keep


# ---------------------------------------------------- Semantic Scholar batch


def s2_batch(ids: list[str], session: requests.Session) -> list[dict | None]:
    payload = {"ids": ids}
    for attempt in range(S2_MAX_RETRIES):
        try:
            r = session.post(
                S2_BATCH_URL,
                params={"fields": S2_FIELDS},
                json=payload,
                timeout=90,
            )
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, list):
                    return [None] * len(ids)
                return data  # entries can be None when not found
            if r.status_code in (429, 502, 503, 504):
                wait = 5 * (attempt + 1)
                print(f"  S2 {r.status_code}, backing off {wait}s", flush=True)
                time.sleep(wait)
                continue
            # 4xx other than 429 -> log and return Nones
            print(f"  S2 {r.status_code}: {r.text[:200]}", flush=True)
            return [None] * len(ids)
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  S2 exception {e!r}, backoff {wait}s", flush=True)
            time.sleep(wait)
    return [None] * len(ids)


# ------------------------------------------------------------------ main


@dataclass
class ItemRecord:
    key: str
    title: str
    year: int | None
    venue: str
    venue_class: str
    citation_count: int | None
    influential_cit: int | None
    match_score: int | None
    quality_score: float | None
    combined_score: float | None
    reason: str
    s2_id: str | None
    tags_to_write: list[str] = field(default_factory=list)


def fetch_collection_items(z: zotero.Zotero) -> list[dict]:
    out: list[dict] = []
    start = 0
    while True:
        batch = z.collection_items(
            COLLECTION_KEY,
            limit=100,
            start=start,
            itemType="-attachment || note",
        )
        if not batch:
            break
        out.extend(batch)
        start += len(batch)
        if len(batch) < 100:
            break
    return out


def update_item_tags(z: zotero.Zotero, item: dict, new_tag_names: list[str], dry_run: bool) -> bool:
    """Patch the item version with new tag list (existing quality tags stripped)."""
    data = item["data"]
    base_tags = strip_quality_tags(data.get("tags") or [])
    merged = base_tags + [{"tag": t} for t in new_tag_names]
    # Deduplicate while preserving order
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
    # pyzotero.update_item returns the httpx response and does NOT raise on
    # non-2xx (notably 412 If-Unmodified-Since-Version conflicts). We retry
    # once with a fresh fetch to handle the version-bumped case.
    try:
        r = z.update_item(payload)
        sc = getattr(r, "status_code", None)
        if sc in (200, 204) or sc is None:
            return True
        if sc == 412:
            # Stale version -- re-fetch and retry once.
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute + write CSV/MD but do NOT push tags to Zotero",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many items (debug)")
    parser.add_argument("--no-write-zotero", action="store_true", help="Alias for --dry-run")
    args = parser.parse_args()
    dry_run = args.dry_run or args.no_write_zotero

    print(f"[1/5] connecting to Zotero (user {USER_ID}, collection {COLLECTION_KEY})", flush=True)
    z = zotero.Zotero(USER_ID, "user", API_KEY)
    coll_meta = z.collection(COLLECTION_KEY)
    print(
        f"      collection: {coll_meta['data']['name']}  numItems={coll_meta['meta'].get('numItems')}",
        flush=True,
    )

    print("[2/5] fetching all items (skipping attachments/notes) ...", flush=True)
    items = fetch_collection_items(z)
    if args.limit:
        items = items[: args.limit]
    print(f"      pulled {len(items)} items", flush=True)

    # Build ID list for batched Semantic Scholar calls.
    records: list[ItemRecord] = []
    id_to_records: dict[str, list[int]] = {}
    have_match = 0
    for it in items:
        d = it["data"]
        title = (d.get("title") or "")[:300]
        year_int: int | None = None
        date = d.get("date") or ""
        m = re.search(r"\b(19\d{2}|20\d{2})\b", date)
        if m:
            try:
                year_int = int(m.group(1))
            except ValueError:
                pass
        match = existing_match_score(d.get("tags") or [])
        if match is not None:
            have_match += 1
        sid = s2_id_for_item(d)
        rec = ItemRecord(
            key=d["key"],
            title=title,
            year=year_int,
            venue="",
            venue_class="",
            citation_count=None,
            influential_cit=None,
            match_score=match,
            quality_score=None,
            combined_score=None,
            reason="" if sid else "no-identifier",
            s2_id=sid,
        )
        records.append(rec)
        if sid:
            id_to_records.setdefault(sid, []).append(len(records) - 1)

    print(
        f"      identifier coverage: {sum(1 for r in records if r.s2_id)}/{len(records)} items have DOI or arXiv",
        flush=True,
    )
    print(f"      pre-existing match:* tags: {have_match}", flush=True)

    print(f"[3/5] querying Semantic Scholar in batches of {S2_BATCH_SIZE} ...", flush=True)
    session = requests.Session()
    unique_ids = list(id_to_records.keys())
    s2_results: dict[str, dict] = {}
    n_batches = math.ceil(len(unique_ids) / S2_BATCH_SIZE)
    for bi in range(n_batches):
        chunk = unique_ids[bi * S2_BATCH_SIZE : (bi + 1) * S2_BATCH_SIZE]
        print(f"      batch {bi+1}/{n_batches}  size={len(chunk)}", flush=True)
        out = s2_batch(chunk, session)
        for sid, entry in zip(chunk, out):
            if entry is not None:
                s2_results[sid] = entry
        time.sleep(S2_SLEEP_S)

    print(f"      S2 hits: {len(s2_results)}/{len(unique_ids)}", flush=True)

    print("[4/5] computing quality scores + writing tags ...", flush=True)
    skipped_no_id = 0
    skipped_no_s2 = 0
    scored = 0
    write_ok = 0
    write_fail = 0
    for rec in records:
        if rec.s2_id is None:
            skipped_no_id += 1
            rec.tags_to_write = ["quality:NA", "qmeta:no-identifier"]
            continue
        s2 = s2_results.get(rec.s2_id)
        if s2 is None:
            skipped_no_s2 += 1
            rec.reason = "s2-miss"
            rec.tags_to_write = ["quality:NA", "qmeta:s2-miss"]
            continue
        # Fill record fields from S2.
        rec.citation_count = int(s2.get("citationCount") or 0)
        rec.influential_cit = int(s2.get("influentialCitationCount") or 0)
        if rec.year is None and s2.get("year"):
            try:
                rec.year = int(s2["year"])
            except (TypeError, ValueError):
                pass
        pv = s2.get("publicationVenue") or {}
        rec.venue = pv.get("name") or s2.get("venue") or (s2.get("journal") or {}).get("name") or ""
        q, sub = quality_score(s2)
        rec.venue_class = sub["venue_class"]
        rec.quality_score = q
        scored += 1
        tags = [f"quality:{int(round(q))}", f"qmeta:venue-{sub['venue_class']}"]
        if rec.match_score is not None:
            combined = rec.match_score * q / 100.0
            rec.combined_score = combined
            tags.append(f"combined:{int(round(combined))}")
        rec.tags_to_write = tags

    # Write tags back to Zotero.
    if not dry_run:
        print("      pushing tag updates to Zotero (live) ...", flush=True)
    else:
        print("      dry-run: NOT pushing tags to Zotero", flush=True)
    by_key = {it["data"]["key"]: it for it in items}
    for i, rec in enumerate(records, 1):
        if not rec.tags_to_write:
            continue
        ok = update_item_tags(z, by_key[rec.key], rec.tags_to_write, dry_run=dry_run)
        if ok:
            write_ok += 1
        else:
            write_fail += 1
        if i % 200 == 0:
            print(f"      ... {i}/{len(records)} items updated", flush=True)

    print(f"      tag writes ok={write_ok} fail={write_fail} (dry_run={dry_run})", flush=True)

    print("[5/5] writing outputs ...", flush=True)

    # Sort: combined first (if available), then quality desc.
    def sort_key(r: ItemRecord) -> tuple[float, float]:
        c = r.combined_score if r.combined_score is not None else (r.quality_score or 0.0)
        q = r.quality_score if r.quality_score is not None else 0.0
        return (-c, -q)

    records_sorted = sorted(records, key=sort_key)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "item_key",
                "title",
                "year",
                "venue",
                "venue_class",
                "citation_count",
                "influential_cit",
                "match_score",
                "quality_score",
                "combined_score",
                "reason",
            ]
        )
        for r in records_sorted:
            w.writerow(
                [
                    r.key,
                    (r.title or "").replace("\n", " ")[:300],
                    r.year if r.year is not None else "",
                    (r.venue or "").replace("\n", " ")[:120],
                    r.venue_class,
                    r.citation_count if r.citation_count is not None else "",
                    r.influential_cit if r.influential_cit is not None else "",
                    r.match_score if r.match_score is not None else "",
                    f"{r.quality_score:.1f}" if r.quality_score is not None else "",
                    f"{r.combined_score:.1f}" if r.combined_score is not None else "",
                    r.reason,
                ]
            )
    print(f"      wrote {OUT_CSV}", flush=True)

    # Top-50 markdown
    top = records_sorted[:50]
    with OUT_MD.open("w") as f:
        f.write(f"# Zotero collection {COLLECTION_KEY} -- top 50 by combined / quality\n\n")
        f.write(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(
            f"Total items: {len(records)} | scored: {scored} | no-identifier: {skipped_no_id} | s2-miss: {skipped_no_s2}\n\n"
        )
        f.write("| # | quality | combined | match | yr | cit | inf | venue | title |\n")
        f.write("|--:|--:|--:|--:|--:|--:|--:|---|---|\n")
        for i, r in enumerate(top, 1):
            q = f"{r.quality_score:.0f}" if r.quality_score is not None else "NA"
            c = f"{r.combined_score:.0f}" if r.combined_score is not None else "-"
            m = str(r.match_score) if r.match_score is not None else "-"
            cc = r.citation_count if r.citation_count is not None else "-"
            ic = r.influential_cit if r.influential_cit is not None else "-"
            ven = (r.venue or r.venue_class or "")[:40].replace("|", "/")
            title = (r.title or "").replace("|", "/")[:120]
            yr = r.year if r.year is not None else "-"
            f.write(f"| {i} | {q} | {c} | {m} | {yr} | {cc} | {ic} | {ven} | {title} |\n")
    print(f"      wrote {OUT_MD}", flush=True)

    # Final stats
    print("\n== Summary ==")
    print(f"  total items considered: {len(records)}")
    print(f"  got quality score:      {scored}")
    print(f"  skipped no-identifier:  {skipped_no_id}")
    print(f"  skipped s2-miss:        {skipped_no_s2}")
    print(f"  with existing match:    {have_match}")
    print(f"  CSV:                    {OUT_CSV}")
    print(f"  MD:                     {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
