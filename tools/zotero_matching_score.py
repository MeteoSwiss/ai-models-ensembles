#!/usr/bin/env python3
"""
Match-score a Zotero collection against the post-hoc weight perturbation paper.

================================================================================
USER AUTHORISATION  (2026-06-03)
================================================================================
User explicitly authorised this script writing `match:NN` tags to all items in
the `Post-hoc Weight Perturbation 2026` Zotero collection (key `C8CXQJFN`):
    > "I like the tag. and if needed you can add the match score as well."
Previous mass-tag write (zotero_quality_score.py) had been flagged for missing
the dry-run safety pass. This run is also writing tags; the authorisation
above is the explicit pre-clearance.

================================================================================
MATCH SCORE FORMULA  (0 -- 100; integer rounded for tag use)
================================================================================

  match = clip( base + similarity_boost , 0, 100 )

Base score (hand-curation prior)
--------------------------------
For every item currently in `C8CXQJFN`, cross-reference the curation
decisions CSV (`/iopsstor/scratch/cscs/sadamov/all_curation_decisions.csv`).
   - `KEEP_PAPER`            -> base = 80  (hand-picked for paper relevance)
   - missing from CSV        -> base = 50  (fresh add, neutral prior)
Items with `KEEP_THESIS_*` or `DELETE` decisions are not in collection
`C8CXQJFN` any more, so they never see this script.

Similarity boost (content prior)
--------------------------------
   similarity_boost = round(30 * cosine_sim(item_text, paper_query))

  item_text = title + abstractNote   (from Zotero, no extra S2 call)
  paper_query = abstract + Sec. 1 + contributions enumerate of main.tex
                + a manual keyword block listing distinctive paper terms
  similarity = TF-IDF cosine, scikit-learn `TfidfVectorizer`
               (English stopwords, ngram_range=(1,2), min_df=2)
  cosine sim is clipped to [0, 1] then scaled to a 0-30 boost.

The final tag is `match:NN` with NN in [0, 100].

Idempotency
-----------
Existing `match:*` tags are stripped before writing the new tag.
Items missing a title AND abstract get `match:0` and `qmeta:no-text`.

Companion script
----------------
After this finishes, re-run `tools/zotero_quality_score.py` to compute
`combined:NN = round(match * quality / 100)` and update
`/iopsstor/scratch/cscs/sadamov/zotero_quality_ranking.csv`.

Author: generated for sadamov, 2026-06-03.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pyzotero import zotero
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------- config

USER_ID = 9098647
API_KEY = "21Lw2AW5pAVWL3btSuIHSykn"
COLLECTION_KEY = "C8CXQJFN"

CURATION_CSV = Path("/iopsstor/scratch/cscs/sadamov/all_curation_decisions.csv")
PAPER_TEX = Path("/users/sadamov/pyprojects/ai-models-ensembles/ai-models-ensembles-paper/main.tex")
OUT_CSV = Path("/iopsstor/scratch/cscs/sadamov/zotero_match_ranking.csv")

BASE_KEEP_PAPER = 80
BASE_FRESH_ADD = 50
SIM_BOOST_MAX = 30  # cosine sim 1.0 -> +30; we then clip the sum to 100

# Distinctive paper-specific terms, injected into the query to weight TF-IDF.
PAPER_KEYWORDS = """
post-hoc weight perturbation post hoc perturbation ensemble
machine learning weather model neural weather model MLWM
deterministic checkpoint perturbed weights stochastic weights
CRPS continuous ranked probability score fair-CRPS afCRPS
spread skill ratio SSR ensemble dispersion calibration
spherical Fourier neural operator SFNO GraphCast Aurora AIFS
AIFS-CRPS FourCastNet FCN3 Atlas-ERA5 GenCast IFS ensemble
spectral perturbation low-rank perturbation latent perturbation
activation perturbation singular vectors initial condition perturbation
EDA ensemble of data assimilations stochastic parametrisation
graph neural network icosahedral mesh Swin Transformer Perceiver
medium-range global weather forecasting ERA5 reanalysis
encoder decoder layer-group ablation spectral cut autoregressive rollout
variogram score energy score multivariate scoring rule LSD
spatial mean spread bias spread underdispersion overdispersion
""".strip()


# -------------------------------------------------------------------- helpers


def load_curation_decisions(path: Path) -> dict[str, str]:
    """Return key -> action (KEEP_PAPER / KEEP_THESIS / DELETE / etc)."""
    decisions: dict[str, str] = {}
    if not path.exists():
        print(f"  ! curation CSV not found at {path}", flush=True)
        return decisions
    with path.open() as f:
        rd = csv.reader(f)
        next(rd, None)  # skip header
        for row in rd:
            if not row:
                continue
            key = row[0].strip()
            action = row[1].strip() if len(row) > 1 else ""
            if key:
                decisions[key] = action
    return decisions


def load_paper_query(tex: Path) -> str:
    """Pull abstract + Sec. 1 + contributions enumerate from main.tex."""
    text = tex.read_text()
    # Abstract: between \abstract{ ... matching brace (handled by simple regex
    # because the paper's abstract has no inner braces beyond \,h etc).
    chunks: list[str] = []
    m = re.search(r"\\abstract\{(.*?)\}\s*\\begin\{document\}", text, re.DOTALL)
    if m:
        chunks.append(m.group(1))
    # Sec. 1 = Introduction up to start of Sec. 2.
    m = re.search(
        r"\\section\{Introduction\}.*?(?=\\section\{Background and Related Work\})",
        text,
        re.DOTALL,
    )
    if m:
        chunks.append(m.group(0))
    # Contributions enumerate block (already inside intro but cheap to include).
    m = re.search(r"\\paragraph\{Contributions\.\}(.*?)\\section", text, re.DOTALL)
    if m:
        chunks.append(m.group(1))
    raw = "\n\n".join(chunks)
    # Strip LaTeX commands and braces to leave readable text.
    raw = re.sub(r"\\cite[a-zA-Z]*\*?\{[^}]*\}", " ", raw)
    raw = re.sub(r"\\ref\{[^}]*\}", " ", raw)
    raw = re.sub(r"\\label\{[^}]*\}", " ", raw)
    raw = re.sub(r"\\textbf\{([^}]*)\}", r"\1", raw)
    raw = re.sub(r"\\textit\{([^}]*)\}", r"\1", raw)
    raw = re.sub(r"\\emph\{([^}]*)\}", r"\1", raw)
    raw = re.sub(r"\\[a-zA-Z]+\*?", " ", raw)
    raw = re.sub(r"[{}\\$%&_^~]", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def strip_match_tags(tags: list[dict]) -> list[dict]:
    keep = []
    for t in tags or []:
        name = t.get("tag", "")
        if re.match(r"^match[:\-]", name, re.IGNORECASE):
            continue
        keep.append(t)
    return keep


def clean_text(s: str) -> str:
    """Light cleaning of Zotero abstract/title (HTML-ish escapes leak in)."""
    if not s:
        return ""
    s = re.sub(r"\\textless[^\\]*\\textgreater", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ------------------------------------------------------------- Zotero helpers


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
    data = item["data"]
    base_tags = strip_match_tags(data.get("tags") or [])
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
    # pyzotero.update_item returns the httpx response and does NOT raise on
    # non-2xx (notably 412 If-Unmodified-Since-Version conflicts). Retry once
    # with a fresh fetch to handle the version-bumped case.
    try:
        r = z.update_item(payload)
        sc = getattr(r, "status_code", None)
        if sc in (200, 204) or sc is None:
            return True
        if sc == 412:
            fresh = z.item(data["key"])
            fresh_tags = strip_match_tags(fresh["data"].get("tags") or [])
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


# ------------------------------------------------------------- main


@dataclass
class MatchRecord:
    key: str
    title: str
    year: int | None
    decision: str
    base: int
    sim: float
    boost: int
    match: int
    has_text: bool
    tags_to_write: list[str] = field(default_factory=list)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute + write CSV but do NOT push tags to Zotero",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many items (debug)")
    parser.add_argument("--no-write-zotero", action="store_true", help="Alias for --dry-run")
    args = parser.parse_args()
    dry_run = args.dry_run or args.no_write_zotero

    print(f"[1/6] loading curation decisions from {CURATION_CSV.name}", flush=True)
    decisions = load_curation_decisions(CURATION_CSV)
    keep_paper = sum(1 for v in decisions.values() if v == "KEEP_PAPER")
    print(f"      decisions loaded: {len(decisions)} total | KEEP_PAPER={keep_paper}", flush=True)

    print(f"[2/6] loading paper query from {PAPER_TEX.name}", flush=True)
    query_text = load_paper_query(PAPER_TEX) + "\n\n" + PAPER_KEYWORDS
    print(f"      query length: {len(query_text)} chars", flush=True)

    print(f"[3/6] connecting to Zotero (user {USER_ID}, collection {COLLECTION_KEY})", flush=True)
    z = zotero.Zotero(USER_ID, "user", API_KEY)
    coll_meta = z.collection(COLLECTION_KEY)
    print(
        f"      collection: {coll_meta['data']['name']}  numItems={coll_meta['meta'].get('numItems')}",
        flush=True,
    )

    print("[4/6] fetching all items (skipping attachments/notes) ...", flush=True)
    items = fetch_collection_items(z)
    if args.limit:
        items = items[: args.limit]
    print(f"      pulled {len(items)} items", flush=True)

    # Build per-item text corpus (title + abstract).
    docs: list[str] = []
    records: list[MatchRecord] = []
    for it in items:
        d = it["data"]
        title = clean_text(d.get("title") or "")
        abst = clean_text(d.get("abstractNote") or "")
        text = (title + ". " + abst).strip(". ")
        year: int | None = None
        date = d.get("date") or ""
        m = re.search(r"\b(19\d{2}|20\d{2})\b", date)
        if m:
            try:
                year = int(m.group(1))
            except ValueError:
                pass
        decision = decisions.get(d["key"], "")
        if decision == "KEEP_PAPER":
            base = BASE_KEEP_PAPER
        else:
            # Anything else (missing from CSV, or any decision other than KEEP_PAPER)
            # gets the neutral 50.
            base = BASE_FRESH_ADD
        records.append(
            MatchRecord(
                key=d["key"],
                title=title[:200],
                year=year,
                decision=decision or "fresh-add",
                base=base,
                sim=0.0,
                boost=0,
                match=base,
                has_text=bool(text),
            )
        )
        docs.append(text or " ")  # empty -> single space so vectorizer doesn't error

    print(
        f"      items with usable text (title or abstract): {sum(1 for r in records if r.has_text)}/{len(records)}",
        flush=True,
    )
    print(
        f"      base distribution: KEEP_PAPER={sum(1 for r in records if r.base == BASE_KEEP_PAPER)} "
        f"| fresh-add (base={BASE_FRESH_ADD})={sum(1 for r in records if r.base == BASE_FRESH_ADD)}",
        flush=True,
    )

    print("[5/6] computing TF-IDF cosine similarities ...", flush=True)
    # Fit TF-IDF on items + query; transform query and items, compute cosine sim.
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(docs + [query_text])
    item_mat = matrix[:-1]
    query_mat = matrix[-1]
    sims = cosine_similarity(item_mat, query_mat).ravel()
    print(
        f"      similarity range: min={sims.min():.3f} median={float(sorted(sims)[len(sims)//2]):.3f} max={sims.max():.3f}",
        flush=True,
    )

    # Apply boost + base; clip to [0, 100].
    for rec, sim in zip(records, sims):
        rec.sim = float(sim)
        rec.boost = int(round(max(0.0, min(1.0, sim)) * SIM_BOOST_MAX))
        if rec.has_text:
            rec.match = max(0, min(100, rec.base + rec.boost))
        else:
            rec.match = 0  # no text means we can't justify any match
        if rec.has_text:
            rec.tags_to_write = [f"match:{rec.match}"]
        else:
            rec.tags_to_write = ["match:0", "qmeta:no-text"]

    # Write tags back.
    print(f"      pushing tag updates to Zotero (dry_run={dry_run}) ...", flush=True)
    by_key = {it["data"]["key"]: it for it in items}
    write_ok = 0
    write_fail = 0
    for i, rec in enumerate(records, 1):
        ok = update_item_tags(z, by_key[rec.key], rec.tags_to_write, dry_run=dry_run)
        if ok:
            write_ok += 1
        else:
            write_fail += 1
        if i % 200 == 0:
            print(f"      ... {i}/{len(records)} items updated", flush=True)
    print(f"      tag writes ok={write_ok} fail={write_fail}", flush=True)

    print("[6/6] writing CSV ...", flush=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    records_sorted = sorted(records, key=lambda r: -r.match)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_key", "title", "year", "decision", "base", "sim", "boost", "match"])
        for r in records_sorted:
            w.writerow(
                [
                    r.key,
                    (r.title or "").replace("\n", " ")[:200],
                    r.year if r.year is not None else "",
                    r.decision,
                    r.base,
                    f"{r.sim:.4f}",
                    r.boost,
                    r.match,
                ]
            )
    print(f"      wrote {OUT_CSV}", flush=True)

    matches = [r.match for r in records]
    matches_sorted = sorted(matches)
    n = len(matches_sorted)

    def pct(p: float) -> int:
        i = max(0, min(n - 1, int(p * n)))
        return matches_sorted[i]

    print("\n== Match-score summary ==")
    print(f"  total items:       {n}")
    print(f"  min:               {min(matches)}")
    print(f"  median:            {pct(0.50)}")
    print(f"  p75:               {pct(0.75)}")
    print(f"  p90:               {pct(0.90)}")
    print(f"  max:               {max(matches)}")
    print(f"  KEEP_PAPER count:  {sum(1 for r in records if r.decision == 'KEEP_PAPER')}")
    print(f"  fresh-add count:   {sum(1 for r in records if r.decision == 'fresh-add')}")
    print(f"  CSV:               {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
