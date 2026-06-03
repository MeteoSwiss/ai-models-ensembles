#!/usr/bin/env python3
"""
Back-fill the paper's Zotero collection (C8CXQJFN) with entries that exist in
`ai-models-ensembles-paper/paper_refs.bib` but are NOT yet in Zotero.

Workflow:
1. Parse paper_refs.bib for bibkeys, DOIs, arXiv IDs, fallback metadata.
2. Pull the existing C8CXQJFN collection, build sets of DOIs + arXiv IDs.
3. For each bib entry not already present:
   - DOI -> Crossref lookup, build Zotero `journalArticle` item.
   - arXiv only -> build Zotero `preprint` item from bib metadata.
   - Neither -> Crossref bibliographic query, else manual entry from bib.
4. Batch-create items (groups of 50) in the C8CXQJFN collection.
5. Tag each new item with `match:90` and `qmeta:backfill-2026-06-03`.
6. Write log CSV at /iopsstor/scratch/cscs/sadamov/zotero_backfill_log.csv.

Read-only against paper_refs.bib; writes Zotero + log CSV only.
"""

from __future__ import annotations

import csv
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
from pyzotero import zotero

USER_ID = 9098647
API_KEY = "21Lw2AW5pAVWL3btSuIHSykn"
COLLECTION_KEY = "C8CXQJFN"

BIB_PATH = Path(
    "/users/sadamov/pyprojects/ai-models-ensembles/ai-models-ensembles-paper/paper_refs.bib"
)
LOG_CSV = Path("/iopsstor/scratch/cscs/sadamov/zotero_backfill_log.csv")

BACKFILL_TAGS = [{"tag": "match:90"}, {"tag": "qmeta:backfill-2026-06-03"}]

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,6})(?:v\d+)?", re.IGNORECASE)
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
USER_AGENT = "zotero-backfill/1.0 (mailto:kode@mailbox.org)"


def parse_bib(path: Path) -> list[dict]:
    text = path.read_text()
    entries: list[dict] = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.DOTALL):
        typ, key, body = m.group(1), m.group(2).strip(), m.group(3)
        rec = {
            "type": typ,
            "key": key,
            "doi": "",
            "arxiv": "",
            "author": "",
            "year": "",
            "title": "",
            "journal": "",
            "booktitle": "",
            "volume": "",
            "pages": "",
            "note": "",
            "organization": "",
        }
        for fld in (
            "doi",
            "title",
            "author",
            "year",
            "journal",
            "booktitle",
            "volume",
            "pages",
            "note",
            "organization",
        ):
            mm = re.search(rf"{fld}\s*=\s*\{{(.+?)\}},?\s*\n", body, re.IGNORECASE | re.DOTALL)
            if mm:
                rec[fld] = re.sub(r"\s+", " ", mm.group(1)).strip()
        mm = re.search(r"eprint\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if mm:
            rec["arxiv"] = mm.group(1).strip().lower()
        rec["doi"] = rec["doi"].lower().rstrip(".")
        entries.append(rec)
    return entries


def normalise_arxiv(s: str) -> str:
    if not s:
        return ""
    m = ARXIV_RE.search(s)
    return m.group(1).lower() if m else ""


def pull_existing(z: zotero.Zotero) -> tuple[set[str], set[str]]:
    items = z.everything(z.collection_items(COLLECTION_KEY))
    dois: set[str] = set()
    arx: set[str] = set()
    for it in items:
        data = it.get("data", {})
        if data.get("itemType") == "attachment":
            continue
        doi = (data.get("DOI") or "").strip().lower().rstrip(".")
        if doi:
            dois.add(doi)
        for src in (data.get("extra", ""), data.get("url", ""), data.get("archiveID", "")):
            a = normalise_arxiv(src or "")
            if a:
                arx.add(a)
    return dois, arx


def split_authors_bib(author_str: str) -> list[dict]:
    """Convert bibtex author field to Zotero creators list."""
    if not author_str:
        return []
    out = []
    for raw in re.split(r"\s+and\s+", author_str):
        raw = raw.strip().strip("{}")
        if not raw or raw.lower() == "others":
            out.append({"creatorType": "author", "name": "others"})
            continue
        if "," in raw:
            last, first = raw.split(",", 1)
            out.append(
                {
                    "creatorType": "author",
                    "firstName": first.strip().strip("{}"),
                    "lastName": last.strip().strip("{}"),
                }
            )
        else:
            parts = raw.rsplit(" ", 1)
            if len(parts) == 2:
                out.append(
                    {
                        "creatorType": "author",
                        "firstName": parts[0].strip().strip("{}"),
                        "lastName": parts[1].strip().strip("{}"),
                    }
                )
            else:
                out.append({"creatorType": "author", "name": raw})
    return out


def strip_braces(s: str) -> str:
    return re.sub(r"[{}]", "", s or "").strip()


def crossref_by_doi(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as r:
            return json.load(r).get("message")
    except Exception as e:
        print(f"  crossref DOI lookup failed for {doi}: {e}", file=sys.stderr)
        return None


def crossref_query(query: str, rows: int = 3) -> list[dict]:
    q = urllib.parse.urlencode({"query.bibliographic": query, "rows": rows})
    url = f"https://api.crossref.org/works?{q}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as r:
            return json.load(r).get("message", {}).get("items", [])
    except Exception as e:
        print(f"  crossref query failed for {query!r}: {e}", file=sys.stderr)
        return []


def item_from_crossref(msg: dict) -> dict:
    """Build a Zotero journalArticle item dict from a Crossref message."""
    item = {
        "itemType": "journalArticle",
        "title": strip_braces((msg.get("title") or [""])[0]),
        "creators": [
            {
                "creatorType": "author",
                "firstName": a.get("given", ""),
                "lastName": a.get("family", a.get("name", "")),
            }
            for a in msg.get("author", [])
        ],
        "publicationTitle": (msg.get("container-title") or [""])[0],
        "volume": msg.get("volume", ""),
        "issue": msg.get("issue", ""),
        "pages": msg.get("page", ""),
        "date": "",
        "DOI": (msg.get("DOI") or "").lower(),
        "url": msg.get("URL", ""),
        "abstractNote": strip_braces(msg.get("abstract", "") or ""),
        "tags": list(BACKFILL_TAGS),
        "collections": [COLLECTION_KEY],
    }
    issued = msg.get("issued", {}).get("date-parts", [[]])
    if issued and issued[0]:
        item["date"] = "-".join(str(p) for p in issued[0])
    typ = msg.get("type", "")
    if typ in ("book-chapter", "proceedings-article"):
        item["itemType"] = "conferencePaper"
        item["proceedingsTitle"] = item.pop("publicationTitle", "")
    elif typ == "book":
        item["itemType"] = "book"
        item.pop("publicationTitle", None)
    elif typ == "report":
        item["itemType"] = "report"
        item.pop("publicationTitle", None)
    return item


def item_from_arxiv_bib(rec: dict) -> dict:
    """Build a Zotero preprint item from bib metadata when only arXiv is available."""
    title = strip_braces(rec.get("title", ""))
    return {
        "itemType": "preprint",
        "title": title,
        "creators": split_authors_bib(rec.get("author", "")),
        "date": rec.get("year", ""),
        "repository": "arXiv",
        "archiveID": f"arXiv:{rec['arxiv']}",
        "url": f"https://arxiv.org/abs/{rec['arxiv']}",
        "abstractNote": "",
        "tags": list(BACKFILL_TAGS),
        "collections": [COLLECTION_KEY],
    }


def item_from_bib_fallback(rec: dict) -> dict:
    """Manual entry from bib metadata when no DOI / arXiv path works."""
    title = strip_braces(rec.get("title", ""))
    typ = rec["type"].lower()
    if typ == "manual":
        return {
            "itemType": "report",
            "title": title,
            "creators": split_authors_bib(rec.get("author", "")),
            "date": rec.get("year", ""),
            "institution": rec.get("organization", ""),
            "abstractNote": "",
            "tags": list(BACKFILL_TAGS),
            "collections": [COLLECTION_KEY],
        }
    if typ == "inproceedings":
        return {
            "itemType": "conferencePaper",
            "title": title,
            "creators": split_authors_bib(rec.get("author", "")),
            "date": rec.get("year", ""),
            "proceedingsTitle": strip_braces(rec.get("booktitle", "")),
            "pages": rec.get("pages", ""),
            "abstractNote": rec.get("note", ""),
            "tags": list(BACKFILL_TAGS),
            "collections": [COLLECTION_KEY],
        }
    # default: journalArticle
    return {
        "itemType": "journalArticle",
        "title": title,
        "creators": split_authors_bib(rec.get("author", "")),
        "publicationTitle": strip_braces(rec.get("journal", "")),
        "volume": rec.get("volume", ""),
        "pages": rec.get("pages", ""),
        "date": rec.get("year", ""),
        "abstractNote": rec.get("note", ""),
        "tags": list(BACKFILL_TAGS),
        "collections": [COLLECTION_KEY],
    }


def resolve_item(rec: dict, existing_dois: set, existing_arx: set) -> tuple[str, dict | None, str]:
    """Returns (action_hint, item_payload, notes).
    action_hint in {dup-doi, dup-arxiv, doi, arxiv-only, crossref-search, manual}."""
    doi = rec.get("doi", "").strip().lower().rstrip(".")
    arx = normalise_arxiv(rec.get("arxiv", ""))
    if doi and doi in existing_dois:
        return ("dup-doi", None, f"DOI {doi} already in collection")
    if arx and arx in existing_arx:
        return ("dup-arxiv", None, f"arXiv {arx} already in collection")
    if doi:
        msg = crossref_by_doi(doi)
        time.sleep(1.0)
        if msg:
            it = item_from_crossref(msg)
            # ensure DOI preserved even if crossref lowered the case
            if not it.get("DOI"):
                it["DOI"] = doi
            return ("doi", it, f"crossref hit for DOI {doi}")
        # crossref miss but we have a DOI -> manual with DOI annotation
        it = item_from_bib_fallback(rec)
        it["DOI"] = doi
        return ("manual", it, f"crossref miss for DOI {doi}, manual fallback")
    if arx:
        return ("arxiv-only", item_from_arxiv_bib(rec), f"arXiv-only entry {arx}")
    # no identifier: try Crossref bibliographic search
    # WARNING: be STRICT here. Crossref search routinely returns EGU abstracts
    # and unrelated short notes with high token overlap. Require:
    #   - title token overlap >= 0.8 (was 0.5; too loose)
    #   - first-author surname match
    #   - year match (within +/- 1 if bib year is uncertain)
    #   - publication type != "proceedings-article" unless bib is @inproceedings
    title = strip_braces(rec.get("title", ""))
    author = strip_braces(rec.get("author", "").split(" and ")[0])
    bib_year = rec.get("year", "").strip()
    bib_type = rec["type"].lower()
    if title:
        query = " ".join(x for x in (author, bib_year, title) if x)
        hits = crossref_query(query, rows=5)
        time.sleep(1.0)
        bib_last = re.findall(r"\w+", author)[-1].lower() if author else ""
        bib_toks = set(re.findall(r"\w+", title.lower()))
        for h in hits:
            ht = (h.get("title") or [""])[0].lower()
            htoks = set(re.findall(r"\w+", ht))
            overlap = len(bib_toks & htoks) / max(len(bib_toks), 1)
            if overlap < 0.8:
                continue
            hit_year = ""
            issued = h.get("issued", {}).get("date-parts", [[]])
            if issued and issued[0]:
                hit_year = str(issued[0][0])
            if bib_year and hit_year and abs(int(bib_year) - int(hit_year)) > 1:
                continue
            authors = h.get("author", [])
            hit_last = (authors[0].get("family", "") if authors else "").lower()
            if bib_last and hit_last and bib_last != hit_last:
                continue
            htype = h.get("type", "")
            if htype == "proceedings-article" and bib_type != "inproceedings":
                continue
            it = item_from_crossref(h)
            return (
                "crossref-search",
                it,
                f"crossref search hit DOI={it.get('DOI','')} (strict match)",
            )
    # last resort: manual entry from bib
    return ("manual", item_from_bib_fallback(rec), "no DOI/arXiv/Crossref hit; manual entry")


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    z = zotero.Zotero(USER_ID, "user", API_KEY)
    print(f"Parsing {BIB_PATH}...", file=sys.stderr)
    bib = parse_bib(BIB_PATH)
    print(f"  {len(bib)} entries in bib", file=sys.stderr)

    print("Pulling existing collection state...", file=sys.stderr)
    existing_dois, existing_arx = pull_existing(z)
    print(
        f"  {len(existing_dois)} DOIs, {len(existing_arx)} arXiv IDs already in C8CXQJFN",
        file=sys.stderr,
    )

    pre_count = next(c for c in z.collections() if c["key"] == COLLECTION_KEY)["meta"]["numItems"]

    log_rows: list[dict] = []
    to_create: list[tuple[dict, dict]] = []  # (bib_rec, zotero_item)

    for rec in bib:
        action, item, notes = resolve_item(rec, existing_dois, existing_arx)
        print(f"  {rec['key']}: {action} - {notes}", file=sys.stderr)
        if action in ("dup-doi", "dup-arxiv"):
            log_rows.append(
                {
                    "bibkey": rec["key"],
                    "doi_or_arxiv": rec.get("doi") or rec.get("arxiv"),
                    "action": "skipped-duplicate",
                    "zotero_key": "",
                    "notes": notes,
                }
            )
            continue
        if item is None:
            log_rows.append(
                {
                    "bibkey": rec["key"],
                    "doi_or_arxiv": rec.get("doi") or rec.get("arxiv"),
                    "action": "failed",
                    "zotero_key": "",
                    "notes": notes,
                }
            )
            continue
        to_create.append((rec, item))

    # batch-create
    created_keys: list[str] = []
    for chunk in batched(to_create, 50):
        payload = [it for _, it in chunk]
        try:
            resp = z.create_items(payload)
        except Exception as e:
            print(f"  batch create failed: {e}", file=sys.stderr)
            for rec, _ in chunk:
                log_rows.append(
                    {
                        "bibkey": rec["key"],
                        "doi_or_arxiv": rec.get("doi") or rec.get("arxiv"),
                        "action": "failed",
                        "zotero_key": "",
                        "notes": f"create_items exception: {e}",
                    }
                )
            time.sleep(1.0)
            continue
        successful = resp.get("successful", {})
        failed = resp.get("failed", {})
        # map index -> recorded result
        for idx_str, item_obj in successful.items():
            idx = int(idx_str)
            rec, _ = chunk[idx]
            key = item_obj.get("key", "")
            created_keys.append(key)
            doi_val = (item_obj.get("data", {}).get("DOI") or "").lower()
            arx_val = ""
            for src in (
                item_obj.get("data", {}).get("extra", ""),
                item_obj.get("data", {}).get("archiveID", ""),
                item_obj.get("data", {}).get("url", ""),
            ):
                a = normalise_arxiv(src or "")
                if a:
                    arx_val = a
                    break
            log_rows.append(
                {
                    "bibkey": rec["key"],
                    "doi_or_arxiv": doi_val or arx_val or (rec.get("doi") or rec.get("arxiv")),
                    "action": "added",
                    "zotero_key": key,
                    "notes": "via batch create",
                }
            )
        for idx_str, err in failed.items():
            idx = int(idx_str)
            rec, _ = chunk[idx]
            log_rows.append(
                {
                    "bibkey": rec["key"],
                    "doi_or_arxiv": rec.get("doi") or rec.get("arxiv"),
                    "action": "failed",
                    "zotero_key": "",
                    "notes": f"create failed: {err}",
                }
            )
        time.sleep(1.0)

    # write log
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LOG_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["bibkey", "doi_or_arxiv", "action", "zotero_key", "notes"]
        )
        w.writeheader()
        w.writerows(log_rows)
    print(f"\nLog written: {LOG_CSV}", file=sys.stderr)

    post_count = next(c for c in z.collections() if c["key"] == COLLECTION_KEY)["meta"]["numItems"]
    print(
        f"Collection size: {pre_count} -> {post_count} (added {len(created_keys)})", file=sys.stderr
    )


if __name__ == "__main__":
    main()
