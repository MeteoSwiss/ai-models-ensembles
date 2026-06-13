"""Add Bengtsson2007 + Strachan2013 to Zotero collection C8CXQJFN with match:90.

Both are TC-tracking methodology refs adjacent to Walsh2007 + Bourdin2022,
added 2026-06-04 to support the two-level warm-core thickness defence in
the Milton case-study draft.
"""

import os
import ssl
import time

import certifi
from pyzotero import zotero

os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

USER_ID = 9098647
API_KEY = "21Lw2AW5pAVWL3btSuIHSykn"
COLL = "C8CXQJFN"

z = zotero.Zotero(USER_ID, "user", API_KEY)

NEW = {
    "Bengtsson2007": {
        "doi": "10.1111/j.1600-0870.2007.00251.x",
        "title": "How May Tropical Cyclones Change in a Warmer Climate?",
        "journal": "Tellus A: Dynamic Meteorology and Oceanography",
        "year": "2007",
        "vol": "59",
        "num": "4",
        "pages": "539-561",
        "authors": [
            ("Bengtsson", "Lennart"),
            ("Hodges", "Kevin I."),
            ("Esch", "Monika"),
            ("Keenlyside", "Noel"),
            ("Kornblueh", "Luis"),
            ("Luo", "Jing-Jia"),
            ("Yamagata", "Toshio"),
        ],
    },
    "Strachan2013": {
        "doi": "10.1175/JCLI-D-12-00012.1",
        "title": "Investigating Global Tropical Cyclone Activity with a Hierarchy of AGCMs: The Role of Model Resolution",
        "journal": "Journal of Climate",
        "year": "2013",
        "vol": "26",
        "num": "1",
        "pages": "133-152",
        "authors": [
            ("Strachan", "Jane"),
            ("Vidale", "Pier Luigi"),
            ("Hodges", "Kevin"),
            ("Roberts", "Malcolm"),
            ("Demory", "Marie-Estelle"),
        ],
    },
}


def existing_dois() -> set[str]:
    out = set()
    start = 0
    while True:
        batch = z.collection_items_top(COLL, start=start, limit=100)
        if not batch:
            break
        for it in batch:
            d = (it["data"].get("DOI") or "").lower().strip()
            if d:
                out.add(d)
        start += len(batch)
        if len(batch) < 100:
            break
    return out


present = existing_dois()
print(f"Collection has {len(present)} items with DOI present", flush=True)

for bibkey, rec in NEW.items():
    doi = rec["doi"].lower()
    if doi in present:
        print(f"  SKIP {bibkey}: already in collection", flush=True)
        continue
    item = {
        "itemType": "journalArticle",
        "title": rec["title"],
        "creators": [
            {"creatorType": "author", "firstName": fn, "lastName": ln} for ln, fn in rec["authors"]
        ],
        "publicationTitle": rec["journal"],
        "date": rec["year"],
        "volume": rec["vol"],
        "issue": rec["num"],
        "pages": rec["pages"],
        "DOI": rec["doi"],
        "collections": [COLL],
        "tags": [{"tag": "match:90"}, {"tag": "milton-case-study"}, {"tag": "method"}],
    }
    print(f"  Creating {bibkey} ...", flush=True)
    resp = z.create_items([item])
    succ = resp.get("successful", {})
    fail = resp.get("failed", {})
    if succ:
        for k, v in succ.items():
            print(f"    OK -> {v['key']}", flush=True)
    if fail:
        for k, v in fail.items():
            print(f"    FAIL: {v}", flush=True)
    time.sleep(1.0)

print("Done.", flush=True)
