"""Look up Bengtsson2007 and Strachan2013 via Crossref to confirm and emit BibTeX."""

import json
import ssl
import urllib.request

import certifi

CTX = ssl.create_default_context(cafile=certifi.where())

DOIS = {
    "Bengtsson2007tropical": "10.1111/j.1600-0870.2007.00251.x",
    "Strachan2013investigating": "10.1175/JCLI-D-12-00012.1",
}


def fetch(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    with urllib.request.urlopen(url, context=CTX, timeout=30) as r:
        return json.load(r)["message"]


for key, doi in DOIS.items():
    print(f"=== {key}: {doi} ===")
    d = fetch(doi)
    print("TITLE:    ", d.get("title", [None])[0])
    print("CONTAINER:", d.get("container-title", [None])[0])
    print("YEAR:     ", d.get("issued", {}).get("date-parts", [[None]])[0][0])
    print("VOL:      ", d.get("volume"))
    print("ISSUE:    ", d.get("issue"))
    print("PAGES:    ", d.get("page"))
    print("DOI:      ", d.get("DOI"))
    auths = "; ".join(
        a.get("family", "?") + ", " + a.get("given", "?") for a in d.get("author", [])
    )
    print("AUTHORS:  ", auths)
    print()
