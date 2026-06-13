"""Append Milton case-study methodology bibliography to paper_refs.bib AND
add the same papers to the Zotero 'Post-hoc Weight Perturbation 2026'
collection (C8CXQJFN). One source of truth, two locations.

Idempotent: skips entries already present (by key) in paper_refs.bib and
Zotero items already containing the DOI.
"""

from __future__ import annotations
from pathlib import Path
import sys

from pyzotero import zotero

BIB = Path("/users/sadamov/pyprojects/ai-models-ensembles/ai-models-ensembles-paper/paper_refs.bib")
ZOT_USER = "9098647"
ZOT_KEY = "21Lw2AW5pAVWL3btSuIHSykn"
ZOT_COLLECTION = "C8CXQJFN"

# Each entry: dict with bib_key, BibTeX entry, and Zotero metadata (DOI + title + authors + year + venue)
REFS = [
    {
        "bib_key": "UllrichZarzycki2017",
        "bibtex": """@article{UllrichZarzycki2017,
  author  = {Ullrich, P. A. and Zarzycki, C. M.},
  title   = {{TempestExtremes}: a Framework for Scale-Insensitive Pointwise Feature Tracking on Unstructured Grids},
  journal = {Geoscientific Model Development},
  year    = {2017},
  volume  = {10},
  pages   = {1069--1090},
  doi     = {10.5194/gmd-10-1069-2017}
}
""",
        "doi": "10.5194/gmd-10-1069-2017",
    },
    {
        "bib_key": "Ullrich2021tempestv2",
        "bibtex": """@article{Ullrich2021tempestv2,
  author  = {Ullrich, P. A. and Zarzycki, C. M. and McClenny, E. E. and Pinheiro, M. C. and Stansfield, A. M. and Reed, K. A.},
  title   = {{TempestExtremes} v2.1: a Community Framework for Feature Detection, Tracking, and Analysis in Large Datasets},
  journal = {Geoscientific Model Development},
  year    = {2021},
  volume  = {14},
  pages   = {5023--5048},
  doi     = {10.5194/gmd-14-5023-2021}
}
""",
        "doi": "10.5194/gmd-14-5023-2021",
    },
    {
        "bib_key": "Walsh2007",
        "bibtex": """@article{Walsh2007,
  author  = {Walsh, K. J. E. and Fiorino, M. and Landsea, C. W. and McInnes, K. L.},
  title   = {Objectively Determined Resolution-Dependent Threshold Criteria for the Detection of Tropical Cyclones in Climate Models and Reanalyses},
  journal = {Journal of Climate},
  year    = {2007},
  volume  = {20},
  pages   = {2307--2314},
  doi     = {10.1175/JCLI4074.1}
}
""",
        "doi": "10.1175/JCLI4074.1",
    },
    {
        "bib_key": "Bourdin2022",
        "bibtex": """@article{Bourdin2022,
  author  = {Bourdin, S. and Fromang, S. and Dulac, W. and Cattiaux, J. and Chauvin, F.},
  title   = {Intercomparison of Four Algorithms for Detecting Tropical Cyclones Using {ERA5}},
  journal = {Geoscientific Model Development},
  year    = {2022},
  volume  = {15},
  pages   = {6759--6786},
  doi     = {10.5194/gmd-15-6759-2022}
}
""",
        "doi": "10.5194/gmd-15-6759-2022",
    },
    {
        "bib_key": "Knapp2010ibtracs",
        "bibtex": """@article{Knapp2010ibtracs,
  author  = {Knapp, K. R. and Kruk, M. C. and Levinson, D. H. and Diamond, H. J. and Neumann, C. J.},
  title   = {The International Best Track Archive for Climate Stewardship ({IBTrACS}): Unifying Tropical Cyclone Data},
  journal = {Bulletin of the American Meteorological Society},
  year    = {2010},
  volume  = {91},
  pages   = {363--376},
  doi     = {10.1175/2009BAMS2755.1}
}
""",
        "doi": "10.1175/2009BAMS2755.1",
    },
    {
        "bib_key": "MarksHouze1987",
        "bibtex": """@article{MarksHouze1987,
  author  = {Marks, F. D. and Houze, R. A.},
  title   = {Inner Core Structure of {H}urricane {A}licia from Airborne {D}oppler Radar Observations},
  journal = {Journal of the Atmospheric Sciences},
  year    = {1987},
  volume  = {44},
  pages   = {1296--1317},
  doi     = {10.1175/1520-0469(1987)044<1296:ICSOHA>2.0.CO;2}
}
""",
        "doi": "10.1175/1520-0469(1987)044<1296:ICSOHA>2.0.CO;2",
    },
    {
        "bib_key": "Rogers2013",
        "bibtex": """@article{Rogers2013,
  author  = {Rogers, R. and Reasor, P. and Lorsolo, S.},
  title   = {Airborne {D}oppler Observations of the Inner-Core Structural Differences between Intensifying and Steady-State Tropical Cyclones},
  journal = {Monthly Weather Review},
  year    = {2013},
  volume  = {141},
  pages   = {2970--2991},
  doi     = {10.1175/MWR-D-12-00357.1}
}
""",
        "doi": "10.1175/MWR-D-12-00357.1",
    },
    {
        "bib_key": "Knaff2019",
        "bibtex": """@article{Knaff2019,
  author  = {Knaff, J. A. and Slocum, C. J. and Musgrave, K. D.},
  title   = {Quantification and Exploration of Diurnal Oscillations in Tropical Cyclones},
  journal = {Weather and Forecasting},
  year    = {2019},
  volume  = {34},
  pages   = {1099--1116},
  doi     = {10.1175/WAF-D-18-0143.1}
}
""",
        "doi": "10.1175/WAF-D-18-0143.1",
    },
    {
        "bib_key": "Bengtsson1995",
        "bibtex": """@article{Bengtsson1995,
  author  = {Bengtsson, L. and Botzet, M. and Esch, M.},
  title   = {Hurricane-Type Vortices in a General Circulation Model},
  journal = {Tellus A: Dynamic Meteorology and Oceanography},
  year    = {1995},
  volume  = {47},
  pages   = {175--196},
  doi     = {10.3402/tellusa.v47i2.11500}
}
""",
        "doi": "10.3402/tellusa.v47i2.11500",
    },
    {
        "bib_key": "Hersbach2000",
        "bibtex": """@article{Hersbach2000,
  author  = {Hersbach, H.},
  title   = {Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems},
  journal = {Weather and Forecasting},
  year    = {2000},
  volume  = {15},
  pages   = {559--570},
  doi     = {10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2}
}
""",
        "doi": "10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2",
    },
]


def existing_keys(bib_text: str) -> set[str]:
    import re

    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))


def main(dry_run: bool = False):
    bib_text = BIB.read_text()
    existing = existing_keys(bib_text)
    print(f"existing bib has {len(existing)} entries")

    to_add = [r for r in REFS if r["bib_key"] not in existing]
    print(f"adding {len(to_add)} new entries to bib:")
    for r in to_add:
        print(f"  - {r['bib_key']}")

    if to_add and not dry_run:
        with BIB.open("a") as f:
            f.write("\n\n%% Milton case-study methodology (added 2026-06-03)\n\n")
            for r in to_add:
                f.write(r["bibtex"] + "\n")
        print(f"wrote {len(to_add)} entries -> {BIB}")

    # Zotero sync via pyzotero
    print()
    print("=" * 60)
    print("Syncing to Zotero collection C8CXQJFN...")
    z = zotero.Zotero(ZOT_USER, "user", ZOT_KEY)
    # Check what's already in the collection by DOI
    items = z.everything(z.collection_items(ZOT_COLLECTION))
    existing_dois = set()
    for it in items:
        data = it.get("data", {})
        doi = data.get("DOI", "").lower()
        if doi:
            existing_dois.add(doi)
        # also check extra for DOI
        extra = data.get("extra", "")
        if "DOI:" in extra:
            for line in extra.splitlines():
                if line.startswith("DOI:"):
                    existing_dois.add(line.split(":", 1)[1].strip().lower())
    print(f"collection has {len(items)} items; {len(existing_dois)} unique DOIs")

    to_z = [r for r in REFS if r["doi"].lower() not in existing_dois]
    print(f"\n{len(to_z)} entries NOT in Zotero -- adding via DOI lookup")
    if dry_run:
        print("(dry run)")
        return

    # Use Zotero's "add by identifier" via the search endpoint approach
    # The cleanest is z.add_items() with skeleton + manual fields.
    # But the modern pyzotero API supports doi-based item creation only via raw template.
    template = z.item_template("journalArticle")
    for r in to_z:
        item = dict(template)
        # parse the bibtex to populate fields
        bib = r["bibtex"]
        import re

        def field(name):
            m = re.search(rf"{name}\s*=\s*\{{([^}}]+)\}}", bib)
            return m.group(1).strip() if m else ""

        title = field("title")
        authors_raw = field("author")
        year = field("year")
        journal = field("journal")
        volume = field("volume")
        pages = field("pages")
        item["title"] = title
        item["publicationTitle"] = journal
        item["date"] = year
        item["volume"] = volume
        item["pages"] = pages
        item["DOI"] = r["doi"]
        item["url"] = f"https://doi.org/{r['doi']}"
        item["collections"] = [ZOT_COLLECTION]
        creators = []
        for a in authors_raw.split(" and "):
            a = a.strip().rstrip(",").strip()
            if "," in a:
                last, first = [s.strip() for s in a.split(",", 1)]
            else:
                parts = a.split()
                first, last = " ".join(parts[:-1]), parts[-1]
            creators.append({"creatorType": "author", "firstName": first, "lastName": last})
        item["creators"] = creators
        item["tags"] = [{"tag": "milton-case-study"}, {"tag": "method"}]
        resp = z.create_items([item])
        if resp.get("successful"):
            print(f"  + {r['bib_key']}: added Zotero key {resp['successful']['0']['key']}")
        else:
            print(f"  ! {r['bib_key']}: {resp.get('failed', resp)}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    main(dry)
