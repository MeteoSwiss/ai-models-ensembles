"""Regenerate the dataset inventory of the project store (docs/data/STORE_INVENTORY.md).

Walks AIENS_STORE, groups every forecast.zarr into an experiment (its path with the
init-date component removed), reads the schema of one zarr per experiment, lists the
evaluation products next to it, and joins the hand-written descriptions in
docs/data/catalog.csv. Experiments missing from the catalog are flagged, so a new run
cannot go undocumented silently.

    python tools/inventory_store.py                  # facts only (~1 min)
    python tools/inventory_store.py --dust           # + sizes and the >= 1 TB table (~1 min for 92 TB)
    python tools/inventory_store.py --dust --publish # + copy the docs into the store

Writes docs/data/STORE_INVENTORY.md and docs/data/store_inventory.json, and with sizes also
docs/data/LARGE_DATASETS.md: every dataset of 1 TB or more with location, owner, status and a
clear-up verdict, inside the store and outside it (EXTERNAL inputs, OWNER_ROOTS). Sizes are disk
usage from `dust` (https://github.com/bootandy/dust, on PATH), in decimal units. Without dust, give
--sizes '<MB>\t<path rel. to store>' and --external '<MB>\t<user:group>\t<abs path>' TSVs
(MB = 10^6 bytes). With --publish also
$STORE/README.md (= docs/DATA.md), $STORE/STORE_INVENTORY.md, $STORE/catalog.csv and a README.md
with the matching inventory rows in every top-level store directory, so the data documents itself
for anyone browsing the file system.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

from _env import REPO, STORE

DOCS = REPO / "docs" / "data"
INIT_RE = re.compile(r"^\d{8}(_\d{4})?$")
FORECAST_ROOTS = [
    "baselines",
    "baselines_m50",
    "ablation",
    "ablation_seed43",
    "_archived",
    "scratch",
]
PRODUCT_DIRS = {
    "eval": "SwissClim modules",
    "spatial_mean_ssr": "spatial-mean SSR (tools/spatial_mean_ssr.py)",
    "per_pixel_ssr": "per-pixel SSR (tools/per_pixel_ssr.py)",
    "energy_variogram_score": "ES/VS (tools/energy_variogram_score.py)",
    "eval_milton": "Milton-box eval",
    "intercomparison": "cross-run intercomparison",
    "_e2s_work": "leftover merge workspace of an interrupted run",
}
# Project inputs outside the store: status (catalog vocabulary) and description.
EXTERNAL = {
    "/capstor/store/cscs/swissai/ab016/IFS/ifs_ens.zarr": (
        "paper",
        "IFS-ENS reference forecasts (WB2 archive, 50 members, 10 used); code default since 2026-11-02",
    ),
    "/capstor/store/cscs/swissai/ab016/IFS/ifs_analysis_perturbed_ic.zarr": (
        "paper",
        "IFS-ENS perturbed analyses, per-member ICs of the Phase 5 (_ic, _ic_only) runs",
    ),
    "/capstor/store/cscs/swissai/ab016/IFS/ifs_control.zarr": (
        "unused",
        "IFS control analyses; prepared for an IC experiment, not read by any current script",
    ),
    "/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr": (
        "duplicate",
        "original of the ab016 copy (frozen a122 tree), identical (file list, sizes, sampled checksums; "
        "2026-11-02); read by every run before 2026-11-02. Others may still use this path",
    ),
    "/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr": (
        "duplicate",
        "original of the ab016 copy, identical; others may still use this path",
    ),
    "/capstor/store/cscs/swissai/a122/IFS/ifs_control.zarr": (
        "duplicate",
        "original of the ab016 copy, identical",
    ),
}
VERDICT = {
    "paper": "keep: data of the paper",
    "talk only": "optional: talk material, not in the paper",
    "diagnostic": "optional: post-submission diagnostic",
    "unused": "optional: not read by any script",
    "superseded": "delete: superseded",
    "failed run": "delete: failed run",
    "smoke test": "delete: smoke test",
    "archive": "delete: kept only for the record",
    "debug": "delete: debug output",
    "duplicate": "delete: duplicate of a verified copy",
    "cache (do not delete)": "keep: only copy, days to rebuild",
}
LARGE_MB = 1_000_000
# The store owner's other data, listed in the full (store) version of the large-dataset table.
OWNER_ROOTS = [
    "/capstor/store/cscs/swissai/ab016/sadamov",
    "/capstor/store/cscs/swissai/a122/sadamov",
    "/capstor/store/cscs/mch/s83/sadamov",
    "/capstor/store/mch/msopr/sadamov",
]
OTHER_MB = 100_000  # the owner's non-project data is listed from 100 GB (full version only)

OTHER_ROOTS = ["analysis", "diagnostics", "e2s_cache_backup", "earth2studio_cache"]
LOG_ROOTS = ["ablation_logs", "baseline_logs", "baseline_eval_logs", "logs", "model_test_logs"]


def _find_zarrs(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        keep = []
        for d in dirnames:
            if d.endswith(".zarr"):
                found.append(Path(dirpath) / d)
            elif (
                depth < 5
                and d != "_e2s_work"
                and not d.startswith(("eval", "intercomparison", "spatial_mean", "per_pixel"))
            ):
                keep.append(d)
        dirnames[:] = keep
    return sorted(found)


def _experiment_key(zarr: Path) -> tuple[str, str]:
    rel = zarr.relative_to(STORE)
    parts = list(rel.parts[:-1]) if rel.name == "forecast.zarr" else list(rel.parts)
    inits = [p for p in parts if INIT_RE.match(p)]
    key = "/".join(p for p in parts if not INIT_RE.match(p))
    return key, (inits[0] if inits else "")


def _schema(zarr: Path) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # stores without consolidated metadata
            ds = xr.open_zarr(zarr, consolidated=None, decode_timedelta=True)
    except Exception as exc:  # truncated/aborted stores are part of the record
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    out = {
        "variables": sorted(ds.data_vars),
        "dims": {k: int(v) for k, v in ds.sizes.items()},
    }
    if "lead_time" in ds.coords:
        hours = ds["lead_time"].values.astype("timedelta64[h]").astype(int)
        out["lead_hours"] = [
            int(hours[0]),
            int(hours[-1]),
            int(hours[1] - hours[0]) if len(hours) > 1 else 0,
        ]
    if "level" in ds.coords:
        out["levels"] = [int(v) for v in ds["level"].values]
    if "latitude" in ds.coords:
        lat = ds["latitude"].values
        out["grid"] = (
            f"{ds.sizes['latitude']}x{ds.sizes['longitude']} lat {lat[0]:.2f}..{lat[-1]:.2f}"
        )
    some = next(iter(ds.data_vars.values()), None)
    if some is not None:
        out["dtype"] = str(some.dtype)
        out["dim_order"] = list(some.dims)
    return out


def _products(exp_dir: Path) -> list[str]:
    if not exp_dir.is_dir():
        return []
    names = []
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir() or INIT_RE.match(p.name):
            continue
        base = next((k for k in PRODUCT_DIRS if p.name.startswith(k)), None)
        if base == "eval":
            modules = sorted(m.name for m in p.iterdir() if m.is_dir())
            names.append(f"{p.name}/{{{','.join(modules)}}}")
        elif base:
            names.append(p.name)
    return names


def _dust(path: Path | str, depth: int) -> dict[str, int]:
    """Disk usage in bytes of path and its descendants down to depth, keyed by absolute path."""
    out = subprocess.run(
        ["dust", "-j", "-d", str(depth), "-o", "b", str(path)], capture_output=True, text=True
    )
    sizes: dict[str, int] = {}

    def walk(node: dict) -> None:
        sizes[node["name"]] = node["size"]
        for child in node.get("children", []):
            walk(child)

    if out.stdout.strip():
        walk(json.loads(out.stdout))
    return sizes


def _load_sizes(path: Path | None, use_dust: bool) -> dict[str, int]:
    """Sizes in MB (10^6 bytes) keyed by path relative to the store."""
    sizes: dict[str, int] = {}
    if path:
        for line in path.read_text().splitlines():
            mb, rel = line.split("\t", 1)
            sizes[rel.rstrip("/")] = int(mb)
    if use_dust:
        for full, nbytes in _dust(STORE, 2).items():
            if full != str(STORE):
                sizes[str(Path(full).relative_to(STORE))] = nbytes // 1_000_000
    return sizes


def _load_external(path: Path | None, use_dust: bool) -> dict[str, tuple[int, str]]:
    """{abs path: (MB, 'user:group')} for EXTERNAL inputs and the children of OWNER_ROOTS."""
    ext: dict[str, tuple[int, str]] = {}
    if path:
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0].isdigit():
                ext[parts[2]] = (int(parts[0]), parts[1])
    if use_dust:
        for p in EXTERNAL:
            if Path(p).exists():
                ext[p] = (_dust(p, 0).get(p, 0) // 1_000_000, _owner(Path(p)))
        for root in OWNER_ROOTS:
            if not Path(root).is_dir():
                continue
            for full, nbytes in _dust(root, 1).items():
                if full != root and not full.startswith(str(STORE)):
                    ext[full] = (nbytes // 1_000_000, _owner(Path(full)))
    return ext


def _size(sizes: dict[str, int], key: str) -> str:
    mb = sizes.get(key)
    if mb is None:
        return ""
    return f"{mb / 1e6:.1f} TB" if mb >= 1e6 else f"{mb / 1e3:.0f} GB" if mb >= 1e3 else f"{mb} MB"


def _catalog() -> dict[str, dict]:
    path = DOCS / "catalog.csv"
    if not path.exists():
        return {}
    with path.open() as fh:
        return {row["key"]: row for row in csv.DictReader(fh)}


def _lookup(catalog: dict[str, dict], key: str) -> dict:
    """Exact entry, else the longest catalog key that is a path prefix (run tags inherit their phase)."""
    if key in catalog:
        return catalog[key]
    parents = [k for k in catalog if key.startswith(k + "/")]
    if not parents:
        return {}
    parent = max(parents, key=len)
    return {**catalog[parent], "description": f"see `{parent}`", "inherited": parent}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dust", action="store_true", help="measure sizes with dust (~1 min)")
    ap.add_argument("--sizes", type=Path, help="TSV '<MB>\\t<path rel. to store>' instead of dust")
    ap.add_argument("--publish", action="store_true", help="copy the docs into the store")
    ap.add_argument(
        "--external",
        type=Path,
        help="TSV '<MB>\\t<user:group>\\t<abs path>' for data outside the store",
    )
    args = ap.parse_args()

    experiments: dict[str, dict] = {}
    for root in FORECAST_ROOTS:
        for z in _find_zarrs(STORE / root):
            key, init = _experiment_key(z)
            exp = experiments.setdefault(key, {"zarrs": 0, "inits": [], "sample": z})
            exp["zarrs"] += 1
            if init:
                exp["inits"].append(init)
    # experiment dirs that only hold eval products (external models, intercomparisons)
    for root in ["baselines", "ablation"]:
        for p in sorted((STORE / root).glob("*" if root == "baselines" else "*/*")):
            key = str(p.relative_to(STORE))
            if p.is_dir() and key not in experiments and _products(p):
                experiments[key] = {"zarrs": 0, "inits": [], "sample": None}

    catalog = _catalog()
    sizes = _load_sizes(args.sizes, args.dust)

    records = []
    for key in sorted(experiments):
        exp = experiments[key]
        inits = sorted(set(exp["inits"]))
        rec = {
            "key": key,
            "path": str(STORE / key),
            "forecast_zarrs": exp["zarrs"],
            "n_inits": len(inits),
            "init_range": [inits[0], inits[-1]] if inits else [],
            "schema": _schema(exp["sample"]) if exp["sample"] else {},
            "products": _products(STORE / key),
            "size": _size(sizes, key),
            "catalog": _lookup(catalog, key),
        }
        records.append(rec)

    others = []
    for root in OTHER_ROOTS:
        base = STORE / root
        if not base.exists():
            continue
        for p in sorted(base.iterdir()):
            if p.name == "README.md":  # written by --publish
                continue
            nfiles = sum(len(f) for _, _, f in os.walk(p)) if p.is_dir() else 1
            key = str(p.relative_to(STORE))
            others.append(
                {
                    "key": key,
                    "files": nfiles,
                    "size": _size(sizes, key),
                    "catalog": _lookup(catalog, key),
                }
            )
    logs = [
        {"key": root, "files": sum(len(f) for _, _, f in os.walk(STORE / root))}
        for root in LOG_ROOTS
        if (STORE / root).exists()
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "store_inventory.json").write_text(
        json.dumps(
            {
                "generated": stamp,
                "store": str(STORE),
                "experiments": records,
                "other": others,
                "logs": logs,
            },
            indent=1,
        )
        + "\n"
    )
    _write_markdown(records, others, logs, stamp, sizes)
    external = _load_external(args.external, args.dust)
    large = _large_tables(sizes, catalog, external, stamp) if sizes else None
    if large:
        (DOCS / "LARGE_DATASETS.md").write_text(large[0])
    if args.publish:
        _publish(stamp)
        if large:
            (STORE / "LARGE_DATASETS.md").write_text(large[1])


def _owner(path: Path) -> str:
    out = subprocess.run(["stat", "-c", "%U:%G", str(path)], capture_output=True, text=True)
    return out.stdout.strip() or "?"


def _large_tables(sizes, catalog, external, stamp) -> tuple[str, str]:
    """(public, full) Markdown of every >= 1 TB dataset; public hides other people's user names."""
    store_user = _owner(STORE).split(":")[0]
    rows = []  # (MB, path, owner, status, description)
    for key, mb in sizes.items():
        if mb >= LARGE_MB and "/" in key and (STORE / key).exists():
            entry = _lookup(catalog, key)
            rows.append(
                (
                    mb,
                    str(STORE / key),
                    _owner(STORE / key),
                    entry.get("status", "?"),
                    entry.get("description", ""),
                )
            )
    other = []
    if external:
        ext = external
        for path, (mb, owner) in ext.items():
            if path in EXTERNAL:
                rows.append((mb, path, owner, *EXTERNAL[path]))
            elif mb >= OTHER_MB and (
                f"/{store_user}/" in path or owner.startswith(store_user + ":")
            ):
                twin = ext.get(path.replace("/a122/", "/ab016/"))
                dup = "/a122/" in path and twin and abs(twin[0] - mb) <= 0.01 * mb
                other.append(
                    (mb, path, owner, "duplicate of the ab016 copy (a122 is frozen)" if dup else "")
                )
    rows.sort(key=lambda r: -r[0])

    def fmt(mb):
        return f"{mb / 1e6:.1f}"

    def table(public: bool) -> list[str]:
        out = [
            "| size (TB) | path | owner | status | verdict | what |",
            "|---|---|---|---|---|---|",
        ]
        for mb, path, owner, status, desc in rows:
            user, _, group = owner.partition(":")
            shown = owner if (not public or user == store_user) else f"another `{group}` member"
            verdict = VERDICT.get(status, "check")
            short = desc.split(". ")[0].rstrip(".").replace("|", "/")
            out.append(f"| {fmt(mb)} | `{path}` | {shown} | {status} | {verdict} | {short} |")
        return out

    totals: dict[str, float] = {}
    for mb, *_rest, status, _desc in rows:
        k = VERDICT.get(status, "check").split(":")[0]
        totals[k] = totals.get(k, 0) + mb
    summary = ", ".join(f"{k} {fmt(v)} TB" for k, v in sorted(totals.items()))
    head = [
        "# Datasets of 1 TB or more",
        "",
        f"Generated {stamp} by `python tools/inventory_store.py --sizes ... --external ...` (sizes from `du`, "
        "disk usage). Status and descriptions come from [catalog.csv](catalog.csv) and `EXTERNAL` in the "
        "script; details per dataset in [STORE_INVENTORY.md](STORE_INVENTORY.md) and [../DATA.md](../DATA.md).",
        "",
        "Verdicts: `keep` = needed to reproduce the paper (or the only copy of something expensive); "
        "`optional` = not in the paper, delete once nobody needs it for the revision or talks; `delete` = "
        "safe to remove (superseded, failed, duplicate). Nothing has been deleted. The files stay owned by "
        "the original account after it is closed: deleting needs the owner, or CSCS / the project PI.",
        "",
        "The paper's data statement does not promise any forecast output (they are regenerable from public "
        "inputs and the code), so `keep` means: keep while the review runs, because reviewers may ask for new "
        "evaluations, which then take hours instead of days of GPU time. After acceptance the `keep` rows can go "
        "too; regenerating one 112-init production experiment costs about 15-30 GH200 node-hours of inference "
        "(warm caches) plus up to 8 x 12 h of evaluation (docs/REPRODUCING.md). Exception: the IFS-ENS perturbed analyses came from ECMWF MARS and need MARS "
        "access to rebuild.",
        "",
        f"Total by verdict: {summary}.",
        "",
    ]
    public = head + table(True)
    full = head + table(False)
    if other:
        full += [
            "",
            f"## Other directories of 100 GB or more owned by `{store_user}` (other projects)",
            "",
            "Not part of this project: the neural-LAM paper (handover note "
            "`/capstor/store/cscs/swissai/ab016/sadamov/HANDOVER.md`) in the swissai trees, earlier MeteoSwiss "
            "work (ICON, pollen, aldernet, gwen, FourCastNet) in `/capstor/store/mch/msopr`. Listed so the "
            "owner's data can be cleared in one go. Searched 2026-11-02 on every Clariden mount "
            "(`/capstor/store/{cscs,mch}`, `/iopsstor`, `/ritom`); file systems only Balfrin/Linard mount "
            "were not checked.",
            "",
            "| size (TB) | path | owner | note |",
            "|---|---|---|---|",
            *[
                f"| {fmt(mb)} | `{path}` | {owner} | {note} |"
                for mb, path, owner, note in sorted(other, key=lambda r: -r[0])
            ],
        ]
    return "\n".join(public) + "\n", "\n".join(full) + "\n"


def _publish(stamp: str) -> None:
    head = (
        f"<!-- Copied from the ai-models-ensembles repo (github.com/MeteoSwiss/ai-models-ensembles) "
        f"by tools/inventory_store.py --publish, {stamp}. Edit the repo, not this file. "
        f"Relative links point into the repo. -->\n\n"
    )
    (STORE / "README.md").write_text(head + (REPO / "docs" / "DATA.md").read_text())
    inventory = (DOCS / "STORE_INVENTORY.md").read_text()
    (STORE / "STORE_INVENTORY.md").write_text(head + inventory)
    (STORE / "catalog.csv").write_text((DOCS / "catalog.csv").read_text())
    sections: dict[str, list[str]] = {}
    current = None
    for line in inventory.splitlines():
        if line.startswith("## "):
            current = line[3:].strip().rstrip("/")
            sections[current] = []
        elif current:
            sections[current].append(line)
    other_rows = sections.get("Other directories", [])
    for top in sorted(p.name for p in STORE.iterdir() if p.is_dir()):
        if top in sections:
            body = sections[top]
        else:
            rows = [r for r in other_rows if r.startswith(f"| `{top}/")]
            if not rows:
                continue
            body = ["", "| path | files | size | description |", "|---|---|---|---|", *rows]
        (STORE / top / "README.md").write_text(
            head
            + f"# {top}/\n\nPart of the ai-models-ensembles project store. Full documentation of every "
            f"dataset: ../README.md (section 6) and ../STORE_INVENTORY.md.\n"
            + "\n".join(body)
            + "\n"
        )


def _write_markdown(records, others, logs, stamp, sizes) -> None:
    def cat(rec, field):
        return (rec["catalog"].get(field) or "").replace("|", "/")

    lines = [
        "# Store inventory (generated)",
        "",
        f"Generated {stamp} by `python tools/inventory_store.py` from `{STORE}`.",
        "Do not edit by hand: change [catalog.csv](catalog.csv) (descriptions) and rerun. "
        "Narrative documentation of every dataset: [../DATA.md](../DATA.md).",
        "",
        "Columns: `zarrs` = number of `forecast.zarr` stores; `inits` = distinct init dates; "
        "`schema` = members x leads (first..last h, step) and grid of one representative store; "
        "`products` = evaluation outputs stored next to the forecasts.",
        "",
    ]
    undocumented = [r["key"] for r in records if not r["catalog"]] + [
        o["key"] for o in others if not o["catalog"]
    ]
    if undocumented:
        lines += ["Not in catalog.csv yet: " + ", ".join(f"`{k}`" for k in undocumented), ""]

    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(r["key"].split("/")[0], []).append(r)
    for group, recs in groups.items():
        lines += [f"## {group}/", "", f"Total size: {_size(sizes, group) or 'n/a'}", ""]
        lines += [
            "| experiment | status | description | zarrs | inits | schema | size | products |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in recs:
            s = r["schema"]
            if "error" in s:
                schema = f"unreadable ({s['error'][:60]})"
            elif s:
                d = s["dims"]
                lh = s.get("lead_hours", [0, 0, 0])
                schema = (
                    f"{d.get('ensemble', 1)} mem x {lh[0]}..{lh[1]} h/{lh[2]} h, {len(s['variables'])} vars"
                    + (f", lev {','.join(map(str, s['levels']))}" if s.get("levels") else "")
                )
            else:
                schema = ""
            rng = "..".join(r["init_range"]) if r["init_range"] else ""
            lines.append(
                f"| `{r['key'].split('/', 1)[-1]}` | {cat(r, 'status')} | {cat(r, 'description')} | "
                f"{r['forecast_zarrs']} | {r['n_inits']} {rng} | {schema} | {r['size']} | "
                f"{'<br>'.join(r['products'])} |"
            )
        lines.append("")

    lines += [
        "## Other directories",
        "",
        "| path | files | size | description |",
        "|---|---|---|---|",
    ]
    for o in others:
        lines.append(f"| `{o['key']}` | {o['files']} | {o['size']} | {cat(o, 'description')} |")
    lines += ["", "## Slurm logs", "", "| path | files |", "|---|---|"]
    lines += [f"| `{lg['key']}` | {lg['files']} |" for lg in logs]

    variables = sorted({v for r in records for v in r["schema"].get("variables", [])})
    lines += [
        "",
        "## Variables seen across all forecasts",
        "",
        ", ".join(f"`{v}`" for v in variables),
        "",
    ]
    (DOCS / "STORE_INVENTORY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
