"""Write a pip-freeze style lock file for each built container image.

Reads the dist-info metadata straight out of the squashfs image with
`unsquashfs` (no container runtime, runs on a login node), so the pins are
exactly what the inference runs used, including the git commits of the
packages the Dockerfiles install from a branch (earth2studio@main, graphcast).

    python containers/freeze_sqsh.py                 # all models -> containers/locks/
    python containers/freeze_sqsh.py sfno aurora     # subset
    AIENS_STORE=/path/to/images python containers/freeze_sqsh.py

Stdlib only; needs `unsquashfs` >= 4.5 (for `-cat`) on PATH.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STORE = Path(
    os.environ.get("AIENS_STORE", "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles")
)
MODELS = ["aurora", "graphcast", "sfno", "fcn3", "atlas", "aifsens"]
SITE = "usr/local/lib/python3.12/dist-packages"
OUT = Path(__file__).resolve().parent / "locks"


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _distributions(image: Path) -> list[str]:
    listing = _run("unsquashfs", "-l", "-d", "", str(image), SITE)
    names = []
    for line in listing.splitlines():
        rel = line.removeprefix("/" + SITE + "/")
        if "/" not in rel and rel.endswith((".dist-info", ".egg-info")):
            names.append(rel)
    return sorted(names, key=str.lower)


def _direct_url(image: Path, dist: str) -> dict | None:
    try:
        raw = _run("unsquashfs", "-cat", str(image), f"{SITE}/{dist}/direct_url.json")
    except subprocess.CalledProcessError:
        return None
    return json.loads(raw) if raw.strip() else None


def _line(image: Path, dist: str) -> str:
    stem = dist.rsplit(".", 1)[0]
    name, _, version = stem.partition("-")
    url = _direct_url(image, dist)
    if url and "vcs_info" in url:
        vcs = url["vcs_info"]
        rev = vcs.get("requested_revision")
        note = f"  # was requested as @{rev}" if rev and rev != vcs["commit_id"] else ""
        return f"{name} @ git+{url['url']}@{vcs['commit_id']}{note}"
    if url and url.get("url", "").startswith("file://"):
        return f"{name}=={version}  # local build: {url['url']}"
    return f"{name}=={version}"


def freeze(model: str) -> Path:
    image = (STORE / f"{model}.sqsh").resolve()
    stat = image.stat()
    built = datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    os_release = _run("unsquashfs", "-cat", str(image), "etc/os-release")
    pretty = next(
        (
            ln.split("=", 1)[1].strip('"')
            for ln in os_release.splitlines()
            if ln.startswith("PRETTY_NAME")
        ),
        "?",
    )
    lines = [
        f"# Frozen from {image}",
        f"# image size {stat.st_size / 1e9:.1f} GB, built {built}, {pretty}, Python 3.12 ({SITE})",
        f"# Dockerfile: containers/{model}.dockerfile; regenerate: python containers/freeze_sqsh.py {model}",
        "# NGC-provided packages carry '+nv'/'local build' tags and come with the base image, not PyPI.",
        "",
    ]
    lines += [_line(image, d) for d in _distributions(image)]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{model}.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    for m in sys.argv[1:] or MODELS:
        print(freeze(m))
