"""Post-download log-pressure fill for the 2 PL levels MARS does not archive
for type=pf step=0 (150 + 600 hPa) in the IFS-ENS perturbed-IC zarr.

This is the CF-long-name pendant of repo-download-ifs/fill_interp_levels.py
(that script keys on GRIB short names t/u/v/q/z/w; the live store at
$IC_ZARR was renamed to CF long names, so the short-name script KeyErrors on
it). Same log-pressure-linear formula and same per-(init_time, level) region
write into the per-init-sharded v3 store:

    f(150) = (1-a)*f(100) + a*f(200),  a = ln(150/100)/ln(200/100)
    f(600) = (1-a)*f(500) + a*f(700),  a = ln(600/500)/ln(700/500)

Idempotent: a (init_time, var) slab whose target level is already finite and
nonzero is skipped. Only ever writes into currently-NaN target cells, so it
cannot corrupt archived levels or already-filled inits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# CF long names in the live IC store, GRIB short -> long.
PL_VARS = (
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
)

# target_level: (lower_archived, upper_archived, alpha)
INTERP = {
    150: (100, 200, (np.log(150) - np.log(100)) / (np.log(200) - np.log(100))),
    600: (500, 700, (np.log(600) - np.log(500)) / (np.log(700) - np.log(500))),
}


def _parse_indices(spec: str, n_init: int) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            out.update(range(int(a) if a else 0, int(b) if b else n_init))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < n_init)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--zarr", default="/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr"
    )
    p.add_argument(
        "--indices", required=True, help="init_time indices to fill, e.g. '0:56' or '0,1,2'"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="report what would be filled, write nothing"
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = Path(args.zarr)

    ds = xr.open_zarr(out, consolidated=False)
    n_init = ds.sizes["init_time"]
    idxs = _parse_indices(args.indices, n_init)
    logging.info(
        "zarr=%s n_init=%d filling %d init_time indices: %s",
        out,
        n_init,
        len(idxs),
        idxs if len(idxs) <= 12 else f"{idxs[:6]}...{idxs[-3:]}",
    )

    lev = ds.level.values
    n_written = 0
    for new_L, (lo, hi, alpha) in INTERP.items():
        li_new = int(np.where(lev == new_L)[0][0])
        li_lo = int(np.where(lev == lo)[0][0])
        li_hi = int(np.where(lev == hi)[0][0])
        logging.info(
            "L=%d <- (1-%.4f)*L%d + %.4f*L%d  [positions new=%d lo=%d hi=%d]",
            new_L,
            alpha,
            lo,
            alpha,
            hi,
            li_new,
            li_lo,
            li_hi,
        )
        for i in idxs:
            done = []
            for v in PL_VARS:
                probe = ds[v].isel(init_time=i, level=li_new, ensemble=0).values
                if np.all(np.isfinite(probe)) and np.any(probe):
                    continue
                if args.dry_run:
                    done.append(v)
                    continue
                lo_slab = ds[v].isel(init_time=i, level=li_lo).load()
                hi_slab = ds[v].isel(init_time=i, level=li_hi).load()
                interp = ((1 - alpha) * lo_slab + alpha * hi_slab).astype("float32")
                interp = interp.expand_dims(
                    {"init_time": [ds.init_time.values[i]], "level": [new_L]}
                )
                interp = interp.drop_vars(
                    [c for c in list(interp.coords) if "init_time" not in interp[c].dims]
                )
                interp.to_dataset(name=v).to_zarr(
                    out,
                    region={"init_time": slice(i, i + 1), "level": slice(li_new, li_new + 1)},
                    consolidated=False,
                )
                done.append(v)
                n_written += 1
            if done:
                logging.info(
                    "  init %d (%s) L=%d: %s %s",
                    i,
                    str(ds.init_time.values[i])[:13],
                    new_L,
                    "would fill" if args.dry_run else "filled",
                    done,
                )
    logging.info(
        "Done. %s %d (init,var,level) slabs.", "would write" if args.dry_run else "wrote", n_written
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
