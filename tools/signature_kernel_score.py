"""Signature Kernel Score over the lead-time trajectory for paper Tab. 3 supplement.

A strictly proper multivariate scoring rule that, unlike ES / VS / CRPS (which
score the marginal forecast distribution at a single lead), scores the whole
forecast PATH over lead time. It therefore tests the temporal-dependency
structure of the ensemble trajectory, the temporal analogue of what the
Variogram Score does for spatial structure. Introduced to weather verification
by Dodson & Dutta, "Signature Kernel Scoring Rule: A Spatio-Temporal Diagnostic
for Probabilistic Weather Forecasting", TMLR 05/2026 (OpenReview LOLXpt4E5D),
building on the signature kernel (Salvi et al. 2021, arXiv:2006.14794) and the
signature kernel scoring rule (Issa et al. 2023, arXiv:2305.04297).

SIGNATURE KERNEL SCORE (their Eq. 1), lower is better:

    phi(F, o) = E_{X,X' ~ F}[ Ksig(X, X') ] - 2 E_{X ~ F}[ Ksig(X, o) ]

with Ksig the signature kernel between two paths, computed as the solution of a
Goursat PDE (Salvi et al. 2021) under a static RBF kernel. The two terms mirror
the Energy Score with a sign switch (kernel = similarity, norm = distance): the
first penalises under-dispersion (members too self-similar), the second rewards
closeness of the member paths to the observed path.

Each pixel's path is the (var, level) trajectory over lead time, standardised
per layer by the per-init truth std (same normalisation as energy_variogram_score.py),
stacked into a multivariate path, then basepoint + time augmented (Morrill et al.
2021) so the signature is unique under translation and reparametrisation.

The score is expensive (a PDE solve per path pair), so it is evaluated on a
cos(lat)-weighted random sample of pixels per init, averaged over inits, exactly
the spatial-subsample strategy the Variogram Score uses.

Usage (one baseline -> one CSV row), mirrors energy_variogram_score.py:
    python tools/signature_kernel_score.py \\
        --forecast-zarrs <init1>/forecast.zarr [<init2>/forecast.zarr ...] \\
        --truth-zarrs <wb2_a.zarr> <wb2_b.zarr> \\
        --variables 2m_temperature mean_sea_level_pressure geopotential \\
                    temperature u_component_of_wind v_component_of_wind \\
                    specific_humidity \\
        --levels 500 850 --lead 240 --lead-stride 2 --n-pixels 128 \\
        --model-label aurora_encoder --out-csv <path>

Validate the kernel implementation first:
    python tools/signature_kernel_score.py --self-test

Output columns: model, score, lead_hours, n_inits, n_members, n_pixels, value.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# --------------------------------------------------------------------------- #
# Signature kernel via the Goursat PDE (Salvi et al. 2021)                     #
# --------------------------------------------------------------------------- #
def _sig_kernel_batch(
    A: np.ndarray,
    B: np.ndarray,
    sigma: float,
    dyadic: int,
    kernel: str = "rbf",
    batch: int = 2048,
) -> np.ndarray:
    """Signature kernel Ksig(A_b, B_b) for a batch of path pairs.

    Parameters
    ----------
    A, B : (P, L, D)
        Two batches of P paths, length L, dimension D.
    sigma : float
        Static-kernel bandwidth. RBF: exp(-||u-v||^2 / (2 sigma^2)).
    dyadic : int
        PDE grid dyadic refinement order (0 or 1 per the paper).
    kernel : {"rbf", "linear"}
        Static kernel on path points ("linear" only used by the self-test).

    Returns
    -------
    (P,) signature-kernel values K[..., -1, -1].
    """
    P = A.shape[0]
    out = np.empty(P, dtype=np.float64)
    for s in range(0, P, batch):
        a = A[s : s + batch].astype(np.float64)
        b = B[s : s + batch].astype(np.float64)
        if kernel == "rbf":
            d2 = ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)
            g = np.exp(-d2 / (2.0 * sigma * sigma))
        elif kernel == "linear":
            g = np.einsum("pid,pjd->pij", a, b)
        else:
            raise ValueError(kernel)
        # RKHS path increments: second mixed difference of the static Gram.
        inc = g[:, 1:, 1:] + g[:, :-1, :-1] - g[:, 1:, :-1] - g[:, :-1, 1:]
        f = 2**dyadic
        if f > 1:
            inc = np.repeat(np.repeat(inc, f, axis=1), f, axis=2) / float(f * f)
        out[s : s + batch] = _goursat(inc)
    return out


def _goursat(inc: np.ndarray) -> np.ndarray:
    """Solve the Goursat PDE on the refined increment grid (second-order scheme).

    inc : (P, m, n) RKHS increments. Returns K[:, -1, -1], shape (P,).
    Update (Salvi et al. 2021, the scheme used by the sigkernel package):
        K[i+1,j+1] = (K[i+1,j]+K[i,j+1])(1 + a/2 + a^2/12) - K[i,j](1 - a^2/12)
    """
    P, m, n = inc.shape
    K = np.ones((P, m + 1, n + 1), dtype=np.float64)
    for i in range(m):
        ki = K[:, i]
        ki1 = K[:, i + 1]
        for j in range(n):
            a = inc[:, i, j]
            a2 = a * a
            ki1[:, j + 1] = (ki1[:, j] + ki[:, j + 1]) * (1.0 + 0.5 * a + a2 / 12.0) - ki[:, j] * (
                1.0 - a2 / 12.0
            )
    return K[:, -1, -1]


# --------------------------------------------------------------------------- #
# Path construction: standardise, basepoint + time augment                    #
# --------------------------------------------------------------------------- #
def _augment(vals: np.ndarray) -> np.ndarray:
    """Basepoint + time augmentation of a value trajectory.

    vals : (..., L, V) standardised layer values over lead time.
    Returns (..., L+1, V+1): a zero basepoint prepended, time in [0, 1] as the
    leading coordinate. Removes translation / reparametrisation invariance so
    the signature kernel is strictly proper (Morrill et al. 2021).
    """
    lead = vals.shape[-2]
    v = vals.shape[-1]
    zero = np.zeros(vals.shape[:-2] + (1, v), dtype=vals.dtype)
    aug = np.concatenate([zero, vals], axis=-2)  # (..., L+1, V)
    t = np.linspace(0.0, 1.0, lead + 1, dtype=vals.dtype)
    t = np.broadcast_to(t, vals.shape[:-2] + (lead + 1,))[..., None]
    return np.concatenate([t, aug], axis=-1)  # (..., L+1, V+1)


# --------------------------------------------------------------------------- #
# Per-init load + score                                                        #
# --------------------------------------------------------------------------- #
def _load_init(
    fcst: xr.Dataset,
    truth: xr.Dataset,
    variables: list[str],
    levels: list[float],
    lead_hours: int,
    lead_stride: int,
    n_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load standardised member + truth paths on a random pixel sample.

    Returns (mem, tru, latw):
        mem  : (n_pix, M, L, V) standardised member trajectories
        tru  : (n_pix, L, V)    standardised truth trajectory
        latw : (n_pix,)         cos(lat) pixel weights
    """
    if "lead_time" not in fcst.dims:
        return None
    lt = fcst["lead_time"].values
    hours = (
        (lt / np.timedelta64(1, "h")).astype(int)
        if np.issubdtype(lt.dtype, np.timedelta64)
        else lt.astype(int)
    )
    sel = np.where(hours <= lead_hours)[0][::lead_stride]
    if sel.size < 3:
        return None
    lead_idx = sel
    lead_h = hours[sel]

    init_np = fcst["init_time"].values[0] if "init_time" in fcst.dims else None
    if init_np is None:
        return None
    valid_times = np.array([init_np + np.timedelta64(int(h), "h") for h in lead_h])

    lat = fcst["latitude"].values
    lon = fcst["longitude"].values
    nlat, nlon = lat.size, lon.size
    # area-weighted pixel sample: draw uniformly, keep cos(lat) as the weight.
    li = rng.integers(0, nlat, n_pixels)
    lj = rng.integers(0, nlon, n_pixels)
    pt = xr.DataArray  # alias
    lat_da = pt(li, dims="pt")
    lon_da = pt(lj, dims="pt")
    latw = np.cos(np.deg2rad(lat[li])).astype(np.float64)
    lat_vals = lat[li]
    lon_vals = lon[lj]

    mem_layers: list[np.ndarray] = []
    tru_layers: list[np.ndarray] = []
    M = None
    for var in variables:
        if var not in fcst.data_vars or var not in truth.data_vars:
            continue
        fa = fcst[var]
        ta = truth[var]
        fdims = "init_time" in fa.dims
        lvls = (
            [lv for lv in levels if float(lv) in [float(x) for x in fa["level"].values]]
            if "level" in fa.dims
            else [None]
        )
        for lvl in lvls:
            fl = fa.isel(init_time=0) if fdims else fa
            fl = fl.isel(lead_time=lead_idx)
            tl = ta.sel(time=valid_times)
            if lvl is not None:
                fl = fl.sel(level=lvl)
                tl = tl.sel(level=lvl)
            # pointwise pixel sample (xarray vectorised indexing)
            fl = fl.isel(latitude=lat_da, longitude=lon_da)
            tl = tl.sel(
                latitude=pt(lat_vals, dims="pt"),
                longitude=pt(lon_vals, dims="pt"),
                method="nearest",
            )
            fv = fl.transpose("ensemble", "lead_time", "pt").values  # (M, L, pt)
            tv = tl.transpose("time", "pt").values  # (L, pt)
            M = fv.shape[0]
            mem_layers.append(fv)
            tru_layers.append(tv)

    if not mem_layers or M is None:
        return None

    mem = np.stack(mem_layers, axis=-1)  # (M, L, pt, V)
    tru = np.stack(tru_layers, axis=-1)  # (L, pt, V)
    # standardise per layer by per-init truth std over the (lead, pixel) sample
    s = tru.reshape(-1, tru.shape[-1]).std(axis=0)  # (V,)
    s = np.where(s > 0, s, 1.0)
    mem = mem / s
    tru = tru / s
    mem = np.moveaxis(mem, 2, 0)  # (pt, M, L, V)
    tru = np.moveaxis(tru, 1, 0)  # (pt, L, V)
    return mem.astype(np.float32), tru.astype(np.float32), latw


def _score_init(
    mem: np.ndarray,
    tru: np.ndarray,
    latw: np.ndarray,
    sigma: float,
    dyadic: int,
    batch: int,
) -> float:
    """cos(lat)-weighted mean of the per-pixel signature kernel score (Eq. 1)."""
    n_pix, M, _, _ = mem.shape
    mem_p = _augment(mem)  # (pt, M, L+1, D)
    tru_p = _augment(tru)  # (pt, L+1, D)
    La, D = mem_p.shape[-2], mem_p.shape[-1]

    pairs = list(combinations(range(M), 2))
    # member-member: spread term E[Ksig(X, X')]
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    A_mm = mem_p[:, ii].reshape(-1, La, D)
    B_mm = mem_p[:, jj].reshape(-1, La, D)
    k_mm = _sig_kernel_batch(A_mm, B_mm, sigma, dyadic, "rbf", batch)
    spread = k_mm.reshape(n_pix, len(pairs)).mean(axis=1)  # (pt,)

    # member-truth: data term E[Ksig(X, o)]
    A_mo = mem_p.reshape(-1, La, D)
    B_mo = np.repeat(tru_p[:, None], M, axis=1).reshape(-1, La, D)
    k_mo = _sig_kernel_batch(A_mo, B_mo, sigma, dyadic, "rbf", batch)
    data = k_mo.reshape(n_pix, M).mean(axis=1)  # (pt,)

    sigk_pixel = spread - 2.0 * data  # (pt,)
    return float(np.sum(sigk_pixel * latw) / np.sum(latw))


# --------------------------------------------------------------------------- #
# Self-test: PDE signature kernel vs truncated-signature inner product         #
# --------------------------------------------------------------------------- #
def _trunc_sig(path: np.ndarray, level: int) -> list[np.ndarray]:
    """Truncated signature of a piecewise-linear path via Chen's identity.

    path : (L, D). Returns [S_0, S_1, ..., S_level], S_k flattened length D^k.
    """
    D = path.shape[1]
    sig = [np.array([1.0])] + [np.zeros(D**k) for k in range(1, level + 1)]
    for seg in np.diff(path, axis=0):
        # signature of one linear segment = tensor exp of the increment
        seg_sig = [np.array([1.0])]
        term = np.array([1.0])
        for k in range(1, level + 1):
            term = np.kron(term, seg) / k
            seg_sig.append(term)
        # Chen: sig = sig (x) seg_sig  (truncated tensor product)
        new = [np.zeros(D**k) for k in range(level + 1)]
        for a in range(level + 1):
            for b in range(level + 1 - a):
                new[a + b] = new[a + b] + np.kron(sig[a], seg_sig[b])
        sig = new
    return sig


def _self_test() -> int:
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(3):
        D = rng.integers(2, 4)
        X = np.cumsum(rng.normal(size=(5, D)) * 0.3, axis=0)
        Y = np.cumsum(rng.normal(size=(6, D)) * 0.3, axis=0)
        # truncated-signature inner product (== linear signature kernel)
        sx = _trunc_sig(X, 8)
        sy = _trunc_sig(Y, 8)
        ref = sum(float(np.dot(sx[k], sy[k])) for k in range(9))
        # PDE signature kernel, linear static kernel, increasing refinement
        kd = [
            float(_sig_kernel_batch(X[None], Y[None], sigma=1.0, dyadic=d, kernel="linear")[0])
            for d in (0, 1, 2, 3)
        ]
        err = abs(kd[-1] - ref) / abs(ref)
        sym = float(_sig_kernel_batch(Y[None], X[None], sigma=1.0, dyadic=3, kernel="linear")[0])
        print(
            f"trial {trial}: ref(trunc-sig)={ref:.6f}  PDE(dyadic 0..3)="
            f"{[round(v, 6) for v in kd]}  rel_err={err:.2e}  sym_err={abs(sym-kd[-1]):.1e}"
        )
        ok = ok and err < 5e-3 and abs(sym - kd[-1]) < 1e-9
    # RBF Gram positive-semidefinite sanity
    paths = np.cumsum(rng.normal(size=(6, 7, 3)) * 0.3, axis=1)
    A = np.repeat(paths, 6, axis=0)
    B = np.tile(paths, (6, 1, 1))
    G = _sig_kernel_batch(A, B, sigma=1.0, dyadic=1, kernel="rbf").reshape(6, 6)
    eig = np.linalg.eigvalsh(0.5 * (G + G.T))
    print(f"RBF signature Gram min eigenvalue = {eig.min():.3e} (>= ~0 expected)")
    ok = ok and eig.min() > -1e-6
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--forecast-zarrs", nargs="+")
    p.add_argument("--truth-zarrs", nargs="+")
    p.add_argument("--variables", nargs="+")
    p.add_argument("--levels", nargs="+", type=float, default=[500.0, 850.0])
    p.add_argument("--lead", type=int, default=240)
    p.add_argument("--lead-stride", type=int, default=2)
    p.add_argument("--n-pixels", type=int, default=128)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--dyadic", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-inits", type=int, default=0)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--model-label")
    p.add_argument("--out-csv")
    args = p.parse_args()

    if args.self_test:
        return _self_test()

    for req in ("forecast_zarrs", "truth_zarrs", "variables", "model_label", "out_csv"):
        if getattr(args, req) is None:
            p.error(f"--{req.replace('_', '-')} is required unless --self-test")

    print("Opening truth zarrs (consolidated=True)...", flush=True)
    truth_parts = [xr.open_zarr(tz, consolidated=True, chunks={}) for tz in args.truth_zarrs]
    truth_ds = xr.concat(truth_parts, dim="time") if len(truth_parts) > 1 else truth_parts[0]
    truth_ds = truth_ds.sortby("time")

    def fcst_ds_iter():
        for fz in args.forecast_zarrs:
            try:
                ds = xr.open_zarr(fz, consolidated=True, chunks={})
            except Exception:
                ds = xr.open_zarr(fz, consolidated=False, chunks={})
            if "init_time" in ds.sizes and ds.sizes["init_time"] > 1:
                for j in range(ds.sizes["init_time"]):
                    yield f"{Path(fz).name}#init_{j}", ds.isel(init_time=[j])
            else:
                yield Path(fz).parent.name, ds

    rng = np.random.default_rng(args.seed)
    per_init: list[float] = []
    n_members = None
    for k, (label, fcst_ds) in enumerate(fcst_ds_iter(), start=1):
        if args.max_inits and len(per_init) >= args.max_inits:
            break
        loaded = _load_init(
            fcst_ds,
            truth_ds,
            args.variables,
            args.levels,
            args.lead,
            args.lead_stride,
            args.n_pixels,
            rng,
        )
        if loaded is None:
            print(f"  [{k}] SKIP {label} (no usable layers/leads)", flush=True)
            continue
        mem, tru, latw = loaded
        n_members = mem.shape[1]
        val = _score_init(mem, tru, latw, args.sigma, args.dyadic, args.batch)
        per_init.append(val)
        print(
            f"  [{k}] {label}: SIGK={val:.5f}  running_mean={np.mean(per_init):.5f}"
            f"  (n={len(per_init)}, L={mem.shape[2]}, M={n_members})",
            flush=True,
        )

    if not per_init:
        print("No inits scored.", file=sys.stderr)
        return 1

    rows = [
        {
            "model": args.model_label,
            "score": "signature_kernel_score",
            "lead_hours": args.lead,
            "n_inits": len(per_init),
            "n_members": n_members,
            "n_pixels": args.n_pixels,
            "value": float(np.mean(per_init)),
        }
    ]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}: SIGK={np.mean(per_init):.5f} over {len(per_init)} inits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
