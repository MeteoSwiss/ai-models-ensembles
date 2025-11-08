from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ai_models_ensembles.utils import ensure_dir

matplotlib.use("Agg")
plt.ioff()

__all__ = ["run_intercompare"]


def _as_paths(items: Iterable[str | Path]) -> list[Path]:
    return [Path(item).expanduser().resolve() for item in items]


def _common_basenames(
    model_dirs: Sequence[Path], subdir: str, pattern: str
) -> list[str]:
    basenames: list[set[str]] = []
    for root in model_dirs:
        folder = root / subdir
        if not folder.is_dir():
            basenames.append(set())
            continue
        basenames.append(
            {child.name for child in folder.glob(pattern) if child.is_file()}
        )
    if not basenames:
        return []
    shared = set.intersection(*basenames) if len(basenames) > 1 else basenames[0]
    return sorted(shared)


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _plot_energy_spectra(
    models: Sequence[Path], labels: Sequence[str], out_root: Path
) -> None:
    basenames = _common_basenames(models, "energy_spectra", "*.npz")
    if not basenames:
        return
    out_dir = ensure_dir(out_root / "energy_spectra")
    colors = plt.cm.get_cmap("tab10", len(labels))
    for base in basenames:
        payloads = [_load_npz(model / "energy_spectra" / base) for model in models]
        wn = np.asarray(payloads[0].get("wavenumber"))
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        target = np.asarray(payloads[0].get("spectrum_target"))
        if target.size:
            ax.loglog(wn, target, color="black", lw=2.0, label="Ground Truth")
        for idx, (label, payload) in enumerate(zip(labels, payloads, strict=False)):
            spectrum = np.asarray(payload.get("spectrum_prediction", []))
            if spectrum.ndim == 2:
                spectrum = np.nanmean(spectrum, axis=0)
            if spectrum.size:
                ax.loglog(wn, spectrum, color=colors(idx), label=label)
            spectrum_unperturbed = np.asarray(payload.get("spectrum_unperturbed", []))
            if spectrum_unperturbed.size:
                ax.loglog(
                    wn,
                    spectrum_unperturbed,
                    color=colors(idx),
                    linestyle="--",
                    alpha=0.7,
                    label=f"{label} Unperturbed",
                )
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Energy Density")
        ax.set_title(base.replace(".npz", ""))
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(frameon=False)
        out_path = out_dir / base.replace(".npz", "_compare.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def _plot_rmse(models: Sequence[Path], labels: Sequence[str], out_root: Path) -> None:
    basenames = _common_basenames(models, "rmse", "*.nc")
    if not basenames:
        return
    out_dir = ensure_dir(out_root / "rmse")
    colors = plt.cm.get_cmap("tab10", len(labels))
    for base in basenames:
        datasets = [xr.load_dataset(model / "rmse" / base) for model in models]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        for idx, (label, ds) in enumerate(zip(labels, datasets, strict=False)):
            series = ds.get("rmse_mean")
            if series is None:
                continue
            ax.plot(series["step"], series.values, color=colors(idx), label=label)
        ax.set_xlabel("Lead Time")
        ax.set_ylabel("RMSE")
        ax.set_title(base.replace(".nc", ""))
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / base.replace(".nc", "_compare.png"), bbox_inches="tight")
        plt.close(fig)


def _plot_timeseries(
    models: Sequence[Path], labels: Sequence[str], out_root: Path
) -> None:
    basenames = _common_basenames(models, "timeseries", "*.nc")
    if not basenames:
        return
    out_dir = ensure_dir(out_root / "timeseries")
    colors = plt.cm.get_cmap("tab10", len(labels))
    for base in basenames:
        datasets = [xr.load_dataset(model / "timeseries" / base) for model in models]
        fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
        # assume ground truth identical across models – take first dataset
        gt = datasets[0].get("ground_truth")
        if gt is not None:
            ax.plot(gt["step"], gt.values, color="black", lw=2.0, label="Ground Truth")
        for idx, (label, ds) in enumerate(zip(labels, datasets, strict=False)):
            members = ds.get("forecast_members")
            if members is not None:
                mean_series = members.mean(
                    dim=[d for d in members.dims if d != "step"], skipna=True
                )
                ax.plot(
                    mean_series["step"],
                    mean_series.values,
                    color=colors(idx),
                    label=f"{label} Mean",
                )
            unperturbed = ds.get("forecast_unperturbed")
            if unperturbed is not None:
                ax.plot(
                    unperturbed["step"],
                    unperturbed.values,
                    color=colors(idx),
                    linestyle="--",
                    alpha=0.7,
                    label=f"{label} Unperturbed",
                )
        ax.set_xlabel("Lead Time")
        ax.set_ylabel("Value")
        ax.set_title(base.replace(".nc", ""))
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / base.replace(".nc", "_compare.png"), bbox_inches="tight")
        plt.close(fig)


def _plot_rank_histogram(
    models: Sequence[Path], labels: Sequence[str], out_root: Path
) -> None:
    basenames = _common_basenames(models, "rank_histogram", "*.npz")
    if not basenames:
        return
    out_dir = ensure_dir(out_root / "rank_histogram")
    colors = plt.cm.get_cmap("tab10", len(labels))
    for base in basenames:
        payloads = [_load_npz(model / "rank_histogram" / base) for model in models]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        for idx, (label, payload) in enumerate(zip(labels, payloads, strict=False)):
            ranks = np.asarray(payload.get("ranks"))
            counts = np.asarray(payload.get("counts"))
            if not ranks.size:
                continue
            ax.step(ranks, counts, where="mid", color=colors(idx), label=label)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(base.replace(".npz", ""))
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / base.replace(".npz", "_compare.png"), bbox_inches="tight")
        plt.close(fig)


def _plot_density(
    models: Sequence[Path], labels: Sequence[str], out_root: Path
) -> None:
    basenames = _common_basenames(models, "density", "*.npz")
    if not basenames:
        return
    out_dir = ensure_dir(out_root / "density")
    colors = plt.cm.get_cmap("tab10", len(labels))
    for base in basenames:
        payloads = [_load_npz(model / "density" / base) for model in models]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        gt_payload = payloads[0]
        x = np.asarray(gt_payload.get("x"))
        gt_pdf = np.asarray(gt_payload.get("ground_truth_pdf"))
        if x.size and gt_pdf.size:
            ax.plot(x, gt_pdf, color="black", lw=2.0, label="Ground Truth")
        for idx, (label, payload) in enumerate(zip(labels, payloads, strict=False)):
            model_pdf = np.asarray(payload.get("ensemble_pdf"))
            x_vals = np.asarray(payload.get("x"))
            if model_pdf.size and x_vals.size:
                ax.plot(x_vals, model_pdf, color=colors(idx), label=label)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(base.replace(".npz", ""))
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / base.replace(".npz", "_compare.png"), bbox_inches="tight")
        plt.close(fig)


_METRIC_DISPATCH = {
    "energy_spectra": _plot_energy_spectra,
    "rmse": _plot_rmse,
    "timeseries": _plot_timeseries,
    "rank_histogram": _plot_rank_histogram,
    "density": _plot_density,
    # New metrics: histogram grids and KDE ridgeline evolution
    "hist_global": lambda models, labels, out_root: _plot_histogram_grid(
        models, labels, out_root
    ),
    "wd_kde_evolve": lambda models, labels, out_root: _plot_wd_kde_ridgeline(
        models, labels, out_root
    ),
}


def run_intercompare(
    model_dirs: Iterable[str | Path],
    labels: Iterable[str],
    out_root: str | Path,
    metrics: Iterable[str],
) -> None:
    model_paths = _as_paths(model_dirs)
    labels_list = list(labels)
    if len(labels_list) != len(model_paths):
        raise ValueError("Number of labels must match number of model directories")
    out_path = ensure_dir(out_root)
    selected = {metric.lower() for metric in metrics}
    for name, func in _METRIC_DISPATCH.items():
        if name in selected:
            func(model_paths, labels_list, out_path)


def _plot_histogram_grid(
    models: Sequence[Path], labels: Sequence[str], out_root: Path
) -> None:
    basenames = _common_basenames(models, "histograms", "hist_global_*grid_data*.npz")
    if not basenames:
        return
    out_dir = ensure_dir(out_root / "histograms")
    colors = plt.cm.get_cmap("tab10", len(labels))
    for base in basenames:
        payloads = [_load_npz(model / "histograms" / base) for model in models]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        # Compare mean densities across leads (aggregate by averaging per-lead histograms)
        lead_hours = np.asarray(payloads[0].get("lead_hours", []))
        if lead_hours.size == 0:
            continue
        for idx, (label, payload) in enumerate(zip(labels, payloads, strict=False)):
            dens_true_list = payload.get("densities_true")
            dens_pred_list = payload.get("densities_pred")
            edges_list = payload.get("edges")
            if dens_true_list is None or dens_pred_list is None or edges_list is None:
                continue
            # Use first edges as representative
            edges0 = np.asarray(edges_list[0])
            # Mean across panels for a compact comparison curve
            mean_true = np.nanmean(np.vstack(dens_true_list), axis=0)
            mean_pred = np.nanmean(np.vstack(dens_pred_list), axis=0)
            ax.plot(
                edges0[:-1],
                mean_true,
                color="black" if idx == 0 else colors(idx),
                linestyle="--" if idx != 0 else "-",
                label=f"GT Mean ({label})" if idx != 0 else "Ground Truth Mean",
            )
            ax.plot(edges0[:-1], mean_pred, color=colors(idx), label=f"{label} Mean")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density (mean across lead panels)")
        ax.set_title(base.replace(".npz", ""))
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / base.replace(".npz", "_compare.png"), bbox_inches="tight")
        plt.close(fig)


def _plot_wd_kde_ridgeline(
    models: Sequence[Path], labels: Sequence[str], out_root: Path
) -> None:
    basenames = _common_basenames(
        models, "wd_kde", "wd_kde_evolve_*ridgeline_data*.npz"
    )
    if not basenames:
        return
    out_dir = ensure_dir(out_root / "wd_kde")
    colors = plt.cm.get_cmap("tab10", len(labels))
    for base in basenames:
        payloads = [_load_npz(model / "wd_kde" / base) for model in models]
        y_eval = np.asarray(payloads[0].get("y_eval", []))
        lead_hours = np.asarray(payloads[0].get("lead_hours", []))
        if y_eval.size == 0 or lead_hours.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        # Plot mean density across leads for target and each model
        target_density = np.asarray(payloads[0].get("density_target"))
        if target_density.size:
            mean_target = np.nanmean(target_density, axis=0)
            ax.plot(
                y_eval, mean_target, color="black", lw=2.0, label="Ground Truth Mean"
            )
        for idx, (label, payload) in enumerate(zip(labels, payloads, strict=False)):
            model_density = np.asarray(payload.get("density_model"))
            if model_density.size:
                mean_model = np.nanmean(model_density, axis=0)
                ax.plot(y_eval, mean_model, color=colors(idx), label=f"{label} Mean")
        ax.set_xlabel("Value (standardized)")
        ax.set_ylabel("Density (mean across leads)")
        ax.set_title(base.replace(".npz", ""))
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / base.replace(".npz", "_compare.png"), bbox_inches="tight")
        plt.close(fig)
