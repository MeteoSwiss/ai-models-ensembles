from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich.traceback import install as rich_traceback_install

from .e2s_inference import run_inference
from .e2s_models import REGISTRY

rich_traceback_install(show_locals=False)
app = typer.Typer(
    help="ai-models-ensembles: earth2studio-backed inference + SwissClim verification"
)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name, default)
    return None if val is None else str(val)


@app.command("models")
def cli_models() -> None:
    """List available earth2studio models."""
    for name, spec in sorted(REGISTRY.items()):
        kind = "probabilistic" if spec.probabilistic else "deterministic"
        typer.echo(f"  {name:<22} step={spec.step_hours}h  {kind:<14} {spec.description}")


@app.command("infer")
def cli_infer(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model registry name (see `ai-ens models`).",
    ),
    init: str = typer.Option(
        ...,
        "--init",
        help="Initialization time, ISO-8601 (e.g. 2023-01-02T00:00).",
    ),
    lead_hours: int = typer.Option(
        240, "--lead-hours", "-L", help="Total forecast lead time in hours."
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output zarr path.",
    ),
    members: int = typer.Option(1, "--members", "-n", help="Number of ensemble members."),
    ic_magnitude: float = typer.Option(
        0.0, "--ic-magnitude", help="Std-dev of multiplicative IC noise (0 disables)."
    ),
    weight_magnitude: float = typer.Option(
        0.0,
        "--weight-magnitude",
        help="Std-dev of multiplicative weight noise (0 disables).",
    ),
    layer: Optional[str] = typer.Option(
        None,
        "--layer",
        help=(
            "Layer spec: single index '42', range '10:50', fraction '0.0:0.33', "
            "or 'all' (default)."
        ),
    ),
    coarse_mode_cut: Optional[int] = typer.Option(
        None,
        "--coarse-mode-cut",
        help=(
            "Phase 3 SFNO only. Restrict perturbation to the first N entries of "
            "the last axis of `*.filter.filter.weight` tensors (spectral conv "
            "weights). Other tensors are passed through. For SFNO 73ch_small, "
            "last axis = 240 latitudinal modes; N=10 targets wavelengths "
            ">= 4000 km (planetary / large-synoptic)."
        ),
    ),
    graph_coarse_sigma: float = typer.Option(
        0.0,
        "--graph-coarse-sigma",
        help=(
            "Phase 3 GraphCast only. Activation perturbation magnitude on the "
            "first --graph-coarse-nodes mesh-node latent features (after the "
            "grid2mesh encoder, before the mesh GNN). 0 disables."
        ),
    ),
    graph_coarse_nodes: int = typer.Option(
        42,
        "--graph-coarse-nodes",
        help=(
            "Phase 3 GraphCast only. Number of leading mesh nodes to perturb. "
            "12 = level-0 only (~6700 km), 42 = level-0+1 (~3300 km), "
            "162 = level-0+1+2 (~1700 km)."
        ),
    ),
    data_source: str = typer.Option(
        "arco",
        "--data-source",
        help="ARCO | CDS | GFS | IFS | IFS_ENS | WB2 | file:PATH (case-insensitive).",
    ),
    output_levels: str = typer.Option(
        "500,850",
        "--output-levels",
        help=(
            "Comma-separated pressure levels (hPa) kept in the forecast zarr. "
            "Set to 'all' to keep every level the model emits."
        ),
    ),
    output_vars: str = typer.Option(
        "all",
        "--output-vars",
        help=(
            "Comma-separated ECMWF long variable names kept in the forecast zarr. "
            "Set to 'all' to keep every variable the model emits."
        ),
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed (member id is added to it)."),
) -> None:
    """Run inference + write a SwissClim-format zarr."""
    model_name = model or ""
    if not model_name:
        raise typer.BadParameter("--model is required.")
    if model_name not in REGISTRY:
        raise typer.BadParameter(
            f"Unknown model '{model_name}'. Available: {sorted(REGISTRY.keys())}"
        )

    init_time = datetime.fromisoformat(init)
    out_path = Path(output)

    levels_list: list[int] | None
    if output_levels.strip().lower() in {"", "all"}:
        levels_list = None
    else:
        levels_list = [int(x) for x in output_levels.split(",") if x.strip()]

    vars_list: list[str] | None
    if output_vars.strip().lower() in {"", "all"}:
        vars_list = None
    else:
        vars_list = [x.strip() for x in output_vars.split(",") if x.strip()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"Inference: model={model_name} init={init_time.isoformat()} "
        f"lead={lead_hours}h members={members} ic={ic_magnitude} "
        f"weight={weight_magnitude} layer={layer} "
        f"coarse_mode_cut={coarse_mode_cut} "
        f"graph_coarse_sigma={graph_coarse_sigma} "
        f"graph_coarse_nodes={graph_coarse_nodes} source={data_source} "
        f"levels={levels_list or 'all'} vars={vars_list or 'all'}"
    )
    run_inference(
        model_name=model_name,
        init_time=init_time,
        lead_hours=lead_hours,
        output=out_path,
        n_members=members,
        ic_magnitude=ic_magnitude,
        weight_magnitude=weight_magnitude,
        layer=layer,
        data_source=data_source,
        seed=seed,
        output_levels=levels_list,
        output_vars=vars_list,
        coarse_mode_cut=coarse_mode_cut,
        graph_coarse_sigma=graph_coarse_sigma,
        graph_coarse_nodes=graph_coarse_nodes,
    )
    typer.echo("*****DONE*****")


@app.command("verify")
def cli_verify(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a SwissClim Evaluations YAML config.",
    ),
) -> None:
    """Run verification via SwissClim Evaluations."""
    cfg_path = Path(config)
    if not cfg_path.exists():
        raise typer.BadParameter(f"Config not found: {cfg_path}")

    swissclim = shutil.which("swissclim-evaluations")
    cmd = (
        [swissclim, "--config", str(cfg_path)]
        if swissclim
        else ["python", "-u", "-m", "swissclim_evaluations.cli", "--config", str(cfg_path)]
    )
    typer.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    typer.echo("*****DONE*****")


@app.command("intercompare")
def cli_intercompare(
    paths: List[str] = typer.Argument(
        ...,
        help="SwissClim output_root directories (or glob patterns) to compare.",
    ),
    labels: Optional[List[str]] = typer.Option(
        None, "--label", "-l", help="Display label per path (repeat option)."
    ),
    label_from: str = typer.Option(
        "leaf",
        "--label-from",
        help=(
            "How to derive labels when --label is not given: "
            "'leaf' (default), 'parent', 'tail2' (last two)."
        ),
    ),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        help="Where to write comparison outputs. Defaults to <first path>/intercomparison_<N>.",
    ),
    modules: List[str] = typer.Option(
        [
            "maps",
            "hist",
            "kde",
            "spectra",
            "vprof",
            "metrics",
            "ets",
            "fss",
            "prob",
            "multivariate",
        ],
        "--module",
        "-m",
        help="Modules to run (repeat option).",
    ),
    config_out: Optional[str] = typer.Option(
        None,
        "--config-out",
        help="Write the rendered intercomparison YAML to this path.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render YAML + print command only."),
) -> None:
    """Run swissclim-evaluations-compare across a set of verify output_roots."""
    import glob

    discovered: List[str] = []
    for p in paths:
        if any(c in p for c in "*?["):
            discovered.extend(sorted(glob.glob(p)))
        else:
            discovered.append(p)

    discovered = [p for p in discovered if Path(p).is_dir()]
    if len(discovered) < 2:
        raise typer.BadParameter(
            f"Need at least 2 output_roots; found {len(discovered)}: {discovered}"
        )

    if labels:
        if len(labels) != len(discovered):
            raise typer.BadParameter(f"Got {len(labels)} labels for {len(discovered)} paths.")
        final_labels = list(labels)
    else:
        if label_from == "leaf":
            final_labels = [Path(p).name for p in discovered]
        elif label_from == "parent":
            final_labels = [Path(p).parent.name for p in discovered]
        elif label_from == "tail2":
            final_labels = [f"{Path(p).parent.name}/{Path(p).name}" for p in discovered]
        elif label_from == "grandparent":
            final_labels = [Path(p).parent.parent.name for p in discovered]
        else:
            raise typer.BadParameter(
                f"Unknown --label-from {label_from!r}; expected one of leaf, parent, tail2, grandparent."
            )

    if out_dir is None:
        out_dir = str(Path(discovered[0]).parent.parent / f"intercomparison_{len(discovered)}")

    config_dict = {
        "models": discovered,
        "labels": final_labels,
        "output_root": out_dir,
        "modules": list(modules),
    }

    if config_out:
        cfg_path = Path(config_out)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(config_dict, sort_keys=False))
    else:
        fd, tmp = tempfile.mkstemp(prefix="swissclim_intercompare_", suffix=".yaml")
        os.close(fd)
        cfg_path = Path(tmp)
        cfg_path.write_text(yaml.safe_dump(config_dict, sort_keys=False))

    typer.echo(f"Intercomparison config: {cfg_path}")
    for label, path in zip(final_labels, discovered):
        typer.echo(f"  - {label}: {path}")
    typer.echo(f"Output root: {out_dir}")

    compare = shutil.which("swissclim-evaluations-compare")
    cmd = (
        [compare, "--config", str(cfg_path)]
        if compare
        else ["python", "-u", "-m", "swissclim_evaluations.intercompare", "--config", str(cfg_path)]
    )
    typer.echo(f"Running: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)
    typer.echo("*****DONE*****")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
