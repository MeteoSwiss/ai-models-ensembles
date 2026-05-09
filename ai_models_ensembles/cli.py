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


def _model_dir() -> Path:
    return Path(_env("OUTPUT_DIR", "")) / _env("DATE_TIME", "") / _env("MODEL_NAME", "")


def _perturbation_dir() -> Path:
    return (
        _model_dir()
        / f"init_{_env('PERTURBATION_INIT', '0.0')}_latent_{_env('PERTURBATION_LATENT', '0.0')}_layer_{_env('LAYER', '0')}"
    )


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
        help=(
            "Model registry name (see `ai-ens models`). " "Defaults to $MODEL_NAME from config.sh."
        ),
    ),
    init: Optional[str] = typer.Option(
        None,
        "--init",
        help=(
            "Initialization time, ISO-8601 (e.g. 2023-01-02T00). "
            "Defaults to $DATE_TIME (YYYYMMDDHHMM) from config.sh."
        ),
    ),
    lead_hours: int = typer.Option(
        240, "--lead-hours", "-L", help="Total forecast lead time in hours."
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output zarr path. Defaults to $PERTURBATION_DIR/forecast.zarr.",
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
            "or 'all' (default). Read from $LAYER if unset."
        ),
    ),
    data_source: str = typer.Option(
        "arco",
        "--data-source",
        help="ARCO | CDS | GFS | IFS | IFS_ENS | WB2 | file:PATH (case-insensitive).",
    ),
    output_levels: str = typer.Option(
        "100,500,850",
        "--output-levels",
        help=(
            "Comma-separated pressure levels (hPa) kept in the forecast zarr. "
            "Set to 'all' to keep every level the model emits. Reads $OUTPUT_LEVELS if set."
        ),
    ),
    seed: int = typer.Option(0, "--seed", help="RNG seed (member id is added to it)."),
) -> None:
    """Run inference + write a SwissClim-format zarr."""
    model_name = model or _env("MODEL_NAME") or ""
    if not model_name:
        raise typer.BadParameter("--model not given and $MODEL_NAME unset.")
    if model_name not in REGISTRY:
        raise typer.BadParameter(
            f"Unknown model '{model_name}'. Available: {sorted(REGISTRY.keys())}"
        )

    if init:
        init_time = datetime.fromisoformat(init)
    else:
        date_time = _env("DATE_TIME")
        if not date_time:
            raise typer.BadParameter("--init not given and $DATE_TIME unset.")
        init_time = datetime.strptime(date_time, "%Y%m%d%H%M")

    if output:
        out_path = Path(output)
    else:
        out_path = _perturbation_dir() / "forecast.zarr"

    if layer is None:
        env_layer = _env("LAYER")
        layer = env_layer if env_layer not in (None, "", "None") else None
    if ic_magnitude == 0.0:
        env_ic = _env("PERTURBATION_INIT")
        if env_ic is not None:
            ic_magnitude = float(env_ic)
    if weight_magnitude == 0.0:
        env_w = _env("PERTURBATION_LATENT")
        if env_w is not None:
            weight_magnitude = float(env_w)
    if members == 1:
        env_n = _env("NUM_MEMBERS")
        if env_n is not None:
            members = int(env_n)

    env_levels = _env("OUTPUT_LEVELS")
    if env_levels is not None and output_levels == "100,500,850":
        output_levels = env_levels
    levels_list: list[int] | None
    if output_levels.strip().lower() in {"", "all"}:
        levels_list = None
    else:
        levels_list = [int(x) for x in output_levels.split(",") if x.strip()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"Inference: model={model_name} init={init_time.isoformat()} "
        f"lead={lead_hours}h members={members} ic={ic_magnitude} "
        f"weight={weight_magnitude} layer={layer} source={data_source} "
        f"levels={levels_list or 'all'}"
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
    )
    typer.echo("*****DONE*****")


@app.command("verify")
def cli_verify(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Path to a SwissClim Evaluations YAML config. Defaults to "
            "$SWISSCLIM_CONFIG, then config/swissclim_eval.yaml in the repo."
        ),
    ),
) -> None:
    """Run verification via SwissClim Evaluations.

    Thin wrapper around `swissclim-evaluations --config <yaml>`. The YAML is
    the source of truth for paths, variables, modules, and metrics.
    """
    cfg_path: Optional[Path] = None
    if config:
        cfg_path = Path(config)
    elif _env("SWISSCLIM_CONFIG"):
        cfg_path = Path(_env("SWISSCLIM_CONFIG", ""))
    else:
        default = Path(__file__).resolve().parent.parent / "config" / "swissclim_eval.yaml"
        if default.exists():
            cfg_path = default

    if cfg_path is None or not cfg_path.exists():
        raise typer.BadParameter(
            "No SwissClim config found. Pass --config <yaml> or set $SWISSCLIM_CONFIG."
        )

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
        None,
        help=(
            "SwissClim output_root directories (or a glob pattern) to compare. "
            "If omitted, recursively finds every 'swissclim_*' directory under "
            "'$OUTPUT_DIR/$DATE_TIME' (covers all AI-model perturbation runs "
            "plus the _ifs_ens baseline)."
        ),
    ),
    labels: Optional[List[str]] = typer.Option(
        None, "--label", "-l", help="Display label per path (repeat option)."
    ),
    label_from: str = typer.Option(
        "leaf",
        "--label-from",
        help=(
            "How to derive labels when --label is not given: "
            "'leaf' (default; uses 'swissclim_<name>'), "
            "'parent' (perturbation/region dir), 'tail2' (last two)."
        ),
    ),
    out_dir: Optional[str] = typer.Option(
        None,
        "--out-dir",
        help="Where to write comparison outputs. Defaults to <first path>/intercomparison_<N>.",
    ),
    modules: List[str] = typer.Option(
        ["maps", "hist", "kde", "spectra", "vprof", "metrics", "ets", "prob", "multivariate"],
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
    if paths:
        for p in paths:
            if any(c in p for c in "*?["):
                discovered.extend(sorted(glob.glob(p)))
            else:
                discovered.append(p)
    else:
        out = _env("OUTPUT_DIR")
        dt = _env("DATE_TIME")
        if not (out and dt):
            raise typer.BadParameter("No paths given and $OUTPUT_DIR/$DATE_TIME are not both set.")
        # Recursive glob picks up:
        #   $OUTPUT_DIR/$DATE_TIME/<model>/init_*_latent_*_layer_*/<region>/swissclim_*
        #   $OUTPUT_DIR/$DATE_TIME/_ifs_ens/<region>/swissclim_ifs_ens
        discovered = sorted(
            str(p) for p in Path(out).joinpath(dt).rglob("swissclim_*") if p.is_dir()
        )

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
        elif label_from == "tail2":
            final_labels = [f"{Path(p).parent.name}/{Path(p).name}" for p in discovered]
        else:
            final_labels = [Path(p).parent.parent.name for p in discovered]

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
