from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

import typer
from rich.traceback import install as rich_traceback_install

from .convert_grib_to_zarr import convert_grib_to_zarr
from .download_ifs_control import download_ifs_control
from .download_ifs_ensemble import download_ifs_ensemble
from .download_re_analysis import download_re_analysis

rich_traceback_install(show_locals=False)
app = typer.Typer(help="AI Models Ensembles CLI")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name, default)
    return None if val is None else str(val)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_basic_config() -> None:
    out = _env("OUTPUT_DIR")
    date_time = _env("DATE_TIME")
    model = _env("MODEL_NAME")
    if not out:
        raise typer.Abort()
    if not date_time or not re.fullmatch(r"\d{12}", date_time):
        raise typer.BadParameter("DATE_TIME must be YYYYMMDDHHMM")
    if model not in {"graphcast", "fourcastnetv2-small", "gencast"}:
        raise typer.BadParameter(f"Unknown MODEL_NAME: {model}")


def _model_dir() -> Path:
    return Path(_env("OUTPUT_DIR", "")) / _env("DATE_TIME", "") / _env("MODEL_NAME", "")


def _perturbation_dir() -> Path:
    return (
        _model_dir()
        / f"init_{_env('PERTURBATION_INIT', '0.0')}_latent_{_env('PERTURBATION_LATENT', '0.0')}_layer_{_env('LAYER', '0')}"
    )


def _exists(p: Path) -> bool:
    return p.exists()


def _symlink(src: Path, dst: Path) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    except FileNotFoundError:
        pass
    dst.symlink_to(src)


@app.command("download-reanalysis")
def cli_download_reanalysis(
    out_dir: Optional[str] = typer.Option(None, help="Output dir (defaults to $OUTPUT_DIR)"),
    start: Optional[str] = typer.Option(None, help="Start (YYYYMMDDHHMM); defaults to $DATE_TIME"),
    end: Optional[str] = typer.Option(None, help="End (YYYYMMDDHHMM); defaults to $END_DATE_TIME"),
    interval: Optional[int] = typer.Option(
        None, help="Hours between analyses; defaults to $INTERVAL"
    ),
    model: Optional[str] = typer.Option(None, help="Model name; defaults to $MODEL_NAME"),
) -> None:
    _validate_basic_config()
    out = out_dir or _env("OUTPUT_DIR") or ""
    start_dt = start or _env("DATE_TIME") or ""
    end_dt = end or _env("END_DATE_TIME") or ""
    inter = int(interval or int(_env("INTERVAL", "6") or 6))
    mdl = model or _env("MODEL_NAME") or "graphcast"
    download_re_analysis(out, start_dt, end_dt, inter, mdl)


@app.command("download-ifs-ensemble")
def cli_download_ifs_ensemble(
    out_dir: Optional[str] = typer.Option(None, help="Output dir (defaults to $OUTPUT_DIR)"),
    date_time: Optional[str] = typer.Option(None, help="YYYYMMDDHHMM; defaults to $DATE_TIME"),
    interval: Optional[int] = typer.Option(None, help="Hours between steps; defaults to $INTERVAL"),
    num_days: Optional[int] = typer.Option(None, help="Days to download; defaults to $NUM_DAYS"),
    model: Optional[str] = typer.Option(None, help="Model name; defaults to $MODEL_NAME"),
) -> None:
    _validate_basic_config()
    out = out_dir or _env("OUTPUT_DIR") or ""
    dt = date_time or _env("DATE_TIME") or ""
    inter = int(interval or int(_env("INTERVAL", "6") or 6))
    days = int(num_days or int(_env("NUM_DAYS", "10") or 10))
    mdl = model or _env("MODEL_NAME") or "graphcast"
    download_ifs_ensemble(out, dt, inter, days, mdl)


@app.command("download-ifs-control")
def cli_download_ifs_control(
    out_dir: Optional[str] = typer.Option(None, help="Output dir (defaults to $OUTPUT_DIR)"),
    date_time: Optional[str] = typer.Option(None, help="YYYYMMDDHHMM; defaults to $DATE_TIME"),
    interval: Optional[int] = typer.Option(None, help="Hours between steps; defaults to $INTERVAL"),
    num_days: Optional[int] = typer.Option(None, help="Days to download; defaults to $NUM_DAYS"),
    model: Optional[str] = typer.Option(None, help="Model name; defaults to $MODEL_NAME"),
) -> None:
    _validate_basic_config()
    out = out_dir or _env("OUTPUT_DIR") or ""
    dt = date_time or _env("DATE_TIME") or ""
    inter = int(interval or int(_env("INTERVAL", "6") or 6))
    days = int(num_days or int(_env("NUM_DAYS", "10") or 10))
    mdl = model or _env("MODEL_NAME") or "graphcast"
    download_ifs_control(out, dt, inter, days, mdl)


@app.command("convert")
def cli_convert(
    path: str = typer.Option(..., help="Path to model or perturbation dir"),
    subdir_search: bool = typer.Option(False, help="Search subdirs for GRIB files"),
) -> None:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"Path not found: {p}")
    convert_grib_to_zarr(str(p), subdir_search=subdir_search)


@app.command("infer")
def cli_infer(
    member: Optional[int] = typer.Option(None, help="Run only this member id (array mode)"),
) -> None:
    _validate_basic_config()
    out_dir = Path(_env("OUTPUT_DIR", ""))
    model_name = _env("MODEL_NAME", "")
    num_members = int(_env("NUM_MEMBERS", "1"))
    lead_time = int(_env("LEAD_TIME", "24"))
    perturb_init = float(_env("PERTURBATION_INIT", "0.0"))
    perturb_latent = float(_env("PERTURBATION_LATENT", "0.0"))

    model_dir = _model_dir()
    pert_dir = _perturbation_dir()
    is_graphcast = model_name == "graphcast"
    is_fourcast = model_name == "fourcastnetv2-small"
    is_gencast = model_name == "gencast"
    effective_perturb_latent = 0.0 if is_gencast else perturb_latent
    if is_gencast and perturb_latent > 0.0:
        typer.echo(
            "Latent perturbations are ignored for probabilistic model 'gencast'.",
            err=True,
        )
    _ensure_dir(pert_dir)

    members = [int(member)] if member is not None else list(range(num_members))
    for m in members:
        member_dir = pert_dir / str(m)
        _ensure_dir(member_dir)

        # Initial condition perturbation
        init_grib = member_dir / "init_field.grib"
        if perturb_init > 0.0:
            if not _exists(init_grib):
                import subprocess

                subprocess.run(
                    [
                        "python",
                        "-u",
                        "-m",
                        "ai_models_ensembles.perturb_era5",
                        str(out_dir),
                        _env("DATE_TIME", ""),
                        model_name,
                        str(perturb_init),
                        str(perturb_latent),
                        str(m),
                    ],
                    check=True,
                )
        else:
            _symlink(model_dir / "init_field.grib", init_grib)

        # Latent/weights perturbation
        if effective_perturb_latent > 0.0:
            if is_graphcast:
                params_dir = member_dir / "params"
                if not params_dir.exists():
                    import subprocess

                    subprocess.run(
                        [
                            "python",
                            "-u",
                            "-m",
                            "ai_models_ensembles.perturb_graphcast_weights",
                            str(out_dir),
                            _env("DATE_TIME", ""),
                            model_name,
                            str(perturb_init),
                            str(perturb_latent),
                            str(m),
                            _env("LAYER", "0"),
                        ],
                        check=True,
                    )
            elif is_fourcast:
                weights_tar = member_dir / "weights.tar"
                if not _exists(weights_tar):
                    import subprocess

                    subprocess.run(
                        [
                            "python",
                            "-u",
                            "-m",
                            "ai_models_ensembles.perturb_fourcastnet_weights",
                            str(out_dir),
                            _env("DATE_TIME", ""),
                            model_name,
                            str(perturb_init),
                            str(perturb_latent),
                            str(m),
                            _env("LAYER", "0"),
                        ],
                        check=True,
                    )
        else:
            if is_graphcast:
                _symlink(model_dir / "params", member_dir / "params")
            elif is_fourcast:
                _symlink(model_dir / "weights.tar", member_dir / "weights.tar")

        # Common assets
        if is_graphcast:
            _symlink(model_dir / "stats", member_dir / "stats")
        elif is_fourcast:
            _symlink(model_dir / "global_means.npy", member_dir / "global_means.npy")
            _symlink(model_dir / "global_stds.npy", member_dir / "global_stds.npy")
        elif is_gencast:
            assets_src = model_dir / "assets"
            if not assets_src.exists():
                raise typer.BadParameter(
                    "GenCast assets not found. Run ai-models with --download-assets gencast first."
                )
            _symlink(assets_src, member_dir / "assets")
            fields_src = model_dir / "fields.txt"
            if fields_src.exists():
                _symlink(fields_src, member_dir / "fields.txt")

        # Run the model
        if not (pert_dir / f"forecast.zarr/member/{m}").exists():
            import subprocess

            cmd = [
                "ai-models",
                "--input",
                "file",
                "--file",
                str(init_grib),
                "--lead-time",
                str(lead_time),
            ]
            if is_gencast:
                cmd.extend([
                    "--num-ensemble-members",
                    "1",
                    "--member-number",
                    str(m + 1),
                ])
            cmd.append(model_name)

            subprocess.run(cmd, cwd=member_dir, check=True)

    typer.echo("*****DONE*****")


@app.command("verify")
def cli_verify() -> None:
    _validate_basic_config()
    out_dir = _env("OUTPUT_DIR", "")
    dt = _env("DATE_TIME", "")
    model_name = _env("MODEL_NAME", "")
    p_init = _env("PERTURBATION_INIT", "0.0")
    p_latent = _env("PERTURBATION_LATENT", "0.0")
    layer = _env("LAYER", "0")
    n_members = _env("NUM_MEMBERS", "1")
    region = _env("CROP_REGION", "europe")

    region_dir = _perturbation_dir() / region
    _ensure_dir(region_dir)

    # Imports for plotting/animations as library calls (no argparse)
    from types import SimpleNamespace

    import seaborn as sns

    import ai_models_ensembles.animate_2d_maps as a2
    import ai_models_ensembles.animate_3d_grids as a3
    import ai_models_ensembles.plot_0d_distributions as p0
    import ai_models_ensembles.plot_1d_timeseries as p1

    from .preprocess_data import calculate_stats, calculate_y_lims, load_and_prepare_data

    # Load data
    path_in = str(_model_dir())
    data = load_and_prepare_data(
        path_in,
        ["u10"],  # default selection from original config
        region,
        model_name,
        float(p_init),
        float(p_latent),
        int(layer),
        int(n_members),
        debug_mode=False,
    )

    # 0D density plots
    png_dir = region_dir / f"png_{model_name}"
    data_dir = region_dir / f"artifacts_{model_name}"
    if not png_dir.exists():
        typer.echo("Evaluating model and creating 0D and 1D figures")
        png_dir.mkdir(parents=True, exist_ok=True)
        variables = list(data["forecast"].data_vars)
        vars_3d = [var for var in variables if "isobaricInhPa" in data["forecast"][var].dims]
        vars_2d = [var for var in variables if "isobaricInhPa" not in data["forecast"][var].dims]
        config = {
            "color_palette": sns.color_palette([
                "#f75b78",
                "#6495ed",
                "#0e2d75",
                "#f9c740",
                "#45b7aa",
                "#353434",
            ]),
            "sample_size": 100000,
            "selected_vars": ["u10"],
            "output_mode": "both",
            "artifact_root": str(data_dir),
        }
        # Individual density plots
        args_list_ind = p0.prepare_density_distribution_args(
            data,
            vars_3d,
            vars_2d,
            config,
            str(png_dir),
            model_name,
            region,
            dt,
            max_samples=config["sample_size"],
            combined=False,
            artifact_root=str(data_dir),
            output_mode=config["output_mode"],
        )
        for plot_args in args_list_ind:
            p0.plot_density_distribution(**plot_args)
        # Combined density plots
        args_list_comb = p0.prepare_density_distribution_args(
            data,
            vars_3d,
            vars_2d,
            config,
            str(png_dir),
            model_name,
            region,
            dt,
            max_samples=config["sample_size"],
            max_variables=5,
            combined=True,
            artifact_root=str(data_dir),
            output_mode=config["output_mode"],
        )
        for plot_args in args_list_comb:
            p0.plot_combined_density_distribution(**plot_args)

        # 1D timeseries and scorecards
        default_stats = calculate_stats(
            data["ground_truth"],
            data["forecast"],
            data["forecast_unperturbed"],
            region,
        )
        ifs_stats = calculate_stats(
            data["ground_truth"],
            data["forecast_ifs"],
            data["forecast_ifs_unperturbed"],
            region,
        )
        # alpha_value is used inside plot_1d_timeseries.prepare_plot_args
        p1.alpha_value = 1 / data["forecast"].member.size ** (5 / 8)
        y_lims = calculate_y_lims(
            vars_3d, vars_2d, data["forecast"], data["forecast_ifs"], default_stats, ifs_stats
        )
        default_plot_args = p1.prepare_plot_args(
            data,
            default_stats,
            vars_3d,
            vars_2d,
            config,
            **y_lims,
            use_ifs=False,
            path_out=str(png_dir),
            model_name=model_name.title(),
            region=region.title(),
            date_time=dt,
            output_mode=config["output_mode"],
            artifact_root=str(data_dir),
        )
        png_dir_ifs = region_dir / "png_ifs"
        data_dir_ifs = region_dir / "artifacts_ifs"
        png_dir_ifs.mkdir(parents=True, exist_ok=True)
        (png_dir / "scorecards").mkdir(parents=True, exist_ok=True)
        (png_dir_ifs / "scorecards").mkdir(parents=True, exist_ok=True)
        ifs_plot_args = p1.prepare_plot_args(
            data,
            ifs_stats,
            vars_3d,
            vars_2d,
            config,
            **y_lims,
            use_ifs=True,
            path_out=str(png_dir_ifs),
            model_name="IFS ENS",
            region=region.title(),
            date_time=dt,
            output_mode=config["output_mode"],
            artifact_root=str(data_dir_ifs),
        )
        # Render plots
        for plot_args in [default_plot_args, ifs_plot_args]:
            for args_i in plot_args["energy_spectra"]:
                p1.plot_energy_spectra(**args_i)
            for args_i in plot_args["rank_histogram"]:
                p1.plot_rank_histogram(**args_i)
            for args_i in plot_args["rmse"]:
                p1.plot_rmse(**args_i)
            for args_i in plot_args["timeseries_fc_gt"]:
                p1.plot_timeseries_fc_gt(**args_i)
            # one scorecard per set
            p1.plot_error_map(**plot_args["error_map"][0])

        # Combined spread-skill ratio plots using both IFS and model
        for args_dict in default_plot_args["spread_skill_ratio"]:
            args_dict = args_dict.copy()
            args_dict["sr_spread_skill_ratio_ifs"] = ifs_stats["sr_spread_skill_ratio"]
            args_dict["sr_ensemble_spread_ifs"] = ifs_stats["sr_ensemble_spread"]
            args_dict["model_names"] = [args_dict.pop("model_name"), "IFS ENS"]
            p1.plot_spread_skill_ratio(**args_dict)

    # Animations (2D maps + ensemble metrics) using library functions
    anim_dir = region_dir / "0/animations"
    gifs_exist = any(anim_dir.rglob("*.gif")) if anim_dir.exists() else False
    if not gifs_exist:
        typer.echo("Generating 2D and 3D animations")
        # Build a lightweight args namespace expected by animation helpers
        args_ns = SimpleNamespace(
            out_dir=out_dir,
            date_time=dt,
            model_name=model_name,
            perturbation_init=float(p_init),
            perturbation_latent=float(p_latent),
            layer=int(layer),
            crop_region=region,
            members=int(n_members),
            debug=False,
        )
        # 2D animations
        path_forecast = str(_perturbation_dir())
        lat = data["ground_truth"].latitude.values
        lon = data["ground_truth"].longitude.values
        stats_default = calculate_stats(
            data["ground_truth"], data["forecast"], data["forecast_unperturbed"], region
        )
        for member in [0, 1]:
            a2.process_member(
                member,
                data["forecast"],
                data["ground_truth"],
                stats_default,
                path_forecast,
                lat,
                lon,
                args_ns,
            )
        a2.process_ensemble_metrics(
            data["forecast"], data["ground_truth"], stats_default, path_forecast, lat, lon, args_ns
        )

        # 3D difference animations (selected variables)
        config = {"selected_vars": ["u10"]}
        for member in [0, 1]:
            a3.process_member(
                member,
                data["forecast"],
                data["forecast_unperturbed"],
                path_forecast,
                args_ns,
                config,
            )

    # Cleanup GRIB files under region_dir
    typer.echo("Cleaning up GRIB files")
    if shutil.which("fd"):
        import subprocess

        subprocess.run(
            ["fd", "-IH", "--type", "f", ".grib", str(region_dir), "-x", "rm", "{}"], check=True
        )
    else:
        for grib in region_dir.rglob("*.grib"):
            try:
                grib.unlink()
            except FileNotFoundError:
                pass
    typer.echo("*****DONE*****")


@app.command("intercompare")
def cli_intercompare(
    model_dirs: List[str] = typer.Argument(
        ..., help="Artifact directories produced by verify runs."
    ),
    labels: Optional[List[str]] = typer.Option(
        None, "--label", "-l", help="Display labels (repeat per model)."
    ),
    out_dir: str = typer.Option(
        "intercomparison", "--out-dir", help="Directory for comparison figures."
    ),
    metrics: List[str] = typer.Option(
        ["energy_spectra", "rmse", "timeseries", "rank_histogram", "density"],
        "--metric",
        "-m",
        help="Metrics to compare (repeat option to select subset).",
    ),
) -> None:
    labels_final = labels or [Path(m).name for m in model_dirs]
    if len(labels_final) != len(model_dirs):
        raise typer.BadParameter("Number of labels must match number of model directories.")
    from .intercompare import run_intercompare

    run_intercompare(model_dirs, labels_final, out_dir, metrics)
    typer.echo("*****DONE*****")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
