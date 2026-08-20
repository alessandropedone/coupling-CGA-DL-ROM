"""Test the effect of the geometry parameters on the DL-ROM performance."""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from utils import compute_displacement_from_history


def replace_variable(text, variable, value):
    """Replace a scalar variable assignment in a Gmsh geometry template."""
    pattern = rf"(?m)^(\s*{re.escape(variable)}\s*=\s*)[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(\s*;)"
    updated_text, replacements = re.subn(
        pattern,
        lambda match: f"{match.group(1)}{value:.16g}{match.group(2)}",
        text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(f"Variable '{variable}' not found")
    return updated_text


def normalized_metrics(workdir, workdir_ref, clamped):
    """Compute speedup and normalized capacitance/displacement errors."""
    execution_time = pd.read_csv(workdir / "execution_time.csv")
    execution_time_ref = pd.read_csv(workdir_ref / "execution_time.csv")
    total_time = execution_time["total_s"].values[0]
    total_time_ref = execution_time_ref["total_s"].values[0]
    speedup = total_time_ref / total_time

    data = pd.read_csv(workdir / "modal_history.csv")
    data_ref = pd.read_csv(workdir_ref / "modal_history.csv")
    thickness_m = 10e-6
    capacity = thickness_m * data["cap_like_F_approx"].values
    capacity_ref = thickness_m * data_ref["cap_like_F"].values
    capacity = np.nan_to_num(capacity, nan=1e-30)
    capacity_ref = np.nan_to_num(capacity_ref, nan=1e-30)
    capacity_range = capacity_ref[1:].max() - capacity_ref[1:].min()
    if capacity_range == 0:
        raise ValueError("The classical ROM capacitance range is zero")
    capacity_error = np.max(np.abs(capacity[1:] - capacity_ref[1:]))
    normalized_capacity_error = capacity_error / capacity_range

    L_m = 1e-4
    x = np.linspace(0, L_m, 100)
    displacement = compute_displacement_from_history(
        4, x, L_m, workdir, clamped=clamped
    )
    displacement_ref = compute_displacement_from_history(
        4, x, L_m, workdir_ref, clamped=clamped
    )
    displacement_diff_linf = np.max(
        np.abs(displacement - displacement_ref), axis=1
    )
    displacement_ref_linf = np.max(np.abs(displacement_ref), axis=1)
    displacement_range = (
        displacement_ref_linf.max() - displacement_ref_linf.min()
    )
    if displacement_range == 0:
        raise ValueError("The classical ROM displacement range is zero")
    normalized_displacement_error = (
        displacement_diff_linf.max() / displacement_range
    )

    return speedup, normalized_capacity_error, normalized_displacement_error


def main():
    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-simulation", action="store_true")
    ap.add_argument("--big-deformation", action="store_true")
    ap.add_argument("--accurate-capacity", action="store_true")
    ap.add_argument("--clamped", action="store_true")
    ap.add_argument(
        "--grid-refinement",
        type=int,
        default=3,
        help="Number of equally spaced values for each geometry parameter",
    )
    args = ap.parse_args()

    if args.big_deformation and args.clamped:
        raise ValueError(
            "The --big-deformation and --clamped options cannot be used together, as the current clamped beam model is not designed for large deformations."
        )
    if args.grid_refinement < 2:
        raise ValueError("--grid-refinement must be at least 2")

    overetch_values = np.linspace(0.0, 0.5, args.grid_refinement)
    distance_values = np.linspace(
        20.0 if args.big_deformation else 1.5,
        30.0 if args.big_deformation else 2.5,
        args.grid_refinement,
    )
    metrics = np.empty((args.grid_refinement, args.grid_refinement, 3))
    rows = []

    template_geo = Path(
        "geometries/cantilever1.geo"
        if not args.big_deformation and not args.clamped
        else (
            "geometries/clamped.geo"
            if args.clamped
            else "geometries/cantilever2.geo"
        )
    )
    original_template_geo_text = template_geo.read_text()

    geometry_case = (
        "big_deformation"
        if args.big_deformation
        else "clamped" if args.clamped else "cantilever"
    )
    output_dir = Path("temp/geometry") / geometry_case

    try:
        for i, overetch in enumerate(overetch_values):
            for j, distance in enumerate(distance_values):
                case_name = f"overetch_{overetch:.6g}_distance_{distance:.6g}"
                workdir_ref = output_dir / case_name / "classical_rom"
                workdir = output_dir / case_name / "dl_rom"

                template_geo_text = replace_variable(
                    original_template_geo_text, "overetch", overetch
                )
                template_geo_text = replace_variable(
                    template_geo_text, "distance", distance
                )
                template_geo.write_text(template_geo_text)

                # ---------------------------------
                # Run the classical ROM simulation
                # ---------------------------------
                cmd = [
                    "python",
                    "-m",
                    "src.multi_physics.solver",
                    "--template-geo",
                    (
                        "geometries/cantilever1.geo"
                        if not args.big_deformation and not args.clamped
                        else (
                            "geometries/clamped.geo"
                            if args.clamped
                            else "geometries/cantilever2.geo"
                        )
                    ),
                    "--workdir",
                    str(workdir_ref),
                    "--nmodes",
                    "4",
                    "--dt",
                    "1e-5",
                    "--nsteps",
                    "40",
                    "--Vdc",
                    "0",
                    "--Vac",
                    "5" if not args.big_deformation else "230",
                    "--freq",
                    "2.5e3",
                    "--Vupper",
                    "0",
                    "--Vouter",
                    "0",
                    "--omega",
                    "6.3e5",
                    "3.9e6",
                    "1.1e7",
                    "2.1e7",
                    "--mass",
                    "1e-12",
                    "1e-12",
                    "1e-12",
                    "1e-12",
                    "--zeta",
                    "0.01",
                    "0.01",
                    "0.01",
                    "0.01",
                    "--print-every",
                    "1",
                    "--fail-fast",
                    "--no-outer-bc",
                ]
                if args.clamped:
                    cmd.append("--clamped")
                if not args.no_simulation:
                    print(
                        f"Running classical ROM with overetch = {overetch:.3g} and distance = {distance:.3g}..."
                    )
                    subprocess.run(cmd, check=True)

                # ----------------------------
                # Run the DL-ROM simulation
                # ----------------------------
                cmd = [
                    "python",
                    "-m",
                    "src.multi_physics.solver",
                    "--template-geo",
                    (
                        "geometries/cantilever1.geo"
                        if not args.big_deformation and not args.clamped
                        else (
                            "geometries/clamped.geo"
                            if args.clamped
                            else "geometries/cantilever2.geo"
                        )
                    ),
                    "--workdir",
                    str(workdir),
                    "--nmodes",
                    "4",
                    "--dt",
                    "1e-5",
                    "--nsteps",
                    "40",
                    "--Vdc",
                    "0",
                    "--Vac",
                    "5" if not args.big_deformation else "230",
                    "--freq",
                    "2.5e3",
                    "--Vupper",
                    "0",
                    "--Vouter",
                    "0",
                    "--omega",
                    "6.3e5",
                    "3.9e6",
                    "1.1e7",
                    "2.1e7",
                    "--mass",
                    "1e-12",
                    "1e-12",
                    "1e-12",
                    "1e-12",
                    "--zeta",
                    "0.01",
                    "0.01",
                    "0.01",
                    "0.01",
                    "--print-every",
                    "1",
                    "--fail-fast",
                    "--no-outer-bc",
                    "--derivative-nn-path",
                    (
                        "models/derivative1.keras"
                        if not args.big_deformation and not args.clamped
                        else (
                            "models/derivative3.keras"
                            if args.clamped
                            else "models/derivative2.keras"
                        )
                    ),
                ]
                if not args.accurate_capacity:
                    cmd.append("--no-postprocessing")
                else:
                    cmd.extend(["--postprocessing-step", "5"])
                if args.clamped:
                    cmd.append("--clamped")
                if not args.no_simulation:
                    print(
                        f"Running DL-ROM with overetch = {overetch:.3g} and distance = {distance:.3g}..."
                    )
                    subprocess.run(cmd, check=True)

                metric = normalized_metrics(workdir, workdir_ref, args.clamped)
                metrics[i, j] = metric
                rows.append(
                    {
                        "overetch": overetch,
                        "distance": distance,
                        "speedup": metric[0],
                        "normalized_capacity_error": metric[1],
                        "normalized_displacement_error": metric[2],
                    }
                )
    finally:
        # Do not leave a project geometry at the last point of the sweep.
        template_geo.write_text(original_template_geo_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "metrics.npy", metrics)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)

    metric_names = (
        "Speedup",
        "Normalized capacitance Linf error",
        "Normalized displacement Linf error",
    )
    print("")
    print("Geometry test summary")
    print("=" * 75)
    print(f"{'Metric':45s} {'Mean':>13s} {'Std. dev.':>13s}")
    print("-" * 75)
    for index, metric_name in enumerate(metric_names):
        print(
            f"{metric_name:45s} {metrics[:, :, index].mean():13.3e} "
            f"{metrics[:, :, index].std():13.3e}"
        )


if __name__ == "__main__":
    main()
