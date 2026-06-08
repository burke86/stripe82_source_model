#!/usr/bin/env python3
"""Run the J230056.54-001711.1 ugri full-pipeline patch test.

The default science path uses a fast hierarchical joint model: every source
stamp is compressed to one Moffat matched-filter likelihood term, then standard
star calibration and the AGN light curve are inferred together.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


TARGET_ID = "J230056.54-001711.1"
TARGET_RA = 345.2355659034
TARGET_DEC = -0.2864342069
DEFAULT_WU_SHEN_CATALOG = Path(
    "/Users/colinburke/research/supercosmos_quasars/wu_qso.parquet"
)
DEFAULT_JAGUAR_SRC = Path("/Users/colinburke/research/jaguar/src")
SDSS_FILTERS = "ugri"
PS1_FILTERS = "gri"
ZTF_FILTERS = "zg,zr,zi"
UGRI_EQUIVALENT = {"u", "g", "r", "i", "zg", "zr", "zi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full J230056 ugri download, fast joint Moffat calibration, and plotting test."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/private/tmp/j230056_full_patch"))
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional pipeline log file. Default: OUT_DIR/pipeline.log.",
    )
    parser.add_argument("--target-id", default=TARGET_ID)
    parser.add_argument("--ra", type=float, default=TARGET_RA)
    parser.add_argument("--dec", type=float, default=TARGET_DEC)
    parser.add_argument("--wu-shen-catalog", type=Path, default=DEFAULT_WU_SHEN_CATALOG)
    parser.add_argument("--jaguar-src", type=Path, default=DEFAULT_JAGUAR_SRC)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subcommands.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--ztf-workers", type=int, default=8)
    parser.add_argument("--standard-radius-deg", type=float, default=0.20)
    parser.add_argument(
        "--photometry-max-stars-per-image",
        type=int,
        default=40,
        help="Cap standards per image for the fast joint model. Use 0 for no cap.",
    )
    parser.add_argument(
        "--photometry-max-images",
        type=int,
        help="Optional cap on images used for the joint fit. Downloads remain uncapped.",
    )
    parser.add_argument("--moffat-fwhm-pix", type=float, default=3.0)
    parser.add_argument("--moffat-beta", type=float, default=3.5)
    parser.add_argument("--host-sersic-n", type=float, default=1.0)
    parser.add_argument("--host-reff-pix", type=float, default=5.0)
    parser.add_argument("--host-axis-ratio", type=float, default=0.8)
    parser.add_argument("--host-pa-deg", type=float, default=0.0)
    parser.add_argument("--prior-host-mag-loc", type=float, default=22.0)
    parser.add_argument("--prior-host-mag-sigma", type=float, default=5.0)
    parser.add_argument("--ignore-header-seeing", action="store_true")
    parser.add_argument("--min-moffat-fwhm-pix", type=float, default=1.0)
    parser.add_argument("--max-moffat-fwhm-pix", type=float, default=12.0)
    parser.add_argument("--fix-image-fwhm", action="store_true")
    parser.add_argument("--fwhm-grid-size", type=int, default=7)
    parser.add_argument("--fwhm-grid-log-span", type=float, default=0.35)
    parser.add_argument("--prior-log-fwhm-sigma", type=float, default=0.25)
    parser.add_argument("--centroid-grid-size", type=int, default=5)
    parser.add_argument("--centroid-grid-radius-pix", type=float, default=1.0)
    parser.add_argument("--fix-centroids", action="store_true")
    parser.add_argument("--prior-centroid-sigma-pix", type=float, default=0.5)
    parser.add_argument("--prior-survey-filter-sigma", type=float, default=20.0)
    parser.add_argument("--prior-night-sigma", type=float, default=0.05)
    parser.add_argument("--prior-image-sigma", type=float, default=0.05)
    parser.add_argument("--prior-ccd-sigma", type=float, default=0.03)
    parser.add_argument("--prior-color-sigma", type=float, default=0.1)
    parser.add_argument("--prior-intrinsic-scatter-sigma", type=float, default=0.03)
    parser.add_argument("--initial-intrinsic-scatter", type=float, default=0.2)
    parser.add_argument("--prior-star-mag-floor", type=float, default=0.01)
    parser.add_argument("--student-t-df", type=float, default=5.0)
    parser.add_argument("--calibration-warmup", type=int, default=200)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument(
        "--reference-system",
        choices=["ps1", "native"],
        default="ps1",
        help="Reference system for standard-star calibration. Default: ps1.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run steps even when outputs exist.")
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Use existing manifests/images and run only manifest, joint inference, and figures.",
    )
    return parser.parse_args()


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def log_line(message: str, *, log_handle: TextIO | None = None) -> None:
    print(message, flush=True)
    if log_handle is not None:
        log_handle.write(message + "\n")
        log_handle.flush()


def run_command(
    label: str,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    log_handle: TextIO | None = None,
) -> None:
    log_line(f"\n==> {label}", log_handle=log_handle)
    log_line("$ " + " ".join(str(item) for item in cmd), log_handle=log_handle)
    started = time.monotonic()
    process = subprocess.Popen(
        [str(item) for item in cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip("\n")
        log_line(text, log_handle=log_handle)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, [str(item) for item in cmd])
    elapsed = time.monotonic() - started
    log_line(f"<== finished {label} in {elapsed / 60.0:.1f} min", log_handle=log_handle)


def should_run(path: Path, force: bool) -> bool:
    return force or not path.exists() or path.stat().st_size == 0


def csv_has_columns(path: Path, columns: set[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
    return columns <= fieldnames


def csv_data_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def remove_if_force(path: Path, force: bool) -> None:
    if force and path.exists():
        path.unlink()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_ugri_manifest(
    combined_manifest: Path,
    out_path: Path,
    *,
    max_images: int | None,
) -> None:
    rows = []
    for row in read_csv(combined_manifest):
        image_path = Path((row.get("image_path") or "").strip())
        filter_name = (row.get("filter") or "").strip()
        if filter_name not in UGRI_EQUIVALENT:
            continue
        if not image_path.exists():
            continue
        rows.append(row)
        if max_images is not None and len(rows) >= max_images:
            break
    if not rows:
        raise RuntimeError(
            f"No existing ugri-equivalent images found in {combined_manifest}; "
            "run without --skip-downloads or check download failures."
        )
    write_csv(out_path, rows, list(rows[0].keys()))
    print(f"wrote {len(rows)} existing image rows: {out_path}")


def main() -> int:
    args = parse_args()
    root = script_dir()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (out / "pipeline.log")

    target_catalog = out / f"{args.target_id}_catalog.csv"
    sdss_manifest = out / f"{args.target_id}_sdss_manifest.csv"
    ps1_manifest = out / f"{args.target_id}_panstarrs_raw_manifest.csv"
    ztf_manifest = out / f"{args.target_id}_ztf_raw_manifest_full.csv"
    combined_manifest = out / f"{args.target_id}_all_surveys_image_manifest_ugri.csv"
    existing_manifest = out / f"{args.target_id}_existing_manifest_ugri.csv"
    standards = out / f"{args.target_id}_ivezic_standards_r{str(args.standard_radius_deg).replace('.', 'p')}.csv"
    compressed_photometry = out / f"{args.target_id}_joint_moffat_compressed_ugri.csv"
    zeropoints = out / f"{args.target_id}_zeropoints_ugri.csv"
    standards_calibrated = out / f"{args.target_id}_standards_calibrated_ugri.csv"
    params = out / f"{args.target_id}_calibration_params_ugri.npz"
    agn_calibrated = out / f"{args.target_id}_agn_calibrated_ugri.csv"
    lightcurve = out / f"{args.target_id}_lightcurves_ugri.png"
    quality_diagnostics_dir = out / "quality_diagnostics"
    quality_diagnostics_summary = out / f"{args.target_id}_quality_diagnostics_ugri.csv"

    env = os.environ.copy()
    if args.jaguar_src:
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(args.jaguar_src)
            if not old_pythonpath
            else f"{args.jaguar_src}{os.pathsep}{old_pythonpath}"
        )
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("a", buffering=1) as log_handle:
        log_line(f"writing pipeline log: {log_path}", log_handle=log_handle)
        log_line(f"output directory: {out}", log_handle=log_handle)

        if not args.skip_downloads:
            prepare_cmd = [
                args.python,
                root / "prepare_agn_patch_test.py",
                "--target-id",
                args.target_id,
                "--ra",
                str(args.ra),
                "--dec",
                str(args.dec),
                "--filters",
                SDSS_FILTERS,
                "--out-dir",
                out,
                "--target-catalog",
                target_catalog,
                "--image-manifest",
                sdss_manifest,
                "--timeout",
                str(args.timeout),
            ]
            if args.wu_shen_catalog.exists():
                prepare_cmd.extend(["--wu-shen-catalog", args.wu_shen_catalog])
            if args.force or should_run(target_catalog, False) or should_run(sdss_manifest, False):
                remove_if_force(target_catalog, args.force)
                remove_if_force(sdss_manifest, args.force)
                run_command(
                    "download/query SDSS ugri corrected frames",
                    prepare_cmd,
                    env=env,
                    log_handle=log_handle,
                )
            else:
                log_line(f"==> using existing SDSS manifest: {sdss_manifest}", log_handle=log_handle)

            if should_run(ps1_manifest, args.force):
                remove_if_force(ps1_manifest, args.force)
                run_command(
                    "download/query Pan-STARRS gri warp images and masks/weights",
                    [
                        args.python,
                        root / "download_panstarrs_warps_stripe82.py",
                        "--positions-csv",
                        target_catalog,
                        "--filters",
                        PS1_FILTERS,
                        "--image-types",
                        "warp,warp.wt,warp.mask",
                        "--out-dir",
                        out / "panstarrs_warps",
                        "--manifest",
                        ps1_manifest,
                        "--timeout",
                        str(args.timeout),
                    ],
                    env=env,
                    log_handle=log_handle,
                )
            else:
                log_line(f"==> using existing Pan-STARRS manifest: {ps1_manifest}", log_handle=log_handle)

            if should_run(ztf_manifest, args.force):
                remove_if_force(ztf_manifest, args.force)
                run_command(
                    "download/query ZTF zg,zr,zi science images and masks",
                    [
                        args.python,
                        root / "download_ztf_stripe82_science.py",
                        "--regions",
                        f"j230056,{args.ra},{args.dec},0.05,0.05",
                        "--filters",
                        ZTF_FILTERS,
                        "--suffixes",
                        "sciimg.fits,mskimg.fits",
                        "--out-dir",
                        out / "ztf_science",
                        "--manifest",
                        ztf_manifest,
                        "--timeout",
                        str(args.timeout),
                        "--workers",
                        str(args.ztf_workers),
                    ],
                    env=env,
                    log_handle=log_handle,
                )
            else:
                log_line(f"==> using existing ZTF manifest: {ztf_manifest}", log_handle=log_handle)
        elif should_run(target_catalog, args.force):
            write_csv(
                target_catalog,
                [{"star_id": args.target_id, "name": args.target_id, "ra": f"{args.ra:.10f}", "dec": f"{args.dec:.10f}"}],
                ["star_id", "name", "ra", "dec"],
            )

        if should_run(standards, args.force) or csv_data_row_count(standards) == 0:
            if standards.exists():
                log_line(f"==> standards file has no data rows; regenerating: {standards}", log_handle=log_handle)
                standards.unlink()
            run_command(
                "subset Ivezić Stripe 82 standard stars",
                [
                    args.python,
                    root / "subset_ivezic_standard_stars.py",
                    "--ra",
                    str(args.ra),
                    "--dec",
                    str(args.dec),
                    "--radius-deg",
                    str(args.standard_radius_deg),
                    "--out-reference",
                    standards,
                    "--timeout",
                    str(args.timeout),
                ],
                env=env,
                log_handle=log_handle,
            )

        run_command(
            "build combined ugri-equivalent image manifest",
            [
                args.python,
                root / "build_patch_image_manifest.py",
                "--sdss-manifest",
                sdss_manifest,
                "--panstarrs-manifest",
                ps1_manifest,
                "--ztf-manifest",
                ztf_manifest,
                "--out-manifest",
                combined_manifest,
            ],
            env=env,
            log_handle=log_handle,
        )
        existing_ugri_manifest(
            combined_manifest,
            existing_manifest,
            max_images=args.photometry_max_images,
        )

        calibration_outputs_stale = not csv_has_columns(
            compressed_photometry,
            {"compression_mode", "marginalized_grid_points", "host_model", "host_rate", "point_host_rate_cov"},
        )
        if calibration_outputs_stale:
            log_line(
                "==> existing compressed likelihoods are missing marginalized host-likelihood columns; rerunning calibration",
                log_handle=log_handle,
            )
        if (
            calibration_outputs_stale
            or should_run(agn_calibrated, args.force)
            or should_run(standards_calibrated, args.force)
            or should_run(zeropoints, args.force)
        ):
            joint_cmd = [
                args.python,
                root / "run_joint_moffat_patch_model.py",
                "--standards",
                standards,
                "--target-catalog",
                target_catalog,
                "--image-manifest",
                existing_manifest,
                "--out-compressed",
                compressed_photometry,
                "--out-zeropoints",
                zeropoints,
                "--out-stars",
                standards_calibrated,
                "--out-agn",
                agn_calibrated,
                "--out-params",
                params,
                "--stamp-radius",
                "15",
                "--edge-margin",
                "20",
                "--moffat-fwhm-pix",
                str(args.moffat_fwhm_pix),
                "--moffat-beta",
                str(args.moffat_beta),
                "--host-sersic-n",
                str(args.host_sersic_n),
                "--host-reff-pix",
                str(args.host_reff_pix),
                "--host-axis-ratio",
                str(args.host_axis_ratio),
                "--host-pa-deg",
                str(args.host_pa_deg),
                "--prior-host-mag-loc",
                str(args.prior_host_mag_loc),
                "--prior-host-mag-sigma",
                str(args.prior_host_mag_sigma),
                "--min-moffat-fwhm-pix",
                str(args.min_moffat_fwhm_pix),
                "--max-moffat-fwhm-pix",
                str(args.max_moffat_fwhm_pix),
                "--fwhm-grid-size",
                str(args.fwhm_grid_size),
                "--fwhm-grid-log-span",
                str(args.fwhm_grid_log_span),
                "--prior-log-fwhm-sigma",
                str(args.prior_log_fwhm_sigma),
                "--centroid-grid-size",
                str(args.centroid_grid_size),
                "--centroid-grid-radius-pix",
                str(args.centroid_grid_radius_pix),
                "--prior-centroid-sigma-pix",
                str(args.prior_centroid_sigma_pix),
                "--prior-survey-filter-sigma",
                str(args.prior_survey_filter_sigma),
                "--prior-night-sigma",
                str(args.prior_night_sigma),
                "--prior-image-sigma",
                str(args.prior_image_sigma),
                "--prior-ccd-sigma",
                str(args.prior_ccd_sigma),
                "--prior-color-sigma",
                str(args.prior_color_sigma),
                "--prior-intrinsic-scatter-sigma",
                str(args.prior_intrinsic_scatter_sigma),
                "--initial-intrinsic-scatter",
                str(args.initial_intrinsic_scatter),
                "--prior-star-mag-floor",
                str(args.prior_star_mag_floor),
                "--student-t-df",
                str(args.student_t_df),
                "--reference-system",
                args.reference_system,
                "--num-warmup",
                str(args.calibration_warmup),
                "--num-samples",
                str(args.calibration_samples),
            ]
            if args.ignore_header_seeing:
                joint_cmd.append("--ignore-header-seeing")
            if args.fix_image_fwhm:
                joint_cmd.append("--fix-image-fwhm")
            if args.fix_centroids:
                joint_cmd.append("--fix-centroids")
            if args.photometry_max_stars_per_image and args.photometry_max_stars_per_image > 0:
                joint_cmd.extend(["--max-stars-per-image", str(args.photometry_max_stars_per_image)])
            run_command(
                "run fast joint Moffat calibration and AGN light-curve model",
                joint_cmd,
                env=env,
                log_handle=log_handle,
            )

        if should_run(lightcurve, args.force):
            run_command(
                "plot AGN and standard-star light curves",
                [
                    args.python,
                    root / "plot_patch_lightcurves.py",
                    "--agn-photometry",
                    agn_calibrated,
                    "--star-photometry",
                    standards_calibrated,
                    "--target-id",
                    args.target_id,
                    "--out-figure",
                    lightcurve,
                ],
                env=env,
                log_handle=log_handle,
            )

        if should_run(quality_diagnostics_summary, args.force):
            run_command(
                "plot calibration quality diagnostics",
                [
                    args.python,
                    root / "plot_quality_diagnostics.py",
                    "--star-photometry",
                    standards_calibrated,
                    "--target-id",
                    args.target_id,
                    "--out-dir",
                    quality_diagnostics_dir,
                    "--out-summary",
                    quality_diagnostics_summary,
                ],
                env=env,
                log_handle=log_handle,
            )

        log_line("\nPipeline complete.", log_handle=log_handle)
        log_line(f"Combined manifest: {combined_manifest}", log_handle=log_handle)
        log_line(f"Joint-fit manifest used: {existing_manifest}", log_handle=log_handle)
        log_line(f"Compressed likelihoods: {compressed_photometry}", log_handle=log_handle)
        log_line(f"AGN calibrated photometry: {agn_calibrated}", log_handle=log_handle)
        log_line(f"Standard-star calibrated photometry: {standards_calibrated}", log_handle=log_handle)
        log_line(f"Light curve figure: {lightcurve}", log_handle=log_handle)
        log_line(f"Quality diagnostic figures: {quality_diagnostics_dir}", log_handle=log_handle)
        log_line(f"Quality diagnostic summary: {quality_diagnostics_summary}", log_handle=log_handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
