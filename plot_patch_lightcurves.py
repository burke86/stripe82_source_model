#!/usr/bin/env python3
"""Plot AGN and calibration-star light curves for a small patch test."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FILTER_TO_BAND = {
    "u": "u",
    "g": "g",
    "r": "r",
    "i": "i",
    "zg": "g",
    "zr": "r",
    "zi": "i",
    "ps1_g": "g",
    "ps1_r": "r",
    "ps1_i": "i",
}
BAND_COLORS = {
    "u": "#1f77b4",
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#9467bd",
}
BAND_ORDER = ["u", "g", "r", "i"]
BAND_MAG_OFFSETS = {
    "u": -0.5,
    "g": -0.25,
    "r": 0.0,
    "i": 0.25,
}


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def choose_mag(row: dict[str, str]) -> tuple[float | None, float | None]:
    if "calibration_status" in row and (row.get("calibration_status") or "").strip() != "ok":
        return None, None
    mag = parse_float(row.get("mag_cal"))
    err = parse_float(row.get("mag_err"))
    if mag is not None:
        return mag, err
    flux = parse_float(row.get("flux"))
    flux_err = parse_float(row.get("flux_err"))
    exptime = parse_float(row.get("exptime")) or 1.0
    scale = parse_float(row.get("calibration_scale")) or 1.0
    if flux is None or flux <= 0 or exptime <= 0:
        return None, None
    mag = -2.5 * math.log10((flux * scale) / exptime)
    if flux_err is not None and flux_err > 0:
        err = 2.5 / math.log(10) * flux_err / flux
    return mag, err


def row_time(row: dict[str, str], index: int) -> float:
    for key in ("mjd", "obs_mjd", "jd"):
        value = parse_float(row.get(key))
        if value is not None:
            return value
    night = parse_float(row.get("night"))
    if night is not None:
        return night
    return float(index)


def display_band(row: dict[str, str]) -> str:
    filter_name = (row.get("filter") or "").strip()
    return FILTER_TO_BAND.get(filter_name, filter_name)


def choose_star_residual(row: dict[str, str]) -> float | None:
    if "calibration_status" in row and (row.get("calibration_status") or "").strip() != "ok":
        return None
    residual = parse_float(row.get("calibration_residual"))
    if residual is not None:
        return residual
    mag, _ = choose_mag(row)
    ref_mag = parse_float(row.get("ref_mag"))
    if mag is None or ref_mag is None:
        return None
    return mag - ref_mag


def row_mask_fraction(row: dict[str, str]) -> float:
    value = parse_float(row.get("mask_fraction"))
    if value is not None:
        return value
    npixels = parse_float(row.get("npixels"))
    if npixels is None:
        return 0.0
    radius = parse_float(row.get("stamp_radius"))
    if radius is not None and radius >= 0:
        area = (2 * int(radius) + 1) ** 2
    else:
        area = 31 * 31
    if area <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - npixels / area))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot patch AGN and standard-star light curves.")
    parser.add_argument("--agn-photometry", required=True, type=Path)
    parser.add_argument("--star-photometry", required=True, type=Path)
    parser.add_argument("--target-id", default="J230056.54-001711.1")
    parser.add_argument("--out-figure", required=True, type=Path)
    parser.add_argument("--max-stars", type=int, default=12)
    parser.add_argument("--min-snr", type=float, default=3.0, help="Minimum absolute matched-filter S/N to show in the AGN panel.")
    parser.add_argument("--max-agn-errorbar", type=float, default=1.0, help="Clip displayed AGN magnitude error bars to this size.")
    parser.add_argument("--agn-mag-padding", type=float, default=0.5, help="Padding around robust AGN magnitude limits.")
    parser.add_argument("--max-star-abs-residual", type=float, default=2.0, help="Hide standard-star residuals outside this magnitude range.")
    parser.add_argument("--max-mask-fraction", type=float, default=0.10, help="Do not plot rows with more than this fraction of masked stamp pixels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("plotting requires matplotlib") from exc

    agn_rows = read_rows(args.agn_photometry)
    star_rows = read_rows(args.star_photometry)
    bands = [band for band in BAND_ORDER if any(display_band(row) == band for row in agn_rows + star_rows)]
    extra_bands = sorted(
        {
            display_band(row)
            for row in agn_rows + star_rows
            if row.get("filter") and display_band(row) not in bands
        }
    )
    bands.extend(extra_bands)
    if not bands:
        raise SystemExit("No filter values found")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False, constrained_layout=True)
    ax_agn, ax_stars = axes
    hidden_agn = 0
    hidden_agn_masked = 0
    hidden_star = 0
    hidden_star_masked = 0

    for band in bands:
        xs: list[float] = []
        ys: list[float] = []
        es: list[float] = []
        for index, row in enumerate(agn_rows):
            if display_band(row) != band:
                continue
            if row_mask_fraction(row) > args.max_mask_fraction:
                hidden_agn_masked += 1
                continue
            mag, err = choose_mag(row)
            if mag is None:
                continue
            snr = parse_float(row.get("snr"))
            if snr is not None and abs(snr) < args.min_snr:
                hidden_agn += 1
                continue
            xs.append(row_time(row, index))
            ys.append(mag + BAND_MAG_OFFSETS.get(band, 0.0))
            if err is None or err < 0:
                es.append(0.0)
            else:
                es.append(min(err, args.max_agn_errorbar))
        if xs:
            ax_agn.errorbar(
                xs,
                ys,
                yerr=es,
                fmt="o",
                ms=4,
                capsize=2,
                color=BAND_COLORS.get(band),
                label=band,
            )

    star_ids = []
    for row in star_rows:
        star_id = (row.get("star_id") or "").strip()
        if star_id and star_id not in star_ids:
            star_ids.append(star_id)
        if len(star_ids) >= args.max_stars:
            break

    labeled_bands: set[str] = set()
    for star_id in star_ids:
        for band in bands:
            points: list[tuple[float, float]] = []
            errors: list[float] = []
            for index, row in enumerate(star_rows):
                if (row.get("star_id") or "").strip() != star_id:
                    continue
                if display_band(row) != band:
                    continue
                if row_mask_fraction(row) > args.max_mask_fraction:
                    hidden_star_masked += 1
                    continue
                residual = choose_star_residual(row)
                if residual is None:
                    continue
                if abs(residual) > args.max_star_abs_residual:
                    hidden_star += 1
                    continue
                points.append((row_time(row, index), residual))
                mag_err = parse_float(row.get("mag_err"))
                ref_mag_err = parse_float(row.get("ref_mag_err")) or 0.0
                if mag_err is None or mag_err < 0:
                    errors.append(0.0)
                else:
                    errors.append(math.sqrt(mag_err**2 + ref_mag_err**2))
            if not points:
                continue
            label = band if band not in labeled_bands else None
            labeled_bands.add(band)
            ax_stars.errorbar(
                [x for x, _y in points],
                [y for _x, y in points],
                yerr=errors,
                fmt="o",
                ms=3,
                elinewidth=0.7,
                capsize=1.5,
                alpha=0.55,
                color=BAND_COLORS.get(band),
                label=label,
            )

    ax_agn.set_title(args.target_id)
    ax_agn.set_ylabel("AGN magnitude + display offset")
    ax_agn.invert_yaxis()
    agn_mags = []
    for row in agn_rows:
        if row_mask_fraction(row) > args.max_mask_fraction:
            continue
        mag, _err = choose_mag(row)
        if mag is not None:
            agn_mags.append(mag + BAND_MAG_OFFSETS.get(display_band(row), 0.0))
    if len(agn_mags) >= 3:
        sorted_mags = sorted(agn_mags)
        lo = sorted_mags[int(0.02 * (len(sorted_mags) - 1))]
        hi = sorted_mags[int(0.98 * (len(sorted_mags) - 1))]
        ax_agn.set_ylim(hi + args.agn_mag_padding, lo - args.agn_mag_padding)
    ax_agn.legend(title="PS1-system band", ncol=min(len(bands), 4), fontsize="small")
    ax_agn.grid(alpha=0.25)
    if hidden_agn or hidden_agn_masked:
        lines = []
        if hidden_agn:
            lines.append(f"hidden low-S/N points: {hidden_agn}")
        if hidden_agn_masked:
            lines.append(f"hidden masked points: {hidden_agn_masked}")
        ax_agn.text(
            0.99,
            0.03,
            "\n".join(lines),
            transform=ax_agn.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            color="0.35",
        )

    ax_stars.set_ylabel("standard-star calibration residual")
    ax_stars.set_xlabel("epoch / night")
    ax_stars.axhline(0.0, color="0.2", lw=0.8)
    ax_stars.set_ylim(-args.max_star_abs_residual, args.max_star_abs_residual)
    ax_stars.grid(alpha=0.25)
    ax_stars.legend(title="PS1-system band", fontsize="small", ncol=min(len(bands), 4))
    if hidden_star or hidden_star_masked:
        lines = []
        if hidden_star:
            lines.append(f"hidden residual outliers: {hidden_star}")
        if hidden_star_masked:
            lines.append(f"hidden masked points: {hidden_star_masked}")
        ax_stars.text(
            0.99,
            0.03,
            "\n".join(lines),
            transform=ax_stars.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            color="0.35",
        )

    args.out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_figure, dpi=180)
    print(f"wrote {args.out_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
