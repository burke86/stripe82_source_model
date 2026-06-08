#!/usr/bin/env python3
"""Plot calibration residuals against image/source quality diagnostics."""

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
}
BAND_COLORS = {
    "u": "#1f77b4",
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#9467bd",
}
DIAGNOSTICS = [
    ("fitted_moffat_fwhm_pix", "fitted Moffat FWHM [pix]", "residual_vs_fwhm.png"),
    ("mask_fraction", "masked pixel fraction", "residual_vs_mask_fraction.png"),
    ("centroid_shift_pix", "source/survey centroid shift [pix]", "residual_vs_centroid_shift.png"),
    ("background", "profiled stamp background", "residual_vs_background.png"),
    ("reduced_chi2", "profiled fit reduced chi2", "residual_vs_reduced_chi2.png"),
    ("snr", "matched-filter S/N", "residual_vs_snr.png"),
    ("calibration_mag_offset", "calibration mag offset", "residual_vs_calibration_offset.png"),
]


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


def display_band(row: dict[str, str]) -> str:
    return FILTER_TO_BAND.get((row.get("filter") or "").strip(), (row.get("band") or row.get("filter") or "").strip())


def star_residual(row: dict[str, str]) -> float | None:
    if (row.get("calibration_status") or "ok").strip() != "ok":
        return None
    residual = parse_float(row.get("calibration_residual"))
    if residual is not None:
        return residual
    mag = parse_float(row.get("mag_cal"))
    ref = parse_float(row.get("star_true_mag")) or parse_float(row.get("ref_mag"))
    if mag is None or ref is None:
        return None
    return mag - ref


def metric_value(row: dict[str, str], metric: str) -> float | None:
    if metric == "centroid_shift_pix":
        dx = parse_float(row.get("fit_centroid_shift_x_pix"))
        dy = parse_float(row.get("fit_centroid_shift_y_pix"))
        if dx is None or dy is None:
            return None
        return math.hypot(dx, dy)
    return parse_float(row.get(metric))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def clipped_ylim(values: list[float], limit: float) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -limit, limit
    lo = percentile(finite, 0.01)
    hi = percentile(finite, 0.99)
    bound = max(abs(lo), abs(hi), 0.1)
    bound = min(max(bound * 1.15, 0.2), limit)
    return -bound, bound


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot standard-star residuals against quality diagnostics.")
    parser.add_argument("--star-photometry", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-summary", type=Path)
    parser.add_argument("--target-id", default="J230056.54-001711.1")
    parser.add_argument("--max-abs-residual", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("plotting requires matplotlib") from exc

    rows = []
    for row in read_rows(args.star_photometry):
        residual = star_residual(row)
        if residual is None:
            continue
        row = dict(row)
        row["_residual"] = residual
        row["_survey"] = (row.get("survey") or "unknown").strip() or "unknown"
        row["_band"] = display_band(row)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No usable standard-star rows found in {args.star_photometry}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    surveys = sorted({row["_survey"] for row in rows})
    residuals = [float(row["_residual"]) for row in rows]
    y_limits = clipped_ylim(residuals, args.max_abs_residual)
    summary_rows = []

    for metric, xlabel, filename in DIAGNOSTICS:
        fig, axes = plt.subplots(
            len(surveys),
            1,
            figsize=(11, max(3.2, 2.7 * len(surveys))),
            sharex=False,
            constrained_layout=True,
        )
        if len(surveys) == 1:
            axes = [axes]
        for ax, survey in zip(axes, surveys):
            survey_rows = [row for row in rows if row["_survey"] == survey]
            hidden = 0
            for band in ["u", "g", "r", "i"]:
                xs = []
                ys = []
                for row in survey_rows:
                    if row["_band"] != band:
                        continue
                    value = metric_value(row, metric)
                    residual = float(row["_residual"])
                    if value is None or not math.isfinite(residual):
                        hidden += 1
                        continue
                    xs.append(value)
                    ys.append(residual)
                if xs:
                    ax.scatter(xs, ys, s=10, alpha=0.35, color=BAND_COLORS.get(band), label=band)
            ax.axhline(0.0, color="0.2", lw=0.8)
            ax.set_title(f"{survey} ({len(survey_rows)} standard-star observations)")
            ax.set_ylabel("calibration residual [mag]")
            ax.set_ylim(*y_limits)
            ax.grid(alpha=0.25)
            if hidden:
                ax.text(0.99, 0.04, f"missing {hidden}", transform=ax.transAxes, ha="right", va="bottom", fontsize="small", color="0.35")
            ax.legend(title="band", ncol=4, fontsize="small")
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(f"{args.target_id}: standard-star residual vs {xlabel}")
        out_path = args.out_dir / filename
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"wrote {out_path}")

        for survey in surveys:
            for band in ["u", "g", "r", "i"]:
                values = [
                    float(row["_residual"])
                    for row in rows
                    if row["_survey"] == survey and row["_band"] == band and metric_value(row, metric) is not None
                ]
                if not values:
                    continue
                summary_rows.append(
                    {
                        "metric": metric,
                        "survey": survey,
                        "band": band,
                        "n": len(values),
                        "median_residual": percentile(values, 0.5),
                        "p16_residual": percentile(values, 0.16),
                        "p84_residual": percentile(values, 0.84),
                    }
                )

    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.out_summary.open("w", newline="") as handle:
            fieldnames = ["metric", "survey", "band", "n", "median_residual", "p16_residual", "p84_residual"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
