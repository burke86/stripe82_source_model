#!/usr/bin/env python3
"""Plot standard-star stamp diagnostics for one image.

This is a lightweight inspection helper for the fast Moffat pipeline outputs.
It recreates the per-source stamp, Moffat model, residual, good-pixel mask,
and a broader science/mask/weight context panel for any image present in the
calibrated standard-star table.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from run_joint_moffat_patch_model import first_2d_hdu, load_noise_and_mask, cutout


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


def load_fastphot():
    try:
        from jaguar.fastphot import matched_filter_flux, moffat_unit_flux

        return matched_filter_flux, moffat_unit_flux
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        fastphot_path = repo_root / "jaguar" / "src" / "jaguar" / "fastphot.py"
        if not fastphot_path.exists():
            raise
        spec = importlib.util.spec_from_file_location("fastphot_local", fastphot_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {fastphot_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.matched_filter_flux, module.moffat_unit_flux


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def image_name_matches(row: dict[str, str], image: str) -> bool:
    image_path = row.get("image_path") or ""
    return image_path == image or Path(image_path).name == Path(image).name


def selected_rows(
    rows: list[dict[str, str]],
    image: str,
    *,
    max_sources: int,
    include_worst: int,
    prefer_full_stamps: bool,
) -> list[dict[str, str]]:
    matches = []
    for row in rows:
        if row.get("source_type", "standard") != "standard":
            continue
        if not image_name_matches(row, image):
            continue
        residual = parse_float(row.get("calibration_residual"))
        if residual is None:
            continue
        row = dict(row)
        row["_abs_residual"] = str(abs(residual))
        row["_npixels_float"] = str(parse_float(row.get("npixels")) or 0.0)
        matches.append(row)
    if not matches:
        raise RuntimeError(f"No standard-star rows found for {image}")

    worst = sorted(matches, key=lambda row: float(row["_abs_residual"]), reverse=True)[: max(include_worst, 0)]
    rest = matches
    if prefer_full_stamps:
        rest = sorted(rest, key=lambda row: (-float(row["_npixels_float"]), float(row["_abs_residual"])))
    else:
        rest = sorted(rest, key=lambda row: float(row["_abs_residual"]))

    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in worst + rest:
        key = row.get("star_id") or f"{row.get('x')},{row.get('y')}"
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) >= max_sources:
            break
    return selected


def manifest_row_for_image(manifest: Path, image: str) -> dict[str, str]:
    for row in read_rows(manifest):
        if image_name_matches(row, image):
            return row
    raise RuntimeError(f"No manifest row found for {image}")


def output_stem(image: str, label: str | None) -> str:
    stem = Path(image).name
    if stem.endswith(".fits"):
        stem = stem[:-5]
    if label:
        stem = f"{stem}_{label}"
    return stem


def plot_stamp_diagnostics(args: argparse.Namespace, rows: list[dict[str, str]], image: np.ndarray, noise, good_mask):
    import matplotlib.pyplot as plt

    matched_filter_flux, moffat_unit_flux = load_fastphot()
    fig = plt.figure(figsize=(14, 3.2 * len(rows)), constrained_layout=True)
    subfigs = fig.subfigures(len(rows), 1)
    if len(rows) == 1:
        subfigs = [subfigs]

    for subfig, row in zip(subfigs, rows):
        axes = subfig.subplots(1, 4)
        x = float(row["x"])
        y = float(row["y"])
        stamp, _origin = cutout(image, x, y, args.stamp_radius, fill=np.nan)
        if np.ndim(noise) == 0:
            noise_stamp = float(noise)
        else:
            noise_stamp, _ = cutout(noise, x, y, args.stamp_radius, fill=np.nan)
        if good_mask is None:
            mask_bool = None
        else:
            mask_stamp, _ = cutout(good_mask.astype(float), x, y, args.stamp_radius, fill=0.0)
            mask_bool = mask_stamp.astype(bool)

        fwhm = parse_float(row.get("fitted_moffat_fwhm_pix")) or parse_float(row.get("moffat_fwhm_pix")) or 3.0
        beta = parse_float(row.get("moffat_beta")) or 3.5
        wcs_dx = x - round(x)
        wcs_dy = y - round(y)
        dx = parse_float(row.get("fit_center_x_pix"))
        dy = parse_float(row.get("fit_center_y_pix"))
        if dx is None or dy is None:
            dx = wcs_dx
            dy = wcs_dy
        psf = moffat_unit_flux(stamp.shape, fwhm_pix=fwhm, beta=beta, dx_pix=dx, dy_pix=dy)
        fit = matched_filter_flux(stamp, noise_stamp, psf, mask=mask_bool, fit_background=True)
        model = fit.flux * psf + fit.background
        residual = stamp - model

        finite = np.isfinite(stamp)
        vmin, vmax = np.nanpercentile(stamp[finite], [2, 99.5]) if finite.any() else (0.0, 1.0)
        resid_values = residual[np.isfinite(residual)]
        resid_limit = np.nanpercentile(np.abs(resid_values), 98) if resid_values.size else 1.0
        mask_panel = mask_bool.astype(float) if mask_bool is not None else np.ones_like(stamp)

        panels = [stamp, model, residual, mask_panel]
        titles = [
            f"data\n{row.get('star_id', '')} res={float(row['calibration_residual']):+.2f} mag",
            f"Moffat model\nfwhm={fwhm:.2f}px dx={dx:+.2f} dy={dy:+.2f}",
            f"data-model\nchi2={fit.reduced_chi2:.1f} npix={fit.npixels}",
            "good pixels mask",
        ]
        cmaps = ["gray", "gray", "coolwarm", "gray"]
        for ax, panel, title, cmap in zip(axes, panels, titles, cmaps):
            if title.startswith("data-model"):
                image_artist = ax.imshow(panel, origin="lower", cmap=cmap, vmin=-resid_limit, vmax=resid_limit)
            elif title == "good pixels mask":
                image_artist = ax.imshow(panel, origin="lower", cmap=cmap, vmin=0, vmax=1)
            else:
                image_artist = ax.imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.plot(args.stamp_radius + wcs_dx, args.stamp_radius + wcs_dy, "x", color="magenta", ms=7, mew=1.2)
            ax.plot(args.stamp_radius + dx, args.stamp_radius + dy, "+", color="cyan", ms=10, mew=1.5)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.02)
    return fig


def plot_context(args: argparse.Namespace, rows: list[dict[str, str]], image: np.ndarray, mask_image, weight_image):
    import matplotlib.pyplot as plt

    xs = np.asarray([float(row["x"]) for row in rows])
    ys = np.asarray([float(row["y"]) for row in rows])
    x0 = max(0, int(xs.min()) - args.context_pad)
    x1 = min(image.shape[1], int(xs.max()) + args.context_pad)
    y0 = max(0, int(ys.min()) - args.context_pad)
    y1 = min(image.shape[0], int(ys.max()) + args.context_pad)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    context = image[y0:y1, x0:x1]
    finite = np.isfinite(context)
    vmin, vmax = np.nanpercentile(context[finite], [1, 99.8]) if finite.any() else (0.0, 1.0)
    axes[0].imshow(context, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("science image context")

    if mask_image is None:
        axes[1].text(0.5, 0.5, "no mask", ha="center", va="center")
    else:
        axes[1].imshow(mask_image[y0:y1, x0:x1] != 0, origin="lower", cmap="gray")
    axes[1].set_title("mask nonzero pixels")

    if weight_image is None:
        axes[2].text(0.5, 0.5, "no weight", ha="center", va="center")
    else:
        weight = weight_image[y0:y1, x0:x1]
        good = np.isfinite(weight)
        lo, hi = np.nanpercentile(weight[good], [2, 98]) if good.any() else (0.0, 1.0)
        axes[2].imshow(weight, origin="lower", cmap="viridis", vmin=lo, vmax=hi)
    axes[2].set_title("weight map")

    for ax in axes:
        for index, row in enumerate(rows, start=1):
            ax.plot(float(row["x"]) - x0, float(row["y"]) - y0, "o", mfc="none", mec=args.marker_color, ms=10, mew=1.5)
            ax.text(float(row["x"]) - x0 + 3, float(row["y"]) - y0 + 3, str(index), color="yellow", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot image-level standard-star fitting diagnostics.")
    parser.add_argument("--star-photometry", required=True, type=Path, help="Calibrated standard-star CSV.")
    parser.add_argument("--image-manifest", required=True, type=Path, help="Manifest with image/mask/weight paths.")
    parser.add_argument("--image", required=True, help="Image path or basename to diagnose.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--label", help="Optional label inserted in output filenames.")
    parser.add_argument("--max-sources", type=int, default=8)
    parser.add_argument("--include-worst", type=int, default=2)
    parser.add_argument("--prefer-full-stamps", action="store_true")
    parser.add_argument("--stamp-radius", type=int, default=15)
    parser.add_argument("--context-pad", type=int, default=80)
    parser.add_argument("--default-noise", type=float, default=1.0)
    parser.add_argument("--marker-color", default="lime")
    return parser.parse_args()


def optional_existing_file(value: str | None) -> Path | None:
    text = (value or "").strip()
    if not text or text == ".":
        return None
    path = Path(text)
    return path if path.exists() and path.is_file() else None


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    star_rows = read_rows(args.star_photometry)
    selected = selected_rows(
        star_rows,
        args.image,
        max_sources=args.max_sources,
        include_worst=args.include_worst,
        prefer_full_stamps=args.prefer_full_stamps,
    )
    image_row = manifest_row_for_image(args.image_manifest, args.image)
    image_path = Path((image_row.get("image_path") or "").strip())
    image, _header = first_2d_hdu(image_path)
    noise, good_mask, noise_source, mask_source = load_noise_and_mask(image_row, image.shape, image, args.default_noise)

    mask_image = None
    mask_path = optional_existing_file(image_row.get("mask_path"))
    if mask_path is not None and not mask_path.name.startswith("fpM-"):
        mask_image, _ = first_2d_hdu(mask_path)
    weight_image = None
    weight_path = optional_existing_file(image_row.get("invvar_path"))
    if weight_path is not None:
        weight_image, _ = first_2d_hdu(weight_path)

    stem = output_stem(args.image, args.label)
    stamp_fig = plot_stamp_diagnostics(args, selected, image, noise, good_mask)
    stamp_fig.suptitle(
        f"standard-star fitting diagnostics\n{Path(args.image).name}\n"
        f"noise={Path(noise_source).name if noise_source else noise_source}; "
        f"mask={Path(mask_source).name if mask_source else mask_source}",
        fontsize=13,
    )
    stamp_path = args.out_dir / f"{stem}_standard_star_fit_diagnostics.png"
    stamp_fig.savefig(stamp_path, dpi=160)

    context_fig = plot_context(args, selected, image, mask_image, weight_image)
    context_fig.suptitle(f"Context for selected stars in {Path(args.image).name}", fontsize=13)
    context_path = args.out_dir / f"{stem}_context_diagnostics.png"
    context_fig.savefig(context_path, dpi=160)

    print(stamp_path)
    print(context_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
