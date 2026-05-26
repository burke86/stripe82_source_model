#!/usr/bin/env python3
"""Run JAGUAR PSF photometry at Stripe 82 standard-star positions.

Inputs:

1. A standard-star catalog with ``star_id,ra,dec`` columns.
2. An image manifest with one row per CCD/image product.

For each manifest row, the script finds standards inside the image WCS, extracts
a stamp around each star, builds a one-component JAGUAR point-source model, and
fits source flux, position, and local background. The output CSV is designed to
feed ``calibrate_standard_star_photometry.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass(frozen=True)
class StandardStar:
    star_id: str
    ra: float
    dec: float
    row: dict[str, str]


@dataclass(frozen=True)
class ImageRecord:
    row: dict[str, str]
    image_path: Path
    survey: str
    night: str
    image_id: str
    ccd_id: str
    filter_name: str
    psf_path: Path | None
    noise_path: Path | None
    invvar_path: Path | None
    variance_path: Path | None
    zeropoint: float | None
    exptime: float
    science_hdu: int | str | None
    psf_hdu: int | str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JAGUAR spatial PSF fitting at standard-star positions in CCD images."
    )
    parser.add_argument("--standards", required=True, type=Path, help="CSV with star_id,ra,dec columns.")
    parser.add_argument("--image-manifest", required=True, type=Path, help="CSV with image metadata and paths.")
    parser.add_argument("--out-photometry", required=True, type=Path, help="Output PSF photometry CSV.")
    parser.add_argument("--star-id-column", default="star_id")
    parser.add_argument("--ra-column", default="ra")
    parser.add_argument("--dec-column", default="dec")

    parser.add_argument("--image-path-column", default="image_path")
    parser.add_argument("--noise-path-column", default="noise_path")
    parser.add_argument("--invvar-path-column", default="invvar_path")
    parser.add_argument("--variance-path-column", default="variance_path")
    parser.add_argument("--psf-path-column", default="psf_path")
    parser.add_argument("--survey-column", default="survey")
    parser.add_argument("--night-column", default="night")
    parser.add_argument("--image-id-column", default="image_id")
    parser.add_argument("--ccd-id-column", default="ccd_id")
    parser.add_argument("--filter-column", default="filter")
    parser.add_argument("--zeropoint-column", default="zeropoint")
    parser.add_argument("--exptime-column", default="exptime")
    parser.add_argument("--science-hdu-column", default="science_hdu")
    parser.add_argument("--psf-hdu-column", default="psf_hdu")

    parser.add_argument("--stamp-radius", type=int, default=15, help="Stamp radius in pixels. Default: 15.")
    parser.add_argument("--edge-margin", type=int, default=20, help="Skip stars this close to an image edge. Default: 20.")
    parser.add_argument("--max-stars-per-image", type=int, help="Optional cap for testing.")
    parser.add_argument("--map-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--position-sigma-pix", type=float, default=0.35)
    parser.add_argument("--fit-method", choices=["map_only", "optax+nuts"], default="map_only")
    parser.add_argument("--nuts-warmup", type=int, default=300)
    parser.add_argument("--nuts-samples", type=int, default=300)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--subtract-edge-background", action="store_true")
    parser.add_argument("--default-noise", type=float, default=1.0)
    parser.add_argument("--default-gaussian-psf-fwhm-pix", type=float, default=3.0)
    parser.add_argument("--default-gaussian-psf-size", type=int, default=25)
    return parser.parse_args()


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


def parse_hdu(value: str | None) -> int | str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def optional_path(row: dict[str, str], column: str) -> Path | None:
    value = (row.get(column) or "").strip()
    return Path(value) if value else None


def read_standards(path: Path, args: argparse.Namespace) -> list[StandardStar]:
    stars: list[StandardStar] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{path} has no CSV header")
        for column in (args.star_id_column, args.ra_column, args.dec_column):
            if column not in reader.fieldnames:
                raise RuntimeError(f"{path} must contain {column!r}")
        for row in reader:
            star_id = (row.get(args.star_id_column) or "").strip()
            ra = parse_float(row.get(args.ra_column))
            dec = parse_float(row.get(args.dec_column))
            if not star_id or ra is None or dec is None:
                continue
            stars.append(StandardStar(star_id=star_id, ra=ra, dec=dec, row=row))
    if not stars:
        raise RuntimeError(f"No standards found in {path}")
    return stars


def read_manifest(path: Path, args: argparse.Namespace) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{path} has no CSV header")
        required = [
            args.image_path_column,
            args.survey_column,
            args.night_column,
            args.image_id_column,
            args.ccd_id_column,
            args.filter_column,
        ]
        for column in required:
            if column not in reader.fieldnames:
                raise RuntimeError(f"{path} must contain {column!r}")
        for row in reader:
            image_path = optional_path(row, args.image_path_column)
            if image_path is None:
                continue
            records.append(
                ImageRecord(
                    row=row,
                    image_path=image_path,
                    survey=(row.get(args.survey_column) or "").strip(),
                    night=(row.get(args.night_column) or "").strip(),
                    image_id=(row.get(args.image_id_column) or "").strip(),
                    ccd_id=(row.get(args.ccd_id_column) or "").strip(),
                    filter_name=(row.get(args.filter_column) or "").strip(),
                    psf_path=optional_path(row, args.psf_path_column),
                    noise_path=optional_path(row, args.noise_path_column),
                    invvar_path=optional_path(row, args.invvar_path_column),
                    variance_path=optional_path(row, args.variance_path_column),
                    zeropoint=parse_float(row.get(args.zeropoint_column)),
                    exptime=parse_float(row.get(args.exptime_column)) or 1.0,
                    science_hdu=parse_hdu(row.get(args.science_hdu_column)),
                    psf_hdu=parse_hdu(row.get(args.psf_hdu_column)),
                )
            )
    if not records:
        raise RuntimeError(f"No image rows found in {path}")
    return records


def first_2d_hdu(hdul: fits.HDUList, preferred: int | str | None = None) -> tuple[np.ndarray, fits.Header]:
    if preferred is not None:
        hdu = hdul[preferred]
        if hdu.data is None:
            raise RuntimeError(f"Requested HDU {preferred!r} has no image data")
        return as_2d_image(hdu.data), hdu.header.copy()
    for hdu in hdul:
        if hdu.data is None:
            continue
        try:
            return as_2d_image(hdu.data), hdu.header.copy()
        except RuntimeError:
            continue
    raise RuntimeError("FITS file has no 2D image HDU")


def as_2d_image(data: Any) -> np.ndarray:
    image = np.asarray(data, dtype=float)
    if image.ndim > 2:
        image = np.asarray(image[0], dtype=float)
    if image.ndim != 2:
        raise RuntimeError("FITS image data is not 2D")
    return image


def read_image(path: Path, hdu: int | str | None = None) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        return first_2d_hdu(hdul, hdu)


def cutout(image: np.ndarray, center_xy: tuple[float, float], radius: int, *, fill: float = 0.0) -> tuple[np.ndarray, tuple[int, int]]:
    cx, cy = center_xy
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    size = 2 * int(radius) + 1
    out = np.full((size, size), fill, dtype=float)
    x0 = cx_i - radius
    x1 = cx_i + radius + 1
    y0 = cy_i - radius
    y1 = cy_i + radius + 1
    src_x0 = max(x0, 0)
    src_x1 = min(x1, image.shape[1])
    src_y0 = max(y0, 0)
    src_y1 = min(y1, image.shape[0])
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image[src_y0:src_y1, src_x0:src_x1]
    return out, (x0, y0)


def robust_edge_background(stamp: np.ndarray) -> float:
    edge = np.concatenate([stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1]])
    return float(np.nanmedian(edge))


def gaussian_psf(size: int, fwhm_pix: float) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    yy, xx = np.indices((size, size), dtype=float)
    center = (size - 1) / 2.0
    sigma = float(fwhm_pix) / 2.354820045
    psf = np.exp(-0.5 * ((xx - center) ** 2 + (yy - center) ** 2) / max(sigma, 1.0e-6) ** 2)
    return psf / np.sum(psf)


def load_psf(record: ImageRecord, args: argparse.Namespace) -> np.ndarray:
    if record.psf_path is None:
        return gaussian_psf(args.default_gaussian_psf_size, args.default_gaussian_psf_fwhm_pix)
    psf, _header = read_image(record.psf_path, record.psf_hdu)
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(np.clip(psf, 0.0, None)))
    if total <= 0:
        raise RuntimeError(f"PSF image has non-positive flux: {record.psf_path}")
    return np.clip(psf, 0.0, None) / total


def load_noise(record: ImageRecord, image: np.ndarray, header: fits.Header, args: argparse.Namespace) -> np.ndarray:
    del header
    if record.noise_path is not None:
        noise, _ = read_image(record.noise_path)
        return np.maximum(np.asarray(noise, dtype=float), 1.0e-12)
    if record.invvar_path is not None:
        invvar, _ = read_image(record.invvar_path)
        noise = np.full_like(invvar, 1.0e30, dtype=float)
        valid = np.isfinite(invvar) & (invvar > 0)
        noise[valid] = 1.0 / np.sqrt(invvar[valid])
        return noise
    if record.variance_path is not None:
        variance, _ = read_image(record.variance_path)
        return np.sqrt(np.maximum(variance, 1.0e-24))
    finite = image[np.isfinite(image)]
    if finite.size > 10:
        mad = np.nanmedian(np.abs(finite - np.nanmedian(finite)))
        sigma = 1.4826 * mad
        if np.isfinite(sigma) and sigma > 0:
            return np.ones_like(image, dtype=float) * sigma
    return np.ones_like(image, dtype=float) * float(args.default_noise)


def pixel_scale_from_header(header: fits.Header) -> float:
    if "PIXSCALE" in header:
        return abs(float(header["PIXSCALE"]))
    for key in ("CDELT1", "CD1_1"):
        if key in header:
            return abs(float(header[key])) * 3600.0
    return 1.0


def counts_per_mjy_from_zeropoint(zeropoint: float | None) -> float | None:
    if zeropoint is None:
        return None
    return float(10 ** ((float(zeropoint) - 16.4) / 2.5))


def standards_in_image(stars: list[StandardStar], wcs: WCS, shape: tuple[int, int], margin: int) -> list[tuple[StandardStar, float, float]]:
    coords = np.asarray([[star.ra, star.dec] for star in stars], dtype=float)
    pixels = wcs.all_world2pix(coords, 0)
    out: list[tuple[StandardStar, float, float]] = []
    eps = 1.0e-6
    for star, (x, y) in zip(stars, pixels):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if margin - eps <= x < shape[1] - margin + eps and margin - eps <= y < shape[0] - margin + eps:
            out.append((star, float(x), float(y)))
    return out


def run_jaguar_fit(stamp: np.ndarray, noise: np.ndarray, psf: np.ndarray, *, filter_name: str, pixel_scale: float, source_offset: tuple[float, float], counts_per_mjy: float | None, args: argparse.Namespace):
    from jaguar.config import ImageBandData, ImageFitConfig, JointFitConfig, SceneComponentConfig, SedComponentConfig
    from jaguar.fit import fit

    band = ImageBandData(
        image=np.nan_to_num(stamp, nan=0.0, posinf=0.0, neginf=0.0),
        noise=np.maximum(np.nan_to_num(noise, nan=1.0e30, posinf=1.0e30, neginf=1.0e30), 1.0e-12),
        psf=psf,
        filter_name=filter_name,
        pixel_scale=pixel_scale,
        counts_per_mjy=counts_per_mjy,
        mask=np.isfinite(stamp) & np.isfinite(noise) & (noise > 0),
    )
    sed_name = "standard_star"
    scene_name = "standard_star_image"
    cfg = JointFitConfig(
        image_bands=[band],
        image=ImageFitConfig(fit_background=True, background_default=0.0),
        sed_components=[
            SedComponentConfig(
                name=sed_name,
                kind="star",
                reference_filter_name=filter_name,
                reference_flux_mjy=1.0,
                fit_reference_flux=False,
            )
        ],
        scene_components=[
            SceneComponentConfig(
                name=scene_name,
                sed_component=sed_name,
                kind="point",
                fit_position=True,
                fixed_center_x_pix=source_offset[0],
                fixed_center_y_pix=source_offset[1],
                center_sigma_pix=args.position_sigma_pix,
            )
        ],
    )
    return fit(
        cfg,
        fit_method=args.fit_method,
        seed=args.seed,
        map_steps=args.map_steps,
        learning_rate=args.learning_rate,
        nuts_warmup=args.nuts_warmup,
        nuts_samples=args.nuts_samples,
        progress_bar=args.progress_bar,
    )


def flux_error_from_samples(result, scene_name: str, filter_name: str, flux: float) -> float:
    if result.samples:
        key = f"{scene_name}/{filter_name}/log_flux"
        if key in result.samples:
            values = np.exp(np.asarray(result.samples[key], dtype=float))
            return float(np.nanstd(values))
    return float(max(abs(flux) * 0.05, 1.0e-12))


def fit_record(record: ImageRecord, stars: list[StandardStar], args: argparse.Namespace) -> list[dict[str, Any]]:
    image, header = read_image(record.image_path, record.science_hdu)
    wcs = WCS(header, naxis=2)
    psf = load_psf(record, args)
    noise_image = load_noise(record, image, header, args)
    pixel_scale = pixel_scale_from_header(header)
    matches = standards_in_image(stars, wcs, image.shape, max(args.edge_margin, args.stamp_radius + 1))
    if args.max_stars_per_image is not None:
        matches = matches[: args.max_stars_per_image]

    rows: list[dict[str, Any]] = []
    for star, x, y in matches:
        stamp, origin = cutout(image, (x, y), args.stamp_radius)
        noise_stamp, _ = cutout(noise_image, (x, y), args.stamp_radius, fill=1.0e30)
        if args.subtract_edge_background:
            stamp = stamp - robust_edge_background(stamp)
        center = args.stamp_radius
        source_offset = (x - round(x), y - round(y))
        try:
            result = run_jaguar_fit(
                stamp,
                noise_stamp,
                psf,
                filter_name=record.filter_name,
                pixel_scale=pixel_scale,
                source_offset=source_offset,
                counts_per_mjy=counts_per_mjy_from_zeropoint(record.zeropoint),
                args=args,
            )
            key = f"standard_star_image/{record.filter_name}/log_flux"
            flux = float(np.exp(np.asarray(result.map_params[key])))
            flux_err = flux_error_from_samples(result, "standard_star_image", record.filter_name, flux)
            fit_x = float(np.asarray(result.map_params.get("standard_star_image/center_x_pix", source_offset[0])))
            fit_y = float(np.asarray(result.map_params.get("standard_star_image/center_y_pix", source_offset[1])))
            summary = result.summary()
            status = "ok"
            reduced_chi2 = summary.get(f"{record.filter_name}_reduced_chi2", np.nan)
        except Exception as exc:
            flux = np.nan
            flux_err = np.nan
            fit_x = np.nan
            fit_y = np.nan
            reduced_chi2 = np.nan
            status = f"failed:{type(exc).__name__}:{exc}"
        mag_inst = -2.5 * math.log10(flux / record.exptime) if np.isfinite(flux) and flux > 0 and record.exptime > 0 else ""
        mag_err = 2.5 / math.log(10) * flux_err / flux if np.isfinite(flux_err) and np.isfinite(flux) and flux > 0 else ""
        rows.append(
            {
                "star_id": star.star_id,
                "ra": star.ra,
                "dec": star.dec,
                "survey": record.survey,
                "night": record.night,
                "image_id": record.image_id,
                "ccd_id": record.ccd_id,
                "filter": record.filter_name,
                "flux": flux,
                "flux_err": flux_err,
                "exptime": record.exptime,
                "mag_inst": mag_inst,
                "mag_err": mag_err,
                "x": x,
                "y": y,
                "stamp_x0": origin[0],
                "stamp_y0": origin[1],
                "fit_center_x_pix": fit_x,
                "fit_center_y_pix": fit_y,
                "reduced_chi2": reduced_chi2,
                "zeropoint": "" if record.zeropoint is None else record.zeropoint,
                "image_path": str(record.image_path),
                "psf_path": "" if record.psf_path is None else str(record.psf_path),
                "fit_status": status,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    try:
        stars = read_standards(args.standards, args)
        records = read_manifest(args.image_manifest, args)
        args.out_photometry.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "star_id",
            "ra",
            "dec",
            "survey",
            "night",
            "image_id",
            "ccd_id",
            "filter",
            "flux",
            "flux_err",
            "exptime",
            "mag_inst",
            "mag_err",
            "x",
            "y",
            "stamp_x0",
            "stamp_y0",
            "fit_center_x_pix",
            "fit_center_y_pix",
            "reduced_chi2",
            "zeropoint",
            "image_path",
            "psf_path",
            "fit_status",
        ]
        total = 0
        with args.out_photometry.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                rows = fit_record(record, stars, args)
                writer.writerows(rows)
                total += len(rows)
                print(f"{record.image_id} {record.ccd_id} {record.filter_name}: wrote {len(rows)} rows")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {total} JAGUAR PSF photometry rows: {args.out_photometry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
