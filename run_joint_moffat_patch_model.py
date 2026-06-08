#!/usr/bin/env python3
"""Fast hierarchical joint calibration/light-curve model for one patch.

This script is the default science path for the patch test. It does not run a
separate pixel optimizer for every source. Instead it compresses each standard
star stamp to one Moffat matched-filter flux likelihood, compresses each AGN
stamp to one joint point-source plus Sersic-host flux likelihood, then infers
calibration terms and source magnitudes jointly with NumPyro.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


FILTER_TO_REF = {"u": "u", "g": "g", "r": "r", "i": "i", "zg": "g", "zr": "r", "zi": "i"}
PS1_BANDS = {"g", "r", "i", "z", "y"}
PS1_SDSS_GI_COEFFS = {
    "g": (-0.013, -0.145, 0.019, 0.013),
    "r": (-0.001, -0.014, 0.001, -0.001),
    "i": (0.004, -0.014, 0.014, -0.001),
    "z": (0.013, -0.039, 0.012, -0.001),
    "y": (0.015, -0.036, 0.012, -0.004),
}
SDSS_FPM_BAD_PLANES = {
    "S_MASK_INTERP",
    "S_MASK_SATUR",
    "S_MASK_NOTCHECKED",
    "S_MASK_BRIGHTOBJECT",
    "S_MASK_GHOST",
    "S_MASK_CR",
}
ZTF_BAD_PIXEL_MASK = 6141


@dataclass
class Source:
    source_id: str
    ra: float
    dec: float
    row: dict[str, str]


@dataclass
class Observation:
    row: dict[str, Any]
    source_type: str
    source_id: str
    survey: str
    night: str
    image_id: str
    ccd_id: str
    filter_name: str
    band: str
    rate: float
    rate_err: float
    rate_surface: tuple[float, ...]
    rate_err_surface: tuple[float, ...]
    fwhm_grid: tuple[float, ...]
    centroid_grid: tuple[float, ...]
    fwhm_prior: float
    host_rate: float | None
    host_rate_err: float | None
    point_host_rate_cov: float
    mag_inst: float | None
    mag_err: float | None
    ref_mag: float | None
    ref_mag_err: float
    color: float
    obs_mjd: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast joint Moffat calibration and AGN light-curve inference.")
    parser.add_argument("--standards", required=True, type=Path, help="Ivezic standard-star CSV.")
    parser.add_argument("--target-catalog", required=True, type=Path, help="AGN target CSV.")
    parser.add_argument("--image-manifest", required=True, type=Path, help="Combined image manifest.")
    parser.add_argument("--out-compressed", type=Path, help="Optional compressed source likelihood CSV.")
    parser.add_argument("--out-zeropoints", required=True, type=Path)
    parser.add_argument("--out-stars", required=True, type=Path)
    parser.add_argument("--out-agn", required=True, type=Path)
    parser.add_argument("--out-params", type=Path)
    parser.add_argument("--max-stars-per-image", type=int, default=40)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--stamp-radius", type=int, default=15)
    parser.add_argument("--edge-margin", type=int, default=20)
    parser.add_argument("--moffat-fwhm-pix", type=float, default=3.0)
    parser.add_argument("--moffat-beta", type=float, default=3.5)
    parser.add_argument("--host-sersic-n", type=float, default=1.0, help="Fixed Sersic index for the AGN host component.")
    parser.add_argument("--host-reff-pix", type=float, default=5.0, help="Fixed AGN host effective radius in pixels.")
    parser.add_argument("--host-axis-ratio", type=float, default=0.8, help="Fixed AGN host minor/major axis ratio.")
    parser.add_argument("--host-pa-deg", type=float, default=0.0, help="Fixed AGN host position angle, degrees east of +x in stamp coordinates.")
    parser.add_argument("--prior-host-mag-loc", type=float, default=22.0)
    parser.add_argument("--prior-host-mag-sigma", type=float, default=5.0)
    parser.add_argument("--ignore-header-seeing", action="store_true", help="Use --moffat-fwhm-pix for every image instead of FITS seeing metadata.")
    parser.add_argument("--min-moffat-fwhm-pix", type=float, default=1.0)
    parser.add_argument("--max-moffat-fwhm-pix", type=float, default=12.0)
    parser.add_argument("--fix-image-fwhm", action="store_true", help="Do not sample one FWHM per image; use the metadata/fallback FWHM.")
    parser.add_argument("--fwhm-grid-size", type=int, default=7)
    parser.add_argument("--fwhm-grid-log-span", type=float, default=0.35)
    parser.add_argument("--prior-log-fwhm-sigma", type=float, default=0.25)
    parser.add_argument("--centroid-grid-size", type=int, default=5, help="Odd grid size used when fitting each source/survey centroid once. Default: 5.")
    parser.add_argument("--centroid-grid-radius-pix", type=float, default=1.0, help="Maximum source/survey centroid shift around the WCS position in pixels. Default: 1.")
    parser.add_argument("--fix-centroids", action="store_true", help="Do not sample source/survey centroid shifts; evaluate the compressed likelihood at the WCS centroid.")
    parser.add_argument("--prior-centroid-sigma-pix", type=float, default=0.5)
    parser.add_argument("--default-noise", type=float, default=1.0)
    parser.add_argument("--reference-system", choices=["ps1", "native"], default="ps1")
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--student-t-df", type=float, default=5.0)
    parser.add_argument("--hide-progress", action="store_true")
    parser.add_argument("--prior-survey-filter-sigma", type=float, default=20.0)
    parser.add_argument("--prior-night-sigma", type=float, default=0.05)
    parser.add_argument("--prior-image-sigma", type=float, default=0.05)
    parser.add_argument("--prior-ccd-sigma", type=float, default=0.03)
    parser.add_argument("--prior-color-sigma", type=float, default=0.1)
    parser.add_argument("--prior-intrinsic-scatter-sigma", type=float, default=0.03)
    parser.add_argument("--initial-intrinsic-scatter", type=float, default=0.2)
    parser.add_argument("--prior-star-mag-floor", type=float, default=0.01, help="Minimum catalog-anchor uncertainty for latent constant standard-star magnitudes.")
    return parser.parse_args()


def require_numpyro():
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ModuleNotFoundError as exc:
        raise RuntimeError("This joint model requires jax and numpyro in the active environment.") from exc
    return jax, jnp, numpyro, dist, MCMC, NUTS


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


def read_sources(path: Path) -> list[Source]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sources = []
    for row in rows:
        source_id = (row.get("star_id") or row.get("name") or row.get("source_id") or "").strip()
        ra = parse_float(row.get("ra"))
        dec = parse_float(row.get("dec"))
        if source_id and ra is not None and dec is not None:
            sources.append(Source(source_id, ra, dec, row))
    if not sources:
        raise RuntimeError(f"No sources found in {path}")
    return sources


def ps1_from_sdss(values: Mapping[str, float], band: str) -> float | None:
    mag = values.get(band)
    g = values.get("g")
    i = values.get("i")
    if mag is None or g is None or i is None:
        return None
    if band not in PS1_SDSS_GI_COEFFS:
        return None
    color = g - i
    c0, c1, c2, c3 = PS1_SDSS_GI_COEFFS[band]
    return mag + c0 + c1 * color + c2 * color**2 + c3 * color**3


def reference_mag(source: Source, band: str, reference_system: str) -> float | None:
    values = {key: value for key, raw in source.row.items() if (value := parse_float(raw)) is not None}
    if reference_system == "ps1" and band in PS1_BANDS:
        return ps1_from_sdss(values, band)
    return values.get(band)


def reference_err(source: Source, band: str) -> float:
    return parse_float(source.row.get(f"{band}_err")) or 0.0


def reference_color(source: Source, reference_system: str) -> float | None:
    g = reference_mag(source, "g", reference_system)
    r = reference_mag(source, "r", reference_system)
    if g is None or r is None:
        return None
    return g - r


def first_2d_hdu(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=float)
            if data.ndim > 2:
                data = data[0]
            if data.ndim == 2:
                return data, hdu.header.copy()
    raise RuntimeError(f"No 2D image found in {path}")


def optional_2d_image(path_text: str | None) -> np.ndarray | None:
    if not path_text:
        return None
    path = Path(str(path_text).strip())
    if not path.exists():
        return None
    try:
        data, _header = first_2d_hdu(path)
    except RuntimeError:
        return None
    return data


def decode_sdss_fpm_plane(hdu: fits.BinTableHDU, shape: tuple[int, int]) -> np.ndarray:
    bad = np.zeros(shape, dtype=bool)
    if hdu.data is None:
        return bad
    for row in hdu.data:
        spans = np.asarray(row["s"], dtype=np.uint8)
        if spans.size < 6:
            continue
        spans = spans[: (spans.size // 6) * 6].reshape(-1, 6).astype(np.int32)
        rows = spans[:, 0] * 256 + spans[:, 1]
        col0 = spans[:, 2] * 256 + spans[:, 3]
        col1 = spans[:, 4] * 256 + spans[:, 5]
        for yy, x0, x1 in zip(rows, col0, col1):
            if yy < 0 or yy >= shape[0]:
                continue
            start = max(0, int(x0))
            stop = min(shape[1], int(x1) + 1)
            if stop > start:
                bad[int(yy), start:stop] = True
    return bad


def optional_sdss_fpm_good_mask(path_text: str | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not path_text:
        return None
    path = Path(str(path_text).strip())
    if not path.exists() or not path.name.startswith("fpM-"):
        return None
    try:
        with fits.open(path) as hdul:
            if len(hdul) < 12 or hdul[-1].data is None:
                return None
            bad = np.zeros(shape, dtype=bool)
            for row in hdul[-1].data:
                plane_name = str(row["attributeName"]).strip()
                plane_index = int(row["Value"]) + 1
                if plane_name in SDSS_FPM_BAD_PLANES and 0 < plane_index < len(hdul):
                    bad |= decode_sdss_fpm_plane(hdul[plane_index], shape)
            return ~bad
    except (OSError, RuntimeError, KeyError, IndexError, ValueError):
        return None


def load_noise_and_mask(
    image_row: Mapping[str, str],
    image_shape: tuple[int, int],
    image: np.ndarray,
    default_noise: float,
) -> tuple[float | np.ndarray, np.ndarray | None, str, str]:
    mask = None
    mask_source = "none"
    mask_path = str(image_row.get("mask_path", "")).strip()
    sdss_mask = optional_sdss_fpm_good_mask(mask_path, image_shape)
    if sdss_mask is not None:
        mask = sdss_mask
        mask_source = f"sdss_fpm:{mask_path}"
    else:
        mask_image = optional_2d_image(mask_path)
        if mask_image is not None and mask_image.shape == image_shape:
            if str(image_row.get("survey", "")).strip().lower() == "ztf":
                mask_bits = np.asarray(mask_image, dtype=np.int64)
                mask = np.isfinite(mask_image) & ((mask_bits & ZTF_BAD_PIXEL_MASK) == 0)
            else:
                mask = np.isfinite(mask_image) & (mask_image == 0)
            mask_source = mask_path
    if mask is not None:
        mask &= np.isfinite(image)
    elif not np.all(np.isfinite(image)):
        mask = np.isfinite(image)
        mask_source = str(image_row.get("mask_path", ""))

    noise_source = "robust_scalar"
    noise_image = optional_2d_image(image_row.get("noise_path"))
    if noise_image is not None and noise_image.shape == image_shape:
        noise = np.asarray(noise_image, dtype=float)
        noise_source = str(image_row.get("noise_path", ""))
    else:
        invvar_image = optional_2d_image(image_row.get("invvar_path"))
        if invvar_image is not None and invvar_image.shape == image_shape:
            invvar = np.asarray(invvar_image, dtype=float)
            noise = np.full(image_shape, np.nan, dtype=float)
            good = np.isfinite(invvar) & (invvar > 0)
            noise[good] = 1.0 / np.sqrt(invvar[good])
            noise_source = str(image_row.get("invvar_path", ""))
        else:
            variance_image = optional_2d_image(image_row.get("variance_path"))
            if variance_image is not None and variance_image.shape == image_shape:
                variance = np.asarray(variance_image, dtype=float)
                noise = np.full(image_shape, np.nan, dtype=float)
                good = np.isfinite(variance) & (variance > 0)
                noise[good] = np.sqrt(variance[good])
                noise_source = str(image_row.get("variance_path", ""))
            else:
                noise = robust_noise_scalar(image, default_noise)

    return noise, mask, noise_source, mask_source


def cutout(image: np.ndarray, x: float, y: float, radius: int, fill: float = 0.0) -> tuple[np.ndarray, tuple[int, int]]:
    cx = int(round(x))
    cy = int(round(y))
    size = 2 * radius + 1
    out = np.full((size, size), fill, dtype=float)
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    sx0, sx1 = max(x0, 0), min(x1, image.shape[1])
    sy0, sy1 = max(y0, 0), min(y1, image.shape[0])
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = image[sy0:sy1, sx0:sx1]
    return out, (x0, y0)


def cutout_batch(image: np.ndarray, x: np.ndarray, y: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fully-contained square cutouts for many positions."""

    cx = np.rint(np.asarray(x, dtype=float)).astype(int)
    cy = np.rint(np.asarray(y, dtype=float)).astype(int)
    offsets = np.arange(-int(radius), int(radius) + 1, dtype=int)
    yy = cy[:, None, None] + offsets[None, :, None]
    xx = cx[:, None, None] + offsets[None, None, :]
    return image[yy, xx], cx - int(radius), cy - int(radius)


def robust_noise_scalar(image: np.ndarray, default: float) -> float:
    finite = image[np.isfinite(image)]
    if finite.size > 10:
        med = np.nanmedian(finite)
        sigma = 1.4826 * np.nanmedian(np.abs(finite - med))
        if np.isfinite(sigma) and sigma > 0:
            return float(sigma)
    return float(default)


def obs_mjd(header: fits.Header) -> float | None:
    for key in ("MJD-OBS", "MJD", "MJDATE"):
        value = parse_float(str(header.get(key, "")))
        if value is not None:
            return value
    obsjd = parse_float(str(header.get("OBSJD", "")))
    if obsjd is not None:
        return obsjd - 2400000.5
    tai = parse_float(str(header.get("TAI", "")))
    if tai is not None:
        return tai / 86400.0
    return None


def pixel_scale_arcsec(header: fits.Header) -> float | None:
    value = parse_float(str(header.get("PIXSCALE", "")))
    if value is not None and value > 0:
        return value
    cdelt1 = parse_float(str(header.get("CDELT1", "")))
    if cdelt1 is not None and cdelt1 != 0:
        return abs(cdelt1) * 3600.0
    cd11 = parse_float(str(header.get("CD1_1", "")))
    cd12 = parse_float(str(header.get("CD1_2", ""))) or 0.0
    if cd11 is not None:
        scale = math.hypot(cd11, cd12) * 3600.0
        if scale > 0:
            return scale
    return None


def image_fwhm_pix(header: fits.Header, survey: str, args: argparse.Namespace) -> tuple[float, str]:
    fallback = float(args.moffat_fwhm_pix)
    if args.ignore_header_seeing:
        return fallback, "fallback"

    survey_name = survey.lower()
    if survey_name == "panstarrs":
        value = parse_float(str(header.get("CHIP.SEEING", "")))
        if value is not None and value > 0:
            return float(np.clip(value, args.min_moffat_fwhm_pix, args.max_moffat_fwhm_pix)), "CHIP.SEEING_pix"

    seeing_arcsec = parse_float(str(header.get("SEEING", "")))
    scale = pixel_scale_arcsec(header)
    if seeing_arcsec is not None and seeing_arcsec > 0 and scale is not None and scale > 0:
        value = seeing_arcsec / scale
        return float(np.clip(value, args.min_moffat_fwhm_pix, args.max_moffat_fwhm_pix)), "SEEING_arcsec"

    for key in ("FWHMPSF", "PSF_FWHM", "FWHM"):
        value = parse_float(str(header.get(key, "")))
        if value is not None and value > 0:
            return float(np.clip(value, args.min_moffat_fwhm_pix, args.max_moffat_fwhm_pix)), key

    return fallback, "fallback"


def fwhm_grid_around(prior_fwhm: float, args: argparse.Namespace) -> np.ndarray:
    prior = float(np.clip(prior_fwhm, args.min_moffat_fwhm_pix, args.max_moffat_fwhm_pix))
    if args.fix_image_fwhm or args.fwhm_grid_size <= 1:
        return np.asarray([prior], dtype=float)
    lo = max(args.min_moffat_fwhm_pix, prior * math.exp(-args.fwhm_grid_log_span))
    hi = min(args.max_moffat_fwhm_pix, prior * math.exp(args.fwhm_grid_log_span))
    if not hi > lo:
        return np.full(int(args.fwhm_grid_size), prior, dtype=float)
    return np.exp(np.linspace(math.log(lo), math.log(hi), int(args.fwhm_grid_size))).astype(float)


def centroid_offset_grid(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if getattr(args, "fix_centroids", False):
        return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)
    size = max(1, int(args.centroid_grid_size))
    if size % 2 == 0:
        size += 1
    radius = max(0.0, float(args.centroid_grid_radius_pix))
    if size == 1 or radius == 0.0:
        return np.asarray([0.0], dtype=float), np.asarray([0.0], dtype=float)
    axis = np.linspace(-radius, radius, size, dtype=float)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx.ravel(), yy.ravel()


def encode_float_list(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.8g}" for value in values)


def sersic_unit_flux_batch(
    shape: tuple[int, int],
    *,
    reff_pix: float,
    n_sersic: float,
    dx_pix: np.ndarray | float = 0.0,
    dy_pix: np.ndarray | float = 0.0,
    q: float = 0.8,
    theta_rad: float = 0.0,
) -> np.ndarray:
    """Render a batch of unit-flux elliptical Sersic images."""

    dx = np.atleast_1d(np.asarray(dx_pix, dtype=float))
    dy = np.atleast_1d(np.asarray(dy_pix, dtype=float))
    dx, dy = np.broadcast_arrays(dx, dy)
    ny, nx = int(shape[0]), int(shape[1])
    y, x = np.indices((ny, nx), dtype=float)
    x = x[None, :, :] - (nx - 1) / 2.0 - dx[:, None, None]
    y = y[None, :, :] - (ny - 1) / 2.0 - dy[:, None, None]
    q = float(np.clip(q, 1.0e-3, 1.0))
    n = float(np.clip(n_sersic, 0.3, 8.0))
    reff = max(float(reff_pix), 1.0e-3)
    cos_t = np.cos(float(theta_rad))
    sin_t = np.sin(float(theta_rad))
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t
    radius = np.sqrt(x_rot**2 + (y_rot / q) ** 2 + 1.0e-12)
    b_n = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    images = np.exp(-b_n * ((radius / reff) ** (1.0 / n) - 1.0))
    images = np.clip(images, 0.0, np.inf)
    totals = np.sum(images, axis=(1, 2), keepdims=True)
    bad = ~np.isfinite(totals) | (totals <= 0)
    if np.any(bad):
        images[bad[:, 0, 0]] = 0.0
        totals[bad] = 1.0
    return images / totals


def fft_convolve_same_batch(images: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Convolve paired image/kernel stamps and return same-sized centered crops."""

    images = np.asarray(images, dtype=float)
    kernels = np.asarray(kernels, dtype=float)
    if images.shape != kernels.shape or images.ndim != 3:
        raise ValueError("images and kernels must have matching shape (n, ny, nx).")
    n, ny, nx = images.shape
    fft_shape = (2 * ny - 1, 2 * nx - 1)
    image_fft = np.fft.rfftn(images, s=fft_shape, axes=(1, 2))
    kernel_fft = np.fft.rfftn(kernels, s=fft_shape, axes=(1, 2))
    full = np.fft.irfftn(image_fft * kernel_fft, s=fft_shape, axes=(1, 2))
    y0 = (ny - 1) // 2
    x0 = (nx - 1) // 2
    out = np.clip(full[:, y0 : y0 + ny, x0 : x0 + nx], 0.0, np.inf)
    totals = np.sum(out, axis=(1, 2), keepdims=True)
    good = np.isfinite(totals) & (totals > 0)
    return np.where(good, out / np.maximum(totals, 1.0e-300), 0.0)


def fit_point_host_flux_batch(
    data: np.ndarray,
    noise: float | np.ndarray,
    psf: np.ndarray,
    host: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Weighted linear solve for point-source flux, host flux, and background."""

    data = np.asarray(data, dtype=float)
    psf = np.asarray(psf, dtype=float)
    host = np.asarray(host, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if data.ndim != 3 or psf.shape != data.shape or host.shape != data.shape:
        raise ValueError("data, psf, and host must have matching shape (n, ny, nx).")
    scalar_noise = noise.ndim == 0
    if scalar_noise:
        noise_value = float(noise)
        valid = np.isfinite(data) & np.isfinite(psf) & np.isfinite(host) & np.isfinite(noise_value) & (noise_value > 0)
        weight = np.ones_like(data) / max(noise_value, 1.0e-300) ** 2
    else:
        if noise.shape == data.shape[1:]:
            noise = np.broadcast_to(noise[None, :, :], data.shape)
        elif noise.shape != data.shape:
            raise ValueError("noise must be scalar, stamp-shaped, or match data batch shape.")
        valid = np.isfinite(data) & np.isfinite(noise) & np.isfinite(psf) & np.isfinite(host) & (noise > 0)
        weight = 1.0 / np.maximum(noise, 1.0e-300) ** 2
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape == data.shape[1:]:
            mask_array = np.broadcast_to(mask_array[None, :, :], data.shape)
        elif mask_array.shape != data.shape:
            raise ValueError("mask must be stamp-shaped or match data batch shape.")
        valid &= mask_array
    weight = np.where(valid, weight, 0.0)
    point_flux = np.full(data.shape[0], np.nan)
    host_flux = np.full(data.shape[0], np.nan)
    background = np.full(data.shape[0], np.nan)
    point_var = np.full(data.shape[0], np.nan)
    host_var = np.full(data.shape[0], np.nan)
    point_host_cov = np.full(data.shape[0], np.nan)
    reduced_chi2 = np.full(data.shape[0], np.nan)
    npixels = np.sum(valid, axis=(1, 2)).astype(int)

    for i in range(data.shape[0]):
        if npixels[i] < 4:
            continue
        y = data[i][valid[i]]
        p = psf[i][valid[i]]
        h = host[i][valid[i]]
        one = np.ones_like(p)
        w = weight[i][valid[i]]
        design = np.asarray(
            [
                [np.sum(w * p * p), np.sum(w * p * h), np.sum(w * p)],
                [np.sum(w * h * p), np.sum(w * h * h), np.sum(w * h)],
                [np.sum(w * p), np.sum(w * h), np.sum(w)],
            ],
            dtype=float,
        )
        rhs = np.asarray([np.sum(w * p * y), np.sum(w * h * y), np.sum(w * y)], dtype=float)
        try:
            cov = np.linalg.inv(design)
        except np.linalg.LinAlgError:
            continue
        params = cov @ rhs
        model = params[0] * p + params[1] * h + params[2] * one
        dof = max(int(npixels[i]) - 3, 1)
        point_flux[i] = float(params[0])
        host_flux[i] = float(params[1])
        background[i] = float(params[2])
        point_var[i] = float(max(cov[0, 0], 0.0))
        host_var[i] = float(max(cov[1, 1], 0.0))
        point_host_cov[i] = float(cov[0, 1])
        reduced_chi2[i] = float(np.sum(w * (y - model) ** 2) / dof)

    return {
        "point_flux": point_flux,
        "host_flux": host_flux,
        "background": background,
        "point_var": point_var,
        "host_var": host_var,
        "point_host_cov": point_host_cov,
        "reduced_chi2": reduced_chi2,
        "npixels": npixels,
    }


def marginalize_flux_grid(
    grid_flux: np.ndarray,
    grid_flux_err: np.ndarray,
    grid_background: np.ndarray,
    grid_chi2: np.ndarray,
    grid_npixels: np.ndarray,
    fwhm_grid: np.ndarray,
    centroid_grid_dx: np.ndarray,
    centroid_grid_dy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Marginalize the local FWHM/centroid grid to one Gaussian flux likelihood.

    The matched-filter solver returns a conditional Gaussian likelihood for the
    flux at each FWHM/centroid grid point. We approximate the local marginalized
    likelihood as a Gaussian mixture collapsed to its first two moments.
    """

    n_obs = int(grid_flux.shape[0])
    n_centroid = int(centroid_grid_dx.size)
    flat_flux = grid_flux.reshape(n_obs, -1)
    flat_flux_err = grid_flux_err.reshape(n_obs, -1)
    flat_background = grid_background.reshape(n_obs, -1)
    flat_chi2 = grid_chi2.reshape(n_obs, -1)
    flat_npixels = grid_npixels.reshape(n_obs, -1).astype(float)
    dof = np.maximum(flat_npixels - 2.0, 1.0)
    chi2_abs = flat_chi2 * dof
    valid = (
        np.isfinite(flat_flux)
        & np.isfinite(flat_flux_err)
        & (flat_flux_err > 0)
        & np.isfinite(flat_background)
        & np.isfinite(chi2_abs)
        & (flat_npixels >= 3)
    )

    flux = np.full(n_obs, np.nan, dtype=float)
    flux_err = np.full(n_obs, np.nan, dtype=float)
    background = np.full(n_obs, np.nan, dtype=float)
    reduced_chi2 = np.full(n_obs, np.nan, dtype=float)
    npixels = np.zeros(n_obs, dtype=int)
    best_fwhm = np.full(n_obs, np.nan, dtype=float)
    best_dx = np.full(n_obs, np.nan, dtype=float)
    best_dy = np.full(n_obs, np.nan, dtype=float)
    grid_points = np.zeros(n_obs, dtype=int)

    for i in range(n_obs):
        good = valid[i]
        if not np.any(good):
            continue
        chi = np.where(good, chi2_abs[i], np.inf)
        best = int(np.argmin(chi))
        delta = chi - chi[best]
        logw = np.where(good, -0.5 * delta, -np.inf)
        max_logw = float(np.max(logw[good]))
        weights = np.exp(logw - max_logw)
        weights[~good] = 0.0
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        weights /= weight_sum

        conditional_flux = np.where(good, flat_flux[i], 0.0)
        conditional_err = np.where(good, flat_flux_err[i], 0.0)
        mean = float(np.sum(weights * conditional_flux))
        second = float(np.sum(weights * (conditional_err**2 + conditional_flux**2)))
        min_err = float(np.nanmin(flat_flux_err[i, good]))
        var = max(second - mean**2, min_err**2, 1.0e-30)

        flux[i] = mean
        flux_err[i] = math.sqrt(var)
        background[i] = float(flat_background[i, best])
        reduced_chi2[i] = float(flat_chi2[i, best])
        npixels[i] = int(flat_npixels[i, best])
        best_fwhm[i] = float(fwhm_grid[best // n_centroid])
        best_centroid = best % n_centroid
        best_dx[i] = float(centroid_grid_dx[best_centroid])
        best_dy[i] = float(centroid_grid_dy[best_centroid])
        grid_points[i] = int(np.sum(good))

    return {
        "flux": flux,
        "flux_err": flux_err,
        "background": background,
        "reduced_chi2": reduced_chi2,
        "npixels": npixels,
        "best_fwhm": best_fwhm,
        "best_dx": best_dx,
        "best_dy": best_dy,
        "grid_points": grid_points,
    }


def marginalize_point_host_grid(
    grid_point_flux: np.ndarray,
    grid_host_flux: np.ndarray,
    grid_point_var: np.ndarray,
    grid_host_var: np.ndarray,
    grid_point_host_cov: np.ndarray,
    grid_background: np.ndarray,
    grid_chi2: np.ndarray,
    grid_npixels: np.ndarray,
    fwhm_grid: np.ndarray,
    centroid_grid_dx: np.ndarray,
    centroid_grid_dy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Collapse local point+host Gaussian mixtures to one 2D Gaussian per AGN stamp."""

    n_obs = int(grid_point_flux.shape[0])
    n_centroid = int(centroid_grid_dx.size)
    flat_point = grid_point_flux.reshape(n_obs, -1)
    flat_host = grid_host_flux.reshape(n_obs, -1)
    flat_point_var = grid_point_var.reshape(n_obs, -1)
    flat_host_var = grid_host_var.reshape(n_obs, -1)
    flat_cov = grid_point_host_cov.reshape(n_obs, -1)
    flat_background = grid_background.reshape(n_obs, -1)
    flat_chi2 = grid_chi2.reshape(n_obs, -1)
    flat_npixels = grid_npixels.reshape(n_obs, -1).astype(float)
    dof = np.maximum(flat_npixels - 3.0, 1.0)
    chi2_abs = flat_chi2 * dof
    valid = (
        np.isfinite(flat_point)
        & np.isfinite(flat_host)
        & np.isfinite(flat_point_var)
        & np.isfinite(flat_host_var)
        & np.isfinite(flat_cov)
        & (flat_point_var > 0)
        & (flat_host_var > 0)
        & np.isfinite(flat_background)
        & np.isfinite(chi2_abs)
        & (flat_npixels >= 4)
    )

    point_flux = np.full(n_obs, np.nan)
    host_flux = np.full(n_obs, np.nan)
    point_var = np.full(n_obs, np.nan)
    host_var = np.full(n_obs, np.nan)
    point_host_cov = np.full(n_obs, np.nan)
    background = np.full(n_obs, np.nan)
    reduced_chi2 = np.full(n_obs, np.nan)
    npixels = np.zeros(n_obs, dtype=int)
    best_fwhm = np.full(n_obs, np.nan)
    best_dx = np.full(n_obs, np.nan)
    best_dy = np.full(n_obs, np.nan)
    grid_points = np.zeros(n_obs, dtype=int)

    for i in range(n_obs):
        good = valid[i]
        if not np.any(good):
            continue
        chi = np.where(good, chi2_abs[i], np.inf)
        best = int(np.argmin(chi))
        logw = np.where(good, -0.5 * (chi - chi[best]), -np.inf)
        max_logw = float(np.max(logw[good]))
        weights = np.exp(logw - max_logw)
        weights[~good] = 0.0
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        weights /= weight_sum

        mu = np.vstack([np.where(good, flat_point[i], 0.0), np.where(good, flat_host[i], 0.0)])
        mean = np.sum(weights[None, :] * mu, axis=1)
        second00 = np.sum(weights * (np.where(good, flat_point_var[i], 0.0) + mu[0] ** 2))
        second11 = np.sum(weights * (np.where(good, flat_host_var[i], 0.0) + mu[1] ** 2))
        second01 = np.sum(weights * (np.where(good, flat_cov[i], 0.0) + mu[0] * mu[1]))
        min_point_var = float(np.nanmin(flat_point_var[i, good]))
        min_host_var = float(np.nanmin(flat_host_var[i, good]))
        cov01 = float(second01 - mean[0] * mean[1])
        var0 = max(float(second00 - mean[0] ** 2), min_point_var, 1.0e-30)
        var1 = max(float(second11 - mean[1] ** 2), min_host_var, 1.0e-30)
        limit = 0.95 * math.sqrt(var0 * var1)

        point_flux[i] = float(mean[0])
        host_flux[i] = float(mean[1])
        point_var[i] = var0
        host_var[i] = var1
        point_host_cov[i] = float(np.clip(cov01, -limit, limit))
        background[i] = float(flat_background[i, best])
        reduced_chi2[i] = float(flat_chi2[i, best])
        npixels[i] = int(flat_npixels[i, best])
        best_fwhm[i] = float(fwhm_grid[best // n_centroid])
        best_centroid = best % n_centroid
        best_dx[i] = float(centroid_grid_dx[best_centroid])
        best_dy[i] = float(centroid_grid_dy[best_centroid])
        grid_points[i] = int(np.sum(good))

    return {
        "point_flux": point_flux,
        "host_flux": host_flux,
        "point_var": point_var,
        "host_var": host_var,
        "point_host_cov": point_host_cov,
        "background": background,
        "reduced_chi2": reduced_chi2,
        "npixels": npixels,
        "best_fwhm": best_fwhm,
        "best_dx": best_dx,
        "best_dy": best_dy,
        "grid_points": grid_points,
    }


def subset_noise(noise: float | np.ndarray, indices: np.ndarray) -> float | np.ndarray:
    if np.ndim(noise) == 0:
        return float(noise)
    return np.asarray(noise)[indices]


def subset_mask(mask: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    return np.asarray(mask)[indices]


def compress_observations(
    standards: list[Source],
    targets: list[Source],
    manifest: Path,
    args: argparse.Namespace,
) -> list[Observation]:
    from jaguar.fastphot import matched_filter_flux_batch, moffat_unit_flux_batch

    observations: list[Observation] = []
    with manifest.open(newline="") as handle:
        image_rows = list(csv.DictReader(handle))
    if args.max_images is not None and args.max_images > 0:
        image_rows = image_rows[: args.max_images]
    all_sources = [("standard", source) for source in standards] + [("agn", source) for source in targets]
    source_types = np.asarray([kind for kind, _source in all_sources])
    source_objects = [source for _kind, source in all_sources]
    source_coords = np.asarray([[source.ra, source.dec] for source in source_objects], dtype=float)
    standard_source_mask = source_types == "standard"

    for image_index, image_row in enumerate(image_rows, start=1):
        image_path = Path((image_row.get("image_path") or "").strip())
        filter_name = (image_row.get("filter") or "").strip()
        band = FILTER_TO_REF.get(filter_name)
        if not band or not image_path.exists():
            continue
        print(f"compressing {image_index}/{len(image_rows)} {image_row.get('survey')} {filter_name} {image_path}", flush=True)
        image, header = first_2d_hdu(image_path)
        noise_model, pixel_mask, noise_source, mask_source = load_noise_and_mask(
            image_row,
            image.shape,
            image,
            args.default_noise,
        )
        fwhm_prior, fwhm_source = image_fwhm_pix(header, image_row.get("survey", ""), args)
        fwhm_grid = fwhm_grid_around(fwhm_prior, args)
        wcs = WCS(header, naxis=2)
        mjd = obs_mjd(header)
        pixels = wcs.all_world2pix(source_coords, 0)
        x_all = pixels[:, 0]
        y_all = pixels[:, 1]
        margin = max(args.edge_margin, args.stamp_radius + 1)
        in_image = (
            np.isfinite(x_all)
            & np.isfinite(y_all)
            & (margin <= x_all)
            & (x_all < image.shape[1] - margin)
            & (margin <= y_all)
            & (y_all < image.shape[0] - margin)
        )
        standard_indices = np.flatnonzero(in_image & standard_source_mask)
        if args.max_stars_per_image is not None and args.max_stars_per_image > 0:
            standard_indices = standard_indices[: args.max_stars_per_image]
        agn_indices = np.flatnonzero(in_image & ~standard_source_mask)
        selected_indices = np.concatenate([standard_indices, agn_indices])
        if selected_indices.size == 0:
            continue

        x = x_all[selected_indices]
        y = y_all[selected_indices]
        selected_is_agn = ~standard_source_mask[selected_indices]
        stamps, stamp_x0, stamp_y0 = cutout_batch(image, x, y, args.stamp_radius)
        if np.ndim(noise_model) == 0:
            noise_stamps: float | np.ndarray = float(noise_model)
        else:
            noise_stamps, _noise_x0, _noise_y0 = cutout_batch(np.asarray(noise_model, dtype=float), x, y, args.stamp_radius)
        if pixel_mask is None:
            mask_stamps = None
        else:
            mask_stamps, _mask_x0, _mask_y0 = cutout_batch(pixel_mask, x, y, args.stamp_radius)
        wcs_dx = x - np.rint(x)
        wcs_dy = y - np.rint(y)
        centroid_grid_dx, centroid_grid_dy = centroid_offset_grid(args)
        centroid_axis = np.unique(centroid_grid_dx)
        survey_name = str(image_row.get("survey", ""))
        grid_flux = []
        grid_flux_err = []
        grid_background = []
        grid_chi2 = []
        grid_npixels = []
        agn_grid_point_flux = []
        agn_grid_host_flux = []
        agn_grid_point_var = []
        agn_grid_host_var = []
        agn_grid_point_host_cov = []
        agn_grid_background = []
        agn_grid_chi2 = []
        agn_grid_npixels = []
        for grid_fwhm in fwhm_grid:
            fwhm_flux = []
            fwhm_flux_err = []
            fwhm_background = []
            fwhm_chi2 = []
            fwhm_npixels = []
            agn_fwhm_point_flux = []
            agn_fwhm_host_flux = []
            agn_fwhm_point_var = []
            agn_fwhm_host_var = []
            agn_fwhm_point_host_cov = []
            agn_fwhm_background = []
            agn_fwhm_chi2 = []
            agn_fwhm_npixels = []
            for centroid_dx, centroid_dy in zip(centroid_grid_dx, centroid_grid_dy):
                psfs = moffat_unit_flux_batch(
                    stamps.shape[1:],
                    fwhm_pix=float(grid_fwhm),
                    beta=args.moffat_beta,
                    dx_pix=wcs_dx + float(centroid_dx),
                    dy_pix=wcs_dy + float(centroid_dy),
                )
                results = matched_filter_flux_batch(stamps, noise_stamps, psfs, mask=mask_stamps, fit_background=True)
                fwhm_flux.append(np.asarray(results.flux, dtype=float))
                fwhm_flux_err.append(np.asarray(results.flux_err, dtype=float))
                fwhm_background.append(np.asarray(results.background, dtype=float))
                fwhm_chi2.append(np.asarray(results.reduced_chi2, dtype=float))
                fwhm_npixels.append(np.asarray(results.npixels, dtype=int))
                if np.any(selected_is_agn):
                    agn_psfs = psfs[selected_is_agn]
                    host_intrinsic = sersic_unit_flux_batch(
                        stamps.shape[1:],
                        reff_pix=args.host_reff_pix,
                        n_sersic=args.host_sersic_n,
                        dx_pix=wcs_dx[selected_is_agn] + float(centroid_dx),
                        dy_pix=wcs_dy[selected_is_agn] + float(centroid_dy),
                        q=args.host_axis_ratio,
                        theta_rad=math.radians(args.host_pa_deg),
                    )
                    host_basis = fft_convolve_same_batch(host_intrinsic, agn_psfs)
                    agn_results = fit_point_host_flux_batch(
                        stamps[selected_is_agn],
                        subset_noise(noise_stamps, selected_is_agn),
                        agn_psfs,
                        host_basis,
                        mask=subset_mask(mask_stamps, selected_is_agn),
                    )
                    agn_fwhm_point_flux.append(np.asarray(agn_results["point_flux"], dtype=float))
                    agn_fwhm_host_flux.append(np.asarray(agn_results["host_flux"], dtype=float))
                    agn_fwhm_point_var.append(np.asarray(agn_results["point_var"], dtype=float))
                    agn_fwhm_host_var.append(np.asarray(agn_results["host_var"], dtype=float))
                    agn_fwhm_point_host_cov.append(np.asarray(agn_results["point_host_cov"], dtype=float))
                    agn_fwhm_background.append(np.asarray(agn_results["background"], dtype=float))
                    agn_fwhm_chi2.append(np.asarray(agn_results["reduced_chi2"], dtype=float))
                    agn_fwhm_npixels.append(np.asarray(agn_results["npixels"], dtype=int))
            grid_flux.append(np.stack(fwhm_flux, axis=1))
            grid_flux_err.append(np.stack(fwhm_flux_err, axis=1))
            grid_background.append(np.stack(fwhm_background, axis=1))
            grid_chi2.append(np.stack(fwhm_chi2, axis=1))
            grid_npixels.append(np.stack(fwhm_npixels, axis=1))
            if np.any(selected_is_agn):
                agn_grid_point_flux.append(np.stack(agn_fwhm_point_flux, axis=1))
                agn_grid_host_flux.append(np.stack(agn_fwhm_host_flux, axis=1))
                agn_grid_point_var.append(np.stack(agn_fwhm_point_var, axis=1))
                agn_grid_host_var.append(np.stack(agn_fwhm_host_var, axis=1))
                agn_grid_point_host_cov.append(np.stack(agn_fwhm_point_host_cov, axis=1))
                agn_grid_background.append(np.stack(agn_fwhm_background, axis=1))
                agn_grid_chi2.append(np.stack(agn_fwhm_chi2, axis=1))
                agn_grid_npixels.append(np.stack(agn_fwhm_npixels, axis=1))
        grid_flux_arr = np.stack(grid_flux, axis=1)
        grid_flux_err_arr = np.stack(grid_flux_err, axis=1)
        grid_background_arr = np.stack(grid_background, axis=1)
        grid_chi2_arr = np.stack(grid_chi2, axis=1)
        grid_npixels_arr = np.stack(grid_npixels, axis=1)
        marginalized = marginalize_flux_grid(
            grid_flux_arr,
            grid_flux_err_arr,
            grid_background_arr,
            grid_chi2_arr,
            grid_npixels_arr,
            fwhm_grid,
            centroid_grid_dx,
            centroid_grid_dy,
        )
        stamp_area = int(stamps.shape[1] * stamps.shape[2])
        mask_fraction = np.clip(1.0 - marginalized["npixels"].astype(float) / max(stamp_area, 1), 0.0, 1.0)
        fitted_center_x = wcs_dx + marginalized["best_dx"]
        fitted_center_y = wcs_dy + marginalized["best_dy"]
        agn_marginalized: dict[str, np.ndarray] | None = None
        agn_batch_lookup: dict[int, int] = {}
        if np.any(selected_is_agn):
            agn_marginalized = marginalize_point_host_grid(
                np.stack(agn_grid_point_flux, axis=1),
                np.stack(agn_grid_host_flux, axis=1),
                np.stack(agn_grid_point_var, axis=1),
                np.stack(agn_grid_host_var, axis=1),
                np.stack(agn_grid_point_host_cov, axis=1),
                np.stack(agn_grid_background, axis=1),
                np.stack(agn_grid_chi2, axis=1),
                np.stack(agn_grid_npixels, axis=1),
                fwhm_grid,
                centroid_grid_dx,
                centroid_grid_dy,
            )
            agn_batch_lookup = {int(batch_i): int(agn_i) for agn_i, batch_i in enumerate(np.flatnonzero(selected_is_agn))}

        for batch_i, source_index in enumerate(selected_indices):
            source_type = str(source_types[source_index])
            source = source_objects[source_index]
            if source_type == "agn" and agn_marginalized is not None:
                agn_i = agn_batch_lookup[batch_i]
                result_flux = float(agn_marginalized["point_flux"][agn_i])
                result_flux_err = float(math.sqrt(max(float(agn_marginalized["point_var"][agn_i]), 0.0)))
                result_host_flux = float(agn_marginalized["host_flux"][agn_i])
                result_host_flux_err = float(math.sqrt(max(float(agn_marginalized["host_var"][agn_i]), 0.0)))
                result_point_host_cov = float(agn_marginalized["point_host_cov"][agn_i])
                result_background = float(agn_marginalized["background"][agn_i])
                result_chi2 = float(agn_marginalized["reduced_chi2"][agn_i])
                result_npixels = int(agn_marginalized["npixels"][agn_i])
                result_best_fwhm = float(agn_marginalized["best_fwhm"][agn_i])
                result_best_dx = float(agn_marginalized["best_dx"][agn_i])
                result_best_dy = float(agn_marginalized["best_dy"][agn_i])
                result_grid_points = int(agn_marginalized["grid_points"][agn_i])
            else:
                result_flux = float(marginalized["flux"][batch_i])
                result_flux_err = float(marginalized["flux_err"][batch_i])
                result_host_flux = np.nan
                result_host_flux_err = np.nan
                result_point_host_cov = 0.0
                result_background = float(marginalized["background"][batch_i])
                result_chi2 = float(marginalized["reduced_chi2"][batch_i])
                result_npixels = int(marginalized["npixels"][batch_i])
                result_best_fwhm = float(marginalized["best_fwhm"][batch_i])
                result_best_dx = float(marginalized["best_dx"][batch_i])
                result_best_dy = float(marginalized["best_dy"][batch_i])
                result_grid_points = int(marginalized["grid_points"][batch_i])
            exptime = parse_float(image_row.get("exptime")) or 1.0
            if not np.isfinite(result_flux) or not np.isfinite(result_flux_err) or result_flux_err <= 0:
                continue
            rate_surface = grid_flux_arr[batch_i] / exptime
            rate_err_surface = grid_flux_err_arr[batch_i] / exptime
            rate = result_flux / exptime
            rate_err = result_flux_err / exptime
            host_rate = result_host_flux / exptime if np.isfinite(result_host_flux) else np.nan
            host_rate_err = result_host_flux_err / exptime if np.isfinite(result_host_flux_err) else np.nan
            point_host_rate_cov = result_point_host_cov / exptime**2 if np.isfinite(result_point_host_cov) else 0.0
            mag_inst = -2.5 * math.log10(rate) if rate > 0 else None
            mag_err = 2.5 / math.log(10) * rate_err / rate if rate > 0 else None
            ref_mag = reference_mag(source, band, args.reference_system) if source_type == "standard" else None
            color = reference_color(source, args.reference_system) if source_type == "standard" else 0.0
            if source_type == "standard" and (ref_mag is None or color is None):
                continue
            row = {
                "source_type": source_type,
                "compression_mode": "marginalized_gaussian_grid",
                "star_id": source.source_id,
                "ra": source.ra,
                "dec": source.dec,
                "survey": image_row.get("survey", ""),
                "night": image_row.get("night", ""),
                "obs_mjd": "" if mjd is None else mjd,
                "image_id": image_row.get("image_id", ""),
                "ccd_id": image_row.get("ccd_id", ""),
                "filter": filter_name,
                "band": band,
                "flux": result_flux,
                "flux_err": result_flux_err,
                "host_flux": "" if not np.isfinite(result_host_flux) else result_host_flux,
                "host_flux_err": "" if not np.isfinite(result_host_flux_err) else result_host_flux_err,
                "point_host_flux_cov": result_point_host_cov,
                "snr": result_flux / result_flux_err,
                "exptime": exptime,
                "rate": rate,
                "rate_err": rate_err,
                "host_rate": "" if not np.isfinite(host_rate) else host_rate,
                "host_rate_err": "" if not np.isfinite(host_rate_err) else host_rate_err,
                "point_host_rate_cov": point_host_rate_cov,
                "fwhm_grid_pix": encode_float_list(fwhm_grid),
                "centroid_grid_pix": encode_float_list(centroid_axis),
                "rate_surface": encode_float_list(rate_surface.ravel()),
                "rate_err_surface": encode_float_list(rate_err_surface.ravel()),
                "mag_inst": "" if mag_inst is None else mag_inst,
                "mag_err": "" if mag_err is None else mag_err,
                "x": float(x[batch_i]),
                "y": float(y[batch_i]),
                "stamp_x0": int(stamp_x0[batch_i]),
                "stamp_y0": int(stamp_y0[batch_i]),
                "wcs_center_x_pix": float(wcs_dx[batch_i]),
                "wcs_center_y_pix": float(wcs_dy[batch_i]),
                "fit_center_x_pix": float(wcs_dx[batch_i] + result_best_dx),
                "fit_center_y_pix": float(wcs_dy[batch_i] + result_best_dy),
                "fit_centroid_shift_x_pix": result_best_dx,
                "fit_centroid_shift_y_pix": result_best_dy,
                "fit_centroid_scope": "local_marginalized_grid",
                "moffat_fwhm_pix": fwhm_prior,
                "fitted_moffat_fwhm_pix": result_best_fwhm,
                "moffat_fwhm_source": fwhm_source,
                "moffat_beta": args.moffat_beta,
                "host_model": "sersic" if source_type == "agn" else "none",
                "host_sersic_n": args.host_sersic_n if source_type == "agn" else "",
                "host_reff_pix": args.host_reff_pix if source_type == "agn" else "",
                "host_axis_ratio": args.host_axis_ratio if source_type == "agn" else "",
                "host_pa_deg": args.host_pa_deg if source_type == "agn" else "",
                "noise_source": noise_source,
                "mask_source": mask_source,
                "background": result_background,
                "reduced_chi2": result_chi2,
                "npixels": result_npixels,
                "mask_fraction": float(np.clip(1.0 - result_npixels / max(stamp_area, 1), 0.0, 1.0)),
                "marginalized_grid_points": result_grid_points,
                "image_path": str(image_path),
                "ref_mag": "" if ref_mag is None else ref_mag,
                "ref_color": color,
            }
            observations.append(
                Observation(
                    row=row,
                    source_type=source_type,
                    source_id=source.source_id,
                    survey=str(row["survey"]),
                    night=str(row["night"]),
                    image_id=str(row["image_id"]),
                    ccd_id=str(row["ccd_id"]),
                    filter_name=filter_name,
                    band=band,
                    rate=rate,
                    rate_err=rate_err,
                    rate_surface=tuple(float(value) for value in rate_surface.ravel()),
                    rate_err_surface=tuple(float(value) for value in rate_err_surface.ravel()),
                    fwhm_grid=tuple(float(value) for value in fwhm_grid),
                    centroid_grid=tuple(float(value) for value in centroid_axis),
                    fwhm_prior=fwhm_prior,
                    host_rate=float(host_rate) if np.isfinite(host_rate) else None,
                    host_rate_err=float(host_rate_err) if np.isfinite(host_rate_err) else None,
                    point_host_rate_cov=float(point_host_rate_cov),
                    mag_inst=mag_inst,
                    mag_err=mag_err,
                    ref_mag=ref_mag,
                    ref_mag_err=reference_err(source, band) if source_type == "standard" else 0.0,
                    color=float(color or 0.0),
                    obs_mjd=mjd,
                )
            )
    if not observations:
        raise RuntimeError("No compressed observations were produced.")
    return observations


def index_labels(values: list[tuple[str, ...]]) -> tuple[np.ndarray, list[tuple[str, ...]]]:
    labels: list[tuple[str, ...]] = []
    lookup: dict[tuple[str, ...], int] = {}
    indices = []
    for value in values:
        if value not in lookup:
            lookup[value] = len(labels)
            labels.append(value)
        indices.append(lookup[value])
    return np.asarray(indices, dtype=int), labels


def build_arrays(observations: list[Observation]) -> tuple[dict[str, Any], dict[str, list[tuple[str, ...]]]]:
    cal_values = [(o.survey, o.band) for o in observations]
    night_values = [(o.survey, o.night, o.band) for o in observations]
    image_values = [(o.survey, o.night, o.image_id, o.ccd_id, o.band) for o in observations]
    ccd_values = [(o.survey, o.ccd_id, o.band) for o in observations]
    source_survey_values = [(o.source_id, o.survey) for o in observations]
    star_values = [(o.source_id, o.band) for o in observations if o.source_type == "standard"]
    agn_values = [(o.source_id, "" if o.obs_mjd is None else f"{o.obs_mjd:.8f}", o.band) for o in observations if o.source_type == "agn"]
    host_values = [(o.source_id, o.band) for o in observations if o.source_type == "agn"]
    sf_i, sf_labels = index_labels(cal_values)
    night_i, night_labels = index_labels(night_values)
    image_i, image_labels = index_labels(image_values)
    ccd_i, ccd_labels = index_labels(ccd_values)
    source_survey_i, source_survey_labels = index_labels(source_survey_values)
    star_i_raw, star_labels = index_labels(star_values or [("__none__", "g")])
    agn_i_raw, agn_labels = index_labels(agn_values or [("__none__", "0", "g")])
    host_i_raw, host_labels = index_labels(host_values or [("__none__", "g")])
    agn_lookup = {label: i for i, label in enumerate(agn_labels)}
    host_lookup = {label: i for i, label in enumerate(host_labels)}
    star_lookup = {label: i for i, label in enumerate(star_labels)}
    sf_lookup = {label: i for i, label in enumerate(sf_labels)}
    image_lookup = {label: i for i, label in enumerate(image_labels)}
    star_i = np.asarray(
        [star_lookup[(o.source_id, o.band)] if o.source_type == "standard" else -1 for o in observations],
        dtype=int,
    )
    agn_i = np.asarray(
        [
            agn_lookup[(o.source_id, "" if o.obs_mjd is None else f"{o.obs_mjd:.8f}", o.band)] if o.source_type == "agn" else -1
            for o in observations
        ],
        dtype=int,
    )
    host_i = np.asarray([host_lookup[(o.source_id, o.band)] if o.source_type == "agn" else -1 for o in observations], dtype=int)
    is_standard = np.asarray([o.source_type == "standard" for o in observations], dtype=bool)
    ref_mag = np.asarray([0.0 if o.ref_mag is None else o.ref_mag for o in observations], dtype=float)
    star_ref_mag = np.full(len(star_labels), 20.0, dtype=float)
    star_ref_mag_err = np.full(len(star_labels), 1.0, dtype=float)
    for obs in observations:
        if obs.source_type != "standard" or obs.ref_mag is None:
            continue
        label = (obs.source_id, obs.band)
        idx = star_lookup[label]
        star_ref_mag[idx] = obs.ref_mag
        star_ref_mag_err[idx] = obs.ref_mag_err
    image_fwhm_prior = np.full(len(image_labels), np.nan, dtype=float)
    for obs in observations:
        label = (obs.survey, obs.night, obs.image_id, obs.ccd_id, obs.band)
        image_fwhm_prior[image_lookup[label]] = obs.fwhm_prior
    image_fwhm_prior = np.where(np.isfinite(image_fwhm_prior), image_fwhm_prior, 3.0)
    rate_surface = np.asarray([o.rate_surface for o in observations], dtype=float)
    rate_err_surface = np.asarray([o.rate_err_surface for o in observations], dtype=float)
    fwhm_surface_grid = np.asarray([o.fwhm_grid for o in observations], dtype=float)
    centroid_surface_grid = np.asarray(observations[0].centroid_grid, dtype=float)
    n_fwhm_grid = fwhm_surface_grid.shape[1]
    n_centroid_grid = centroid_surface_grid.size
    rate_surface = rate_surface.reshape(len(observations), n_fwhm_grid, n_centroid_grid, n_centroid_grid)
    rate_err_surface = rate_err_surface.reshape(len(observations), n_fwhm_grid, n_centroid_grid, n_centroid_grid)
    valid_surface = np.isfinite(rate_surface) & np.isfinite(rate_err_surface) & (rate_err_surface > 0)
    fallback_err = np.nanmedian(rate_err_surface[valid_surface]) if np.any(valid_surface) else 1.0
    rate_surface = np.where(np.isfinite(rate_surface), rate_surface, 0.0)
    rate_err_surface = np.where(np.isfinite(rate_err_surface) & (rate_err_surface > 0), rate_err_surface, max(float(fallback_err), 1.0))
    data = {
        "rate": np.asarray([o.rate for o in observations], dtype=float),
        "rate_err": np.asarray([o.rate_err for o in observations], dtype=float),
        "host_rate": np.asarray([0.0 if o.host_rate is None else o.host_rate for o in observations], dtype=float),
        "host_rate_err": np.asarray([1.0 if o.host_rate_err is None else o.host_rate_err for o in observations], dtype=float),
        "point_host_rate_cov": np.asarray([o.point_host_rate_cov for o in observations], dtype=float),
        "rate_surface": rate_surface,
        "rate_err_surface": rate_err_surface,
        "fwhm_grid": fwhm_surface_grid,
        "centroid_grid": centroid_surface_grid,
        "image_fwhm_prior": image_fwhm_prior,
        "mag_inst_for_init": np.asarray([np.nan if o.mag_inst is None else o.mag_inst for o in observations], dtype=float),
        "ref_mag": ref_mag,
        "ref_mag_err": np.asarray([o.ref_mag_err for o in observations], dtype=float),
        "color": np.asarray([o.color for o in observations], dtype=float),
        "is_standard": is_standard,
        "is_agn": ~is_standard,
        "survey_filter_index": sf_i,
        "night_index": night_i,
        "image_index": image_i,
        "ccd_index": ccd_i,
        "source_survey_index": source_survey_i,
        "agn_index": agn_i,
        "host_index": host_i,
        "star_index": star_i,
        "star_ref_mag": star_ref_mag,
        "star_ref_mag_err": star_ref_mag_err,
        "night_parent": np.asarray([sf_lookup[(s, b)] for s, _n, b in night_labels], dtype=int),
        "image_parent": np.asarray([sf_lookup[(s, b)] for s, _n, _im, _c, b in image_labels], dtype=int),
        "ccd_parent": np.asarray([sf_lookup[(s, b)] for s, _c, b in ccd_labels], dtype=int),
        "n_survey_filter": len(sf_labels),
        "n_night": len(night_labels),
        "n_image": len(image_labels),
        "n_ccd": len(ccd_labels),
        "n_source_survey": len(source_survey_labels),
        "n_star": len(star_labels),
        "n_agn": len(agn_labels),
        "n_host": len(host_labels),
    }
    labels = {
        "survey_filter": sf_labels,
        "night": night_labels,
        "image": image_labels,
        "ccd": ccd_labels,
        "source_survey": source_survey_labels,
        "star": star_labels,
        "agn": agn_labels,
        "host": host_labels,
    }
    return data, labels


def center_by_parent(raw, parent, n_parent):
    _jax, jnp, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    sums = jnp.zeros(n_parent).at[parent].add(raw)
    counts = jnp.zeros(n_parent).at[parent].add(1.0)
    means = sums / jnp.maximum(counts, 1.0)
    return raw - means[parent]


def interp_surface_at(fwhm, dx, dy, fwhm_grid, centroid_grid, surface):
    """Trilinear interpolation over per-observation FWHM and common centroid axes."""

    _jax, jnp, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    n_centroid = centroid_grid.shape[0]
    if n_centroid == 1:
        return jnp.interp(fwhm, fwhm_grid, surface[:, 0, 0])
    x = jnp.clip(dx, centroid_grid[0], centroid_grid[-1])
    y = jnp.clip(dy, centroid_grid[0], centroid_grid[-1])
    ix0 = jnp.clip(jnp.searchsorted(centroid_grid, x, side="right") - 1, 0, n_centroid - 2)
    iy0 = jnp.clip(jnp.searchsorted(centroid_grid, y, side="right") - 1, 0, n_centroid - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    x0 = centroid_grid[ix0]
    x1 = centroid_grid[ix1]
    y0 = centroid_grid[iy0]
    y1 = centroid_grid[iy1]
    tx = (x - x0) / jnp.maximum(x1 - x0, 1.0e-12)
    ty = (y - y0) / jnp.maximum(y1 - y0, 1.0e-12)
    v00 = surface[:, iy0, ix0]
    v10 = surface[:, iy0, ix1]
    v01 = surface[:, iy1, ix0]
    v11 = surface[:, iy1, ix1]
    by_centroid = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) + ty * ((1.0 - tx) * v01 + tx * v11)
    return jnp.interp(fwhm, fwhm_grid, by_centroid)


def jax_vmap_interp_surface(fwhm, dx, dy, fwhm_grid, centroid_grid, surface):
    _jax, jnp, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    return _jax.vmap(lambda f, x, y, fg, s: interp_surface_at(f, x, y, fg, centroid_grid, s))(fwhm, dx, dy, fwhm_grid, surface)


def joint_model(data: Mapping[str, Any], config: Mapping[str, float]) -> None:
    jax, jnp, numpyro, dist, _MCMC, _NUTS = require_numpyro()
    sf = numpyro.sample("cal/survey_filter_zp", dist.Normal(0.0, config["prior_survey_filter_sigma"]).expand([data["n_survey_filter"]]))
    night_raw = numpyro.sample("cal/night_zp_raw", dist.Normal(0.0, config["prior_night_sigma"]).expand([data["n_night"]]))
    image_raw = numpyro.sample("cal/image_zp_raw", dist.Normal(0.0, config["prior_image_sigma"]).expand([data["n_image"]]))
    ccd_raw = numpyro.sample("cal/ccd_zp_raw", dist.Normal(0.0, config["prior_ccd_sigma"]).expand([data["n_ccd"]]))
    color_coeff = numpyro.sample("cal/color_coeff", dist.Normal(0.0, config["prior_color_sigma"]).expand([data["n_survey_filter"]]))
    intrinsic = numpyro.sample("cal/intrinsic_scatter", dist.HalfNormal(config["prior_intrinsic_scatter_sigma"]))
    star_mag = numpyro.sample(
        "star/true_mag",
        dist.Normal(data["star_ref_mag"], jnp.maximum(data["star_ref_mag_err"], config["prior_star_mag_floor"])),
    )
    agn_mag = numpyro.sample("agn/mag", dist.Normal(20.0, 5.0).expand([data["n_agn"]]))
    host_mag = numpyro.sample("agn/host_mag", dist.Normal(config["prior_host_mag_loc"], config["prior_host_mag_sigma"]).expand([data["n_host"]]))
    night = center_by_parent(night_raw, data["night_parent"], data["n_survey_filter"])
    image = center_by_parent(image_raw, data["image_parent"], data["n_survey_filter"])
    ccd = center_by_parent(ccd_raw, data["ccd_parent"], data["n_survey_filter"])
    offset = (
        sf[data["survey_filter_index"]]
        + night[data["night_index"]]
        + image[data["image_index"]]
        + ccd[data["ccd_index"]]
        + color_coeff[data["survey_filter_index"]] * data["color"]
    )
    agn_index = jnp.maximum(data["agn_index"], 0)
    star_index = jnp.maximum(data["star_index"], 0)
    true_mag = jnp.where(data["is_standard"], star_mag[star_index], agn_mag[agn_index])
    pred_mag = true_mag + offset
    pred_rate = 10.0 ** (-0.4 * pred_mag)
    model_mag_sigma = jnp.broadcast_to(intrinsic, pred_rate.shape)
    model_rate_sigma = jnp.log(10.0) / 2.5 * pred_rate * model_mag_sigma
    rate = data["rate"]
    rate_err = data["rate_err"]
    sigma = jnp.sqrt(rate_err**2 + model_rate_sigma**2)
    numpyro.deterministic("cal/mag_offset", offset)
    numpyro.deterministic("source/true_mag", true_mag)
    numpyro.deterministic("obs/pred_rate", pred_rate)
    standard_logp = dist.StudentT(config["student_t_df"], pred_rate, sigma).log_prob(rate)
    numpyro.factor("obs/standard_rate", jnp.sum(jnp.where(data["is_standard"], standard_logp, 0.0)))

    host_index = jnp.maximum(data["host_index"], 0)
    host_pred_rate = 10.0 ** (-0.4 * (host_mag[host_index] + offset))
    agn_mask = data["is_agn"]
    var_point = rate_err**2 + model_rate_sigma**2
    var_host = data["host_rate_err"] ** 2
    cov = data["point_host_rate_cov"]
    max_abs_cov = 0.95 * jnp.sqrt(jnp.maximum(var_point * var_host, 1.0e-300))
    cov = jnp.clip(cov, -max_abs_cov, max_abs_cov)
    det = jnp.maximum(var_point * var_host - cov**2, 1.0e-300)
    d_point = rate - pred_rate
    d_host = data["host_rate"] - host_pred_rate
    quad = (var_host * d_point**2 + var_point * d_host**2 - 2.0 * cov * d_point * d_host) / det
    agn_logp = -0.5 * (jnp.log((2.0 * jnp.pi) ** 2 * det) + quad)
    numpyro.factor("obs/agn_point_host_rate", jnp.sum(jnp.where(agn_mask, agn_logp, 0.0)))
    numpyro.deterministic("agn/host_pred_rate", host_pred_rate)


def as_jax_data(data: Mapping[str, Any]) -> dict[str, Any]:
    _jax, jnp, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    out = {}
    for key, value in data.items():
        if key.startswith("n_"):
            out[key] = int(value)
        elif key in {"is_standard", "is_agn"}:
            out[key] = jnp.asarray(value, dtype=bool)
        elif "index" in key or key.endswith("parent"):
            out[key] = jnp.asarray(value, dtype=jnp.int32)
        else:
            out[key] = jnp.asarray(value, dtype=jnp.float64)
    return out


def initial_values(
    data: Mapping[str, Any],
    *,
    fix_centroids: bool = False,
    initial_intrinsic_scatter: float = 0.2,
) -> dict[str, np.ndarray | float]:
    sf_init = np.zeros(int(data["n_survey_filter"]), dtype=float)
    mag_inst = np.asarray(data["mag_inst_for_init"], dtype=float)
    ref_mag = np.asarray(data["ref_mag"], dtype=float)
    is_standard = np.asarray(data["is_standard"], dtype=bool)
    sf_index = np.asarray(data["survey_filter_index"], dtype=int)
    agn_index = np.asarray(data["agn_index"], dtype=int)
    for sf_i in range(sf_init.size):
        mask = is_standard & (sf_index == sf_i) & np.isfinite(mag_inst)
        if np.any(mask):
            sf_init[sf_i] = float(np.nanmedian(mag_inst[mask] - ref_mag[mask]))

    agn_init = np.full(int(data["n_agn"]), 20.0, dtype=float)
    for agn_i in range(agn_init.size):
        mask = (agn_index == agn_i) & np.isfinite(mag_inst)
        if np.any(mask):
            agn_init[agn_i] = float(np.nanmedian(mag_inst[mask] - sf_init[sf_index[mask]]))
    host_index = np.asarray(data["host_index"], dtype=int)
    host_rate = np.asarray(data["host_rate"], dtype=float)
    host_init = np.full(int(data["n_host"]), 22.0, dtype=float)
    for host_i in range(host_init.size):
        mask = (host_index == host_i) & (host_rate > 0) & np.isfinite(host_rate)
        if np.any(mask):
            host_init[host_i] = float(np.nanmedian(-2.5 * np.log10(host_rate[mask]) - sf_init[sf_index[mask]]))

    values = {
        "cal/survey_filter_zp": sf_init,
        "cal/night_zp_raw": np.zeros(int(data["n_night"]), dtype=float),
        "cal/image_zp_raw": np.zeros(int(data["n_image"]), dtype=float),
        "cal/ccd_zp_raw": np.zeros(int(data["n_ccd"]), dtype=float),
        "cal/color_coeff": np.zeros(int(data["n_survey_filter"]), dtype=float),
        "cal/intrinsic_scatter": float(max(initial_intrinsic_scatter, 1.0e-4)),
        "star/true_mag": np.asarray(data["star_ref_mag"], dtype=float),
        "agn/mag": agn_init,
        "agn/host_mag": host_init,
    }
    return values


def run_inference(data: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    jax, _jnp, _numpyro, _dist, MCMC, NUTS = require_numpyro()
    from numpyro.infer.initialization import init_to_value

    config = {
        "student_t_df": args.student_t_df,
        "prior_survey_filter_sigma": args.prior_survey_filter_sigma,
        "prior_night_sigma": args.prior_night_sigma,
        "prior_image_sigma": args.prior_image_sigma,
        "prior_ccd_sigma": args.prior_ccd_sigma,
        "prior_color_sigma": args.prior_color_sigma,
        "prior_intrinsic_scatter_sigma": args.prior_intrinsic_scatter_sigma,
        "prior_host_mag_loc": args.prior_host_mag_loc,
        "prior_host_mag_sigma": args.prior_host_mag_sigma,
        "prior_log_fwhm_sigma": args.prior_log_fwhm_sigma,
        "prior_centroid_sigma_pix": args.prior_centroid_sigma_pix,
        "prior_star_mag_floor": args.prior_star_mag_floor,
    }
    init_strategy = init_to_value(
        values=initial_values(
            data,
            fix_centroids=args.fix_centroids,
            initial_intrinsic_scatter=args.initial_intrinsic_scatter,
        )
    )
    print(
        "starting NumPyro NUTS: "
        f"observations={len(data['rate'])}, images={data['n_image']}, "
        f"warmup={args.num_warmup}, samples={args.num_samples}",
        flush=True,
    )
    mcmc = MCMC(NUTS(joint_model, init_strategy=init_strategy), num_warmup=args.num_warmup, num_samples=args.num_samples, progress_bar=not args.hide_progress)
    mcmc.run(jax.random.PRNGKey(args.rng_seed), as_jax_data(data), config)
    samples = mcmc.get_samples()
    print("finished NumPyro NUTS", flush=True)
    return {key: np.asarray(value) for key, value in samples.items()}


def centered_numpy(raw: np.ndarray, parent: np.ndarray, n_parent: int) -> np.ndarray:
    sums = np.zeros(n_parent)
    counts = np.zeros(n_parent)
    for value, group in zip(raw, parent):
        sums[group] += value
        counts[group] += 1.0
    means = sums / np.maximum(counts, 1.0)
    return raw - means[parent]


def interp_surface_at_numpy(fwhm: float, dx: float, dy: float, fwhm_grid: np.ndarray, centroid_grid: np.ndarray, surface: np.ndarray) -> float:
    x = float(np.clip(dx, centroid_grid[0], centroid_grid[-1]))
    y = float(np.clip(dy, centroid_grid[0], centroid_grid[-1]))
    ix0 = int(np.clip(np.searchsorted(centroid_grid, x, side="right") - 1, 0, centroid_grid.size - 2))
    iy0 = int(np.clip(np.searchsorted(centroid_grid, y, side="right") - 1, 0, centroid_grid.size - 2))
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    tx = (x - centroid_grid[ix0]) / max(centroid_grid[ix1] - centroid_grid[ix0], 1.0e-12)
    ty = (y - centroid_grid[iy0]) / max(centroid_grid[iy1] - centroid_grid[iy0], 1.0e-12)
    v00 = surface[:, iy0, ix0]
    v10 = surface[:, iy0, ix1]
    v01 = surface[:, iy1, ix0]
    v11 = surface[:, iy1, ix1]
    by_centroid = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) + ty * ((1.0 - tx) * v01 + tx * v11)
    return float(np.interp(fwhm, fwhm_grid, by_centroid))


def posterior_products(observations: list[Observation], data: Mapping[str, Any], labels: Mapping[str, list[tuple[str, ...]]], samples: Mapping[str, np.ndarray]):
    means = {key: np.mean(value, axis=0) for key, value in samples.items()}
    stds = {key: np.std(value, axis=0) for key, value in samples.items()}
    sf = means["cal/survey_filter_zp"]
    night = centered_numpy(means["cal/night_zp_raw"], data["night_parent"], data["n_survey_filter"])
    image = centered_numpy(means["cal/image_zp_raw"], data["image_parent"], data["n_survey_filter"])
    ccd = centered_numpy(means["cal/ccd_zp_raw"], data["ccd_parent"], data["n_survey_filter"])
    color = means["cal/color_coeff"]
    offsets = sf[data["survey_filter_index"]] + night[data["night_index"]] + image[data["image_index"]] + ccd[data["ccd_index"]] + color[data["survey_filter_index"]] * data["color"]
    agn_mag = means["agn/mag"]
    agn_mag_err = stds["agn/mag"]
    host_mag = means.get("agn/host_mag", np.full(data["n_host"], np.nan))
    host_mag_err = stds.get("agn/host_mag", np.full(data["n_host"], np.nan))
    star_mag = means["star/true_mag"]
    star_mag_err = stds["star/true_mag"]
    star_true = np.asarray([star_mag[i] if i >= 0 else np.nan for i in data["star_index"]])
    star_true_err = np.asarray([star_mag_err[i] if i >= 0 else np.nan for i in data["star_index"]])
    agn_true = np.asarray([agn_mag[i] if i >= 0 else np.nan for i in data["agn_index"]])
    agn_true_err = np.asarray([agn_mag_err[i] if i >= 0 else np.nan for i in data["agn_index"]])
    host_true = np.asarray([host_mag[i] if i >= 0 else np.nan for i in data["host_index"]])
    host_true_err = np.asarray([host_mag_err[i] if i >= 0 else np.nan for i in data["host_index"]])
    fitted_fwhm = np.asarray(data["image_fwhm_prior"], dtype=float)
    obs_dx = np.asarray([parse_float(str(o.row.get("fit_centroid_shift_x_pix", ""))) or 0.0 for o in observations], dtype=float)
    obs_dy = np.asarray([parse_float(str(o.row.get("fit_centroid_shift_y_pix", ""))) or 0.0 for o in observations], dtype=float)
    fitted_rate = np.asarray(data["rate"], dtype=float)
    fitted_rate_err = np.asarray(data["rate_err"], dtype=float)
    return means, offsets, star_true, star_true_err, agn_true, agn_true_err, host_true, host_true_err, fitted_fwhm, obs_dx, obs_dy, fitted_rate, fitted_rate_err


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(args: argparse.Namespace, observations: list[Observation], data: Mapping[str, Any], labels: Mapping[str, list[tuple[str, ...]]], samples: Mapping[str, np.ndarray]) -> None:
    (
        means,
        offsets,
        star_true,
        star_true_err,
        agn_true,
        agn_true_err,
        host_true,
        host_true_err,
        fitted_fwhm,
        fitted_dx,
        fitted_dy,
        fitted_rate,
        fitted_rate_err,
    ) = posterior_products(observations, data, labels, samples)
    rows = [dict(o.row) for o in observations]
    compressed_fields = list(rows[0].keys())
    if args.out_compressed:
        write_csv(args.out_compressed, rows, compressed_fields)

    star_rows: list[dict[str, Any]] = []
    agn_rows: list[dict[str, Any]] = []
    for i, (obs, row) in enumerate(zip(observations, rows)):
        offset = float(offsets[i])
        row["calibration_mag_offset"] = offset
        row["calibration_scale"] = 10 ** (-0.4 * offset)
        row["fit_centroid_shift_x_pix"] = float(fitted_dx[i])
        row["fit_centroid_shift_y_pix"] = float(fitted_dy[i])
        row["fit_center_x_pix"] = float(row["wcs_center_x_pix"]) + float(fitted_dx[i])
        row["fit_center_y_pix"] = float(row["wcs_center_y_pix"]) + float(fitted_dy[i])
        row["fit_centroid_scope"] = row.get("fit_centroid_scope", "local_marginalized_grid")
        row["prior_rate"] = row.get("rate", "")
        row["prior_rate_err"] = row.get("rate_err", "")
        row["prior_mag_inst"] = row.get("mag_inst", "")
        row["prior_mag_err"] = row.get("mag_err", "")
        row["fitted_rate"] = float(fitted_rate[i])
        row["fitted_rate_err"] = float(fitted_rate_err[i])
        fitted_mag_inst = -2.5 * math.log10(fitted_rate[i]) if fitted_rate[i] > 0 else None
        fitted_mag_err = 2.5 / math.log(10) * fitted_rate_err[i] / fitted_rate[i] if fitted_rate[i] > 0 and fitted_rate_err[i] > 0 else None
        row["fitted_mag_inst"] = "" if fitted_mag_inst is None else fitted_mag_inst
        row["fitted_mag_err"] = "" if fitted_mag_err is None else fitted_mag_err
        row["compressed_mag_err"] = row.get("mag_err", "")
        if obs.source_type == "standard":
            if fitted_mag_inst is None:
                row["mag_cal"] = ""
                row["calibration_residual"] = ""
                row["calibration_status"] = "nonpositive_flux"
            else:
                row["mag_inst"] = fitted_mag_inst
                row["mag_err"] = "" if fitted_mag_err is None else fitted_mag_err
                mag_cal = fitted_mag_inst - offset
                row["mag_cal"] = mag_cal
                row["star_true_mag"] = float(star_true[i])
                row["star_true_mag_err"] = float(star_true_err[i])
                row["catalog_residual"] = float(star_true[i] - float(obs.ref_mag))
                row["calibration_residual"] = mag_cal - float(star_true[i])
                row["calibration_status"] = "ok"
            star_rows.append(row)
        else:
            row["mag_cal"] = float(agn_true[i])
            row["mag_err"] = float(agn_true_err[i])
            row["mag_model_err"] = float(agn_true_err[i])
            row["host_mag_cal"] = float(host_true[i])
            row["host_mag_err"] = float(host_true_err[i])
            row["host_flux_calibration_scale"] = row["calibration_scale"]
            row["calibration_residual"] = ""
            row["calibration_status"] = "ok"
            agn_rows.append(row)

    extra = [
        "fitted_moffat_fwhm_pix",
        "prior_rate",
        "prior_rate_err",
        "prior_mag_inst",
        "prior_mag_err",
        "fitted_rate",
        "fitted_rate_err",
        "fitted_mag_inst",
        "fitted_mag_err",
        "compressed_mag_err",
        "calibration_mag_offset",
        "calibration_scale",
        "mag_cal",
        "mag_model_err",
        "host_mag_cal",
        "host_mag_err",
        "host_flux_calibration_scale",
        "star_true_mag",
        "star_true_mag_err",
        "catalog_residual",
        "calibration_residual",
        "calibration_status",
    ]
    write_csv(args.out_stars, star_rows, compressed_fields + [x for x in extra if x not in compressed_fields])
    write_csv(args.out_agn, agn_rows, compressed_fields + [x for x in extra if x not in compressed_fields])

    image_rows = []
    sf_lookup = {label: i for i, label in enumerate(labels["survey_filter"])}
    night_lookup = {label: i for i, label in enumerate(labels["night"])}
    ccd_lookup = {label: i for i, label in enumerate(labels["ccd"])}
    night = centered_numpy(np.mean(samples["cal/night_zp_raw"], axis=0), data["night_parent"], data["n_survey_filter"])
    image = centered_numpy(np.mean(samples["cal/image_zp_raw"], axis=0), data["image_parent"], data["n_survey_filter"])
    ccd = centered_numpy(np.mean(samples["cal/ccd_zp_raw"], axis=0), data["ccd_parent"], data["n_survey_filter"])
    sf = np.mean(samples["cal/survey_filter_zp"], axis=0)
    color = np.mean(samples["cal/color_coeff"], axis=0)
    intrinsic = float(np.mean(samples["cal/intrinsic_scatter"]))
    for image_i, label in enumerate(labels["image"]):
        survey, night_label, image_id, ccd_id, band = label
        sf_i = sf_lookup[(survey, band)]
        n_i = night_lookup[(survey, night_label, band)]
        c_i = ccd_lookup[(survey, ccd_id, band)]
        total = sf[sf_i] + night[n_i] + image[image_i] + ccd[c_i]
        image_rows.append(
            {
                "survey": survey,
                "night": night_label,
                "image_id": image_id,
                "ccd_id": ccd_id,
                "filter": band,
                "survey_filter_zp": sf[sf_i],
                "night_zp": night[n_i],
                "image_zp": image[image_i],
                "ccd_zp": ccd[c_i],
                "color_coeff": color[sf_i],
                "total_zp_at_center": total,
                "intrinsic_scatter": intrinsic,
                "moffat_fwhm_pix": fitted_fwhm[image_i],
                "moffat_fwhm_prior_pix": data["image_fwhm_prior"][image_i],
            }
        )
    write_csv(
        args.out_zeropoints,
        image_rows,
        ["survey", "night", "image_id", "ccd_id", "filter", "survey_filter_zp", "night_zp", "image_zp", "ccd_zp", "color_coeff", "total_zp_at_center", "intrinsic_scatter", "moffat_fwhm_pix", "moffat_fwhm_prior_pix"],
    )
    if args.out_params:
        payload = {key.replace("/", "__"): value for key, value in samples.items()}
        for name, label_values in labels.items():
            payload[f"labels__{name}"] = np.asarray(["|".join(label) for label in label_values])
        args.out_params.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_params, **payload)


def main() -> int:
    args = parse_args()
    try:
        standards = read_sources(args.standards)
        targets = read_sources(args.target_catalog)
        observations = compress_observations(standards, targets, args.image_manifest, args)
        n_standard = sum(o.source_type == "standard" for o in observations)
        n_agn = sum(o.source_type == "agn" for o in observations)
        print(f"compressed observations: standards={n_standard}, agn={n_agn}", flush=True)
        data, labels = build_arrays(observations)
        samples = run_inference(data, args)
        write_outputs(args, observations, data, labels, samples)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote zeropoints: {args.out_zeropoints}")
    print(f"wrote calibrated standards: {args.out_stars}")
    print(f"wrote calibrated AGN: {args.out_agn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
