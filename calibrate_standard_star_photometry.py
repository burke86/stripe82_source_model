#!/usr/bin/env python3
"""Bayesian Stripe 82 standard-star calibration for PSF source modeling.

This script infers survey/filter, night, image/CCD, CCD, color, and optional
linear spatial calibration terms from standard-star PSF photometry. It is
designed to expose stable NumPyro sample-site names that can later be called
from JAGUAR's source-modeling likelihood.

The model is:

    m_inst - m_ref =
        survey_filter_zp[survey, filter]
      + night_zp[survey, night, filter]
      + image_zp[image_id, ccd_id, filter]
      + ccd_zp[survey, ccd_id, filter]
      + color_coeff[survey, filter] * color
      + optional spatial terms
      + residual

Calibration is modeled in magnitudes. The multiplicative flux/count scale for
JAGUAR is:

    calibration_scale = 10 ** (-0.4 * calibration_mag_offset)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FILTER_MAP_DEFAULT = "u:u,g:g,r:r,i:i,z:z,ps1_g:g,ps1_r:r,ps1_i:i,ps1_z:z,ps1_y:z,zg:g,zr:r,zi:i"


@dataclass
class ReferenceStar:
    star_id: str
    values: dict[str, float]


@dataclass
class Measurement:
    row: dict[str, str]
    star_id: str
    survey: str
    night: str
    image_id: str
    ccd_id: str
    filter_name: str
    mag_inst: float
    mag_err: float
    ref_mag: float
    ref_mag_err: float
    color: float
    x_norm: float
    y_norm: float


@dataclass(frozen=True)
class BayesianCalibrationConfig:
    use_spatial: bool
    student_t_df: float
    prior_survey_filter_sigma: float
    prior_night_sigma: float
    prior_image_sigma: float
    prior_ccd_sigma: float
    prior_color_sigma: float
    prior_spatial_sigma: float
    prior_intrinsic_scatter_sigma: float


@dataclass
class CalibrationArrays:
    measurements: list[Measurement]
    data: dict[str, Any]
    labels: dict[str, list[tuple[str, ...]]]
    parent_indices: dict[str, list[int]]


@dataclass
class PosteriorSummary:
    arrays: CalibrationArrays
    posterior_mean: dict[str, Any]
    posterior_std: dict[str, Any]
    offsets: list[float]
    offset_std: list[float]
    scales: list[float]
    scale_std: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer hierarchical Bayesian calibration terms from Stripe 82 standard-star PSF photometry."
    )
    parser.add_argument("--reference", required=True, type=Path, help="Reference-star CSV.")
    parser.add_argument("--photometry", required=True, type=Path, help="PSF-photometry CSV.")
    parser.add_argument("--out-calibrated", required=True, type=Path, help="Output calibrated photometry CSV.")
    parser.add_argument("--out-zeropoints", required=True, type=Path, help="Output posterior coefficient summary CSV.")
    parser.add_argument("--out-params", type=Path, help="Optional .npz posterior/label export for JAGUAR integration.")

    parser.add_argument("--star-id-column", default="star_id")
    parser.add_argument("--survey-column", default="survey")
    parser.add_argument("--night-column", default="night")
    parser.add_argument("--image-id-column", default="image_id")
    parser.add_argument("--ccd-id-column", default="ccd_id")
    parser.add_argument("--filter-column", default="filter")

    parser.add_argument("--ref-filter-map", default=FILTER_MAP_DEFAULT)
    parser.add_argument("--ref-mag-prefix", default="", help="Optional prefix for reference magnitude columns.")
    parser.add_argument("--color", default="g-r", help="Reference color as MAG1-MAG2. Default: g-r.")
    parser.add_argument("--mag-column", default="mag_inst")
    parser.add_argument("--mag-err-column", default="mag_err")
    parser.add_argument("--flux-column", default="flux")
    parser.add_argument("--flux-err-column", default="flux_err")
    parser.add_argument("--exptime-column", default="exptime")
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--spatial-order", type=int, choices=[0, 1], default=0)
    parser.add_argument("--default-mag-err", type=float, default=0.05)
    parser.add_argument("--max-mag-err", type=float, default=0.2)
    parser.add_argument("--max-ref-mag-err", type=float)

    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--student-t-df", type=float, default=5.0)
    parser.add_argument("--hide-progress", action="store_true")

    parser.add_argument("--prior-survey-filter-sigma", type=float, default=0.2)
    parser.add_argument("--prior-night-sigma", type=float, default=0.05)
    parser.add_argument("--prior-image-sigma", type=float, default=0.05)
    parser.add_argument("--prior-ccd-sigma", type=float, default=0.03)
    parser.add_argument("--prior-color-sigma", type=float, default=0.1)
    parser.add_argument("--prior-spatial-sigma", type=float, default=0.03)
    parser.add_argument("--prior-intrinsic-scatter-sigma", type=float, default=0.03)
    return parser.parse_args()


def require_numpyro():
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hierarchical Bayesian calibration requires jax and numpyro. "
            "Run this in the JAGUAR environment or install: pip install 'jax>=0.4.30' 'jaxlib>=0.4.30' 'numpyro>=0.15'"
        ) from exc
    return jax, jnp, np, numpyro, dist, MCMC, NUTS


def parse_mapping(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, mapped = item.split(":", 1)
        mapping[key.strip()] = mapped.strip()
    return mapping


def parse_color(value: str) -> tuple[str, str]:
    if "-" not in value:
        raise ValueError("--color must look like MAG1-MAG2, e.g. g-r")
    left, right = value.split("-", 1)
    return left.strip(), right.strip()


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() in {"nan", "null", "none"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def prefixed(prefix: str, column: str) -> str:
    return column if column.startswith(prefix) else f"{prefix}{column}"


def read_reference(path: Path, *, star_id_column: str) -> dict[str, ReferenceStar]:
    stars: dict[str, ReferenceStar] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or star_id_column not in reader.fieldnames:
            raise RuntimeError(f"{path} must contain a {star_id_column!r} column")
        for row in reader:
            star_id = (row.get(star_id_column) or "").strip()
            if not star_id:
                continue
            values: dict[str, float] = {}
            for key, value in row.items():
                parsed = parse_float(value)
                if parsed is not None:
                    values[key] = parsed
            stars[star_id] = ReferenceStar(star_id=star_id, values=values)
    return stars


def instrumental_mag(row: dict[str, str], args: argparse.Namespace) -> tuple[float | None, float | None]:
    mag = parse_float(row.get(args.mag_column))
    mag_err = parse_float(row.get(args.mag_err_column))
    if mag is not None:
        return mag, mag_err

    flux = parse_float(row.get(args.flux_column))
    if flux is None or flux <= 0:
        return None, None
    exptime = parse_float(row.get(args.exptime_column)) or 1.0
    if exptime <= 0:
        return None, None
    mag = -2.5 * math.log10(flux / exptime)

    flux_err = parse_float(row.get(args.flux_err_column))
    if mag_err is None and flux_err is not None and flux_err > 0:
        mag_err = 2.5 / math.log(10) * flux_err / flux
    return mag, mag_err


def normalize_xy(measurements: list[Measurement]) -> None:
    by_image: dict[tuple[str, str, str, str], list[Measurement]] = {}
    for measurement in measurements:
        key = (measurement.survey, measurement.image_id, measurement.ccd_id, measurement.filter_name)
        by_image.setdefault(key, []).append(measurement)
    for group in by_image.values():
        xs = [measurement.x_norm for measurement in group]
        ys = [measurement.y_norm for measurement in group]
        x_mid = 0.5 * (min(xs) + max(xs))
        y_mid = 0.5 * (min(ys) + max(ys))
        x_scale = max(max(xs) - min(xs), 1.0)
        y_scale = max(max(ys) - min(ys), 1.0)
        for measurement in group:
            measurement.x_norm = (measurement.x_norm - x_mid) / x_scale
            measurement.y_norm = (measurement.y_norm - y_mid) / y_scale


def read_measurements(reference: dict[str, ReferenceStar], args: argparse.Namespace) -> tuple[list[str], list[Measurement]]:
    filter_map = parse_mapping(args.ref_filter_map)
    color_left, color_right = parse_color(args.color)
    required = [
        args.star_id_column,
        args.survey_column,
        args.night_column,
        args.image_id_column,
        args.ccd_id_column,
        args.filter_column,
    ]
    if args.spatial_order == 1:
        required.extend([args.x_column, args.y_column])

    measurements: list[Measurement] = []
    with args.photometry.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{args.photometry} has no CSV header")
        for column in dict.fromkeys(required):
            if column not in reader.fieldnames:
                raise RuntimeError(f"{args.photometry} must contain {column!r}")
        input_fieldnames = list(reader.fieldnames)

        for row in reader:
            star_id = (row.get(args.star_id_column) or "").strip()
            ref_star = reference.get(star_id)
            if not ref_star:
                continue

            filter_name = (row.get(args.filter_column) or "").strip()
            ref_column = prefixed(args.ref_mag_prefix, filter_map.get(filter_name, filter_name))
            ref_mag = ref_star.values.get(ref_column)
            if ref_mag is None:
                continue
            ref_mag_err = ref_star.values.get(f"{ref_column}_err", 0.0)
            if args.max_ref_mag_err is not None and ref_mag_err > args.max_ref_mag_err:
                continue

            mag_inst, mag_err = instrumental_mag(row, args)
            if mag_inst is None:
                continue
            mag_err = args.default_mag_err if mag_err is None else mag_err
            if mag_err <= 0 or mag_err > args.max_mag_err:
                continue

            left = ref_star.values.get(prefixed(args.ref_mag_prefix, color_left))
            right = ref_star.values.get(prefixed(args.ref_mag_prefix, color_right))
            if left is None or right is None:
                continue

            x = parse_float(row.get(args.x_column)) if args.spatial_order == 1 else 0.0
            y = parse_float(row.get(args.y_column)) if args.spatial_order == 1 else 0.0
            if x is None or y is None:
                continue

            values = {
                "survey": (row.get(args.survey_column) or "").strip(),
                "night": (row.get(args.night_column) or "").strip(),
                "image_id": (row.get(args.image_id_column) or "").strip(),
                "ccd_id": (row.get(args.ccd_id_column) or "").strip(),
            }
            if any(not value for value in values.values()):
                continue

            measurements.append(
                Measurement(
                    row=row,
                    star_id=star_id,
                    survey=values["survey"],
                    night=values["night"],
                    image_id=values["image_id"],
                    ccd_id=values["ccd_id"],
                    filter_name=filter_name,
                    mag_inst=mag_inst,
                    mag_err=mag_err,
                    ref_mag=ref_mag,
                    ref_mag_err=ref_mag_err,
                    color=left - right,
                    x_norm=x,
                    y_norm=y,
                )
            )

    if args.spatial_order == 1:
        normalize_xy(measurements)
    return input_fieldnames, measurements


def index_labels(values: list[tuple[str, ...]]) -> tuple[list[int], list[tuple[str, ...]]]:
    labels: list[tuple[str, ...]] = []
    index: dict[tuple[str, ...], int] = {}
    indices: list[int] = []
    for value in values:
        if value not in index:
            index[value] = len(labels)
            labels.append(value)
        indices.append(index[value])
    return indices, labels


def build_calibration_arrays(measurements: list[Measurement], *, use_spatial: bool) -> CalibrationArrays:
    if not measurements:
        raise RuntimeError("No matched standard-star measurements found")

    survey_filter_values = [(m.survey, m.filter_name) for m in measurements]
    night_values = [(m.survey, m.night, m.filter_name) for m in measurements]
    image_values = [(m.survey, m.night, m.image_id, m.ccd_id, m.filter_name) for m in measurements]
    ccd_values = [(m.survey, m.ccd_id, m.filter_name) for m in measurements]

    survey_filter_index, survey_filter_labels = index_labels(survey_filter_values)
    night_index, night_labels = index_labels(night_values)
    image_index, image_labels = index_labels(image_values)
    ccd_index, ccd_labels = index_labels(ccd_values)
    survey_filter_lookup = {label: i for i, label in enumerate(survey_filter_labels)}

    night_parent = [survey_filter_lookup[(survey, filter_name)] for survey, _night, filter_name in night_labels]
    image_parent = [survey_filter_lookup[(survey, filter_name)] for survey, _night, _image, _ccd, filter_name in image_labels]
    ccd_parent = [survey_filter_lookup[(survey, filter_name)] for survey, _ccd, filter_name in ccd_labels]

    data = {
        "mag_diff": [m.mag_inst - m.ref_mag for m in measurements],
        "mag_err": [m.mag_err for m in measurements],
        "ref_mag_err": [m.ref_mag_err for m in measurements],
        "color": [m.color for m in measurements],
        "x": [m.x_norm for m in measurements],
        "y": [m.y_norm for m in measurements],
        "survey_filter_index": survey_filter_index,
        "night_index": night_index,
        "image_index": image_index,
        "ccd_index": ccd_index,
        "night_parent_survey_filter": night_parent,
        "image_parent_survey_filter": image_parent,
        "ccd_parent_survey_filter": ccd_parent,
        "n_survey_filter": len(survey_filter_labels),
        "n_night": len(night_labels),
        "n_image": len(image_labels),
        "n_ccd": len(ccd_labels),
        "use_spatial": use_spatial,
    }
    labels = {
        "survey_filter": survey_filter_labels,
        "night": night_labels,
        "image": image_labels,
        "ccd": ccd_labels,
    }
    parent_indices = {
        "night_parent_survey_filter": night_parent,
        "image_parent_survey_filter": image_parent,
        "ccd_parent_survey_filter": ccd_parent,
    }
    return CalibrationArrays(measurements=measurements, data=data, labels=labels, parent_indices=parent_indices)


def _as_jax_data(data: Mapping[str, Any]):
    _jax, jnp, _np, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    out = dict(data)
    for key in ("mag_diff", "mag_err", "ref_mag_err", "color", "x", "y"):
        out[key] = jnp.asarray(out[key], dtype=jnp.float64)
    for key in (
        "survey_filter_index",
        "night_index",
        "image_index",
        "ccd_index",
        "night_parent_survey_filter",
        "image_parent_survey_filter",
        "ccd_parent_survey_filter",
    ):
        out[key] = jnp.asarray(out[key], dtype=jnp.int32)
    return out


def _center_by_parent(raw, parent_index, n_parent):
    _jax, jnp, _np, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    sums = jnp.zeros(n_parent, dtype=raw.dtype).at[parent_index].add(raw)
    counts = jnp.zeros(n_parent, dtype=raw.dtype).at[parent_index].add(1.0)
    means = sums / jnp.maximum(counts, 1.0)
    return raw - means[parent_index]


def _config_value(config: BayesianCalibrationConfig | Mapping[str, Any], name: str):
    if isinstance(config, Mapping):
        return config[name]
    return getattr(config, name)


def predict_calibration_mag_offset(data: Mapping[str, Any], params: Mapping[str, Any], config: BayesianCalibrationConfig | Mapping[str, Any]):
    """Return the calibration magnitude offset for each observation."""

    _jax, jnp, _np, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    survey_filter_zp = params["cal/survey_filter_zp"]
    night_zp = _center_by_parent(
        params["cal/night_zp_raw"],
        data["night_parent_survey_filter"],
        data["n_survey_filter"],
    )
    image_zp = _center_by_parent(
        params["cal/image_zp_raw"],
        data["image_parent_survey_filter"],
        data["n_survey_filter"],
    )
    ccd_zp = _center_by_parent(
        params["cal/ccd_zp_raw"],
        data["ccd_parent_survey_filter"],
        data["n_survey_filter"],
    )
    color_coeff = params["cal/color_coeff"]
    offset = (
        survey_filter_zp[data["survey_filter_index"]]
        + night_zp[data["night_index"]]
        + image_zp[data["image_index"]]
        + ccd_zp[data["ccd_index"]]
        + color_coeff[data["survey_filter_index"]] * data["color"]
    )
    if _config_value(config, "use_spatial"):
        offset = (
            offset
            + params["cal/spatial_x_coeff"][data["survey_filter_index"]] * data["x"]
            + params["cal/spatial_y_coeff"][data["survey_filter_index"]] * data["y"]
        )
    return jnp.asarray(offset, dtype=jnp.float64)


def calibration_scale_from_mag_offset(mag_offset):
    """Convert a magnitude calibration offset to a multiplicative flux scale."""

    _jax, jnp, _np, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    return 10.0 ** (-0.4 * mag_offset)


def standard_star_calibration_model(data: Mapping[str, Any], config: BayesianCalibrationConfig | Mapping[str, Any]) -> None:
    """NumPyro model for hierarchical standard-star calibration."""

    _jax, jnp, _np, numpyro, dist, _MCMC, _NUTS = require_numpyro()
    survey_filter_zp = numpyro.sample(
        "cal/survey_filter_zp",
        dist.Normal(0.0, _config_value(config, "prior_survey_filter_sigma")).expand([data["n_survey_filter"]]),
    )
    night_zp_raw = numpyro.sample(
        "cal/night_zp_raw",
        dist.Normal(0.0, _config_value(config, "prior_night_sigma")).expand([data["n_night"]]),
    )
    image_zp_raw = numpyro.sample(
        "cal/image_zp_raw",
        dist.Normal(0.0, _config_value(config, "prior_image_sigma")).expand([data["n_image"]]),
    )
    ccd_zp_raw = numpyro.sample(
        "cal/ccd_zp_raw",
        dist.Normal(0.0, _config_value(config, "prior_ccd_sigma")).expand([data["n_ccd"]]),
    )
    color_coeff = numpyro.sample(
        "cal/color_coeff",
        dist.Normal(0.0, _config_value(config, "prior_color_sigma")).expand([data["n_survey_filter"]]),
    )
    params = {
        "cal/survey_filter_zp": survey_filter_zp,
        "cal/night_zp_raw": night_zp_raw,
        "cal/image_zp_raw": image_zp_raw,
        "cal/ccd_zp_raw": ccd_zp_raw,
        "cal/color_coeff": color_coeff,
    }
    if _config_value(config, "use_spatial"):
        params["cal/spatial_x_coeff"] = numpyro.sample(
            "cal/spatial_x_coeff",
            dist.Normal(0.0, _config_value(config, "prior_spatial_sigma")).expand([data["n_survey_filter"]]),
        )
        params["cal/spatial_y_coeff"] = numpyro.sample(
            "cal/spatial_y_coeff",
            dist.Normal(0.0, _config_value(config, "prior_spatial_sigma")).expand([data["n_survey_filter"]]),
        )
    else:
        params["cal/spatial_x_coeff"] = jnp.zeros(data["n_survey_filter"])
        params["cal/spatial_y_coeff"] = jnp.zeros(data["n_survey_filter"])

    intrinsic_scatter = numpyro.sample(
        "cal/intrinsic_scatter",
        dist.HalfNormal(_config_value(config, "prior_intrinsic_scatter_sigma")),
    )
    offset = predict_calibration_mag_offset(data, params, config)
    sigma = jnp.sqrt(data["mag_err"] ** 2 + data["ref_mag_err"] ** 2 + intrinsic_scatter**2)
    numpyro.deterministic("cal/mag_offset", offset)
    numpyro.deterministic("cal/scale", calibration_scale_from_mag_offset(offset))
    numpyro.sample(
        "cal/standard_star_residual",
        dist.StudentT(_config_value(config, "student_t_df"), offset, sigma),
        obs=data["mag_diff"],
    )


def summarize_posterior(samples: Mapping[str, Any], arrays: CalibrationArrays, config: BayesianCalibrationConfig) -> PosteriorSummary:
    _jax, jnp, np, _numpyro, _dist, _MCMC, _NUTS = require_numpyro()
    data = _as_jax_data(arrays.data)
    params_per_sample = {
        key: value
        for key, value in samples.items()
        if key.startswith("cal/") and key not in {"cal/mag_offset", "cal/scale", "cal/standard_star_residual"}
    }
    if "cal/spatial_x_coeff" not in params_per_sample:
        params_per_sample["cal/spatial_x_coeff"] = jnp.zeros((len(next(iter(params_per_sample.values()))), data["n_survey_filter"]))
    if "cal/spatial_y_coeff" not in params_per_sample:
        params_per_sample["cal/spatial_y_coeff"] = jnp.zeros((len(next(iter(params_per_sample.values()))), data["n_survey_filter"]))

    n_samples = len(next(iter(params_per_sample.values())))
    offsets = []
    scales = []
    for i in range(n_samples):
        params_i = {key: value[i] for key, value in params_per_sample.items()}
        offsets.append(predict_calibration_mag_offset(data, params_i, config))
        scales.append(calibration_scale_from_mag_offset(offsets[-1]))
    offset_stack = jnp.stack(offsets)
    scale_stack = jnp.stack(scales)
    posterior_mean = {key: np.asarray(jnp.mean(value, axis=0)) for key, value in params_per_sample.items()}
    posterior_std = {key: np.asarray(jnp.std(value, axis=0)) for key, value in params_per_sample.items()}
    posterior_mean["cal/intrinsic_scatter"] = np.asarray(jnp.mean(samples["cal/intrinsic_scatter"], axis=0))
    posterior_std["cal/intrinsic_scatter"] = np.asarray(jnp.std(samples["cal/intrinsic_scatter"], axis=0))
    return PosteriorSummary(
        arrays=arrays,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        offsets=list(map(float, np.asarray(jnp.mean(offset_stack, axis=0)))),
        offset_std=list(map(float, np.asarray(jnp.std(offset_stack, axis=0)))),
        scales=list(map(float, np.asarray(jnp.mean(scale_stack, axis=0)))),
        scale_std=list(map(float, np.asarray(jnp.std(scale_stack, axis=0)))),
    )


def run_bayesian_calibration(arrays: CalibrationArrays, config: BayesianCalibrationConfig, args: argparse.Namespace) -> PosteriorSummary:
    jax, _jnp, _np, _numpyro, _dist, MCMC, NUTS = require_numpyro()
    data = _as_jax_data(arrays.data)
    kernel = NUTS(standard_star_calibration_model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=not args.hide_progress,
    )
    mcmc.run(jax.random.PRNGKey(args.rng_seed), data, config)
    return summarize_posterior(mcmc.get_samples(group_by_chain=False), arrays, config)


def _center_numpy(raw, parent_index, n_parent):
    import numpy as np

    raw = np.asarray(raw, dtype=float)
    parent = np.asarray(parent_index, dtype=int)
    sums = np.zeros(n_parent, dtype=float)
    counts = np.zeros(n_parent, dtype=float)
    for value, group in zip(raw, parent):
        sums[group] += value
        counts[group] += 1.0
    means = sums / np.maximum(counts, 1.0)
    return raw - means[parent]


def write_zeropoints(path: Path, summary: PosteriorSummary) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = summary.arrays.labels
    parents = summary.arrays.parent_indices
    mean = summary.posterior_mean
    std = summary.posterior_std
    night_zp = _center_numpy(mean["cal/night_zp_raw"], parents["night_parent_survey_filter"], len(labels["survey_filter"]))
    image_zp = _center_numpy(mean["cal/image_zp_raw"], parents["image_parent_survey_filter"], len(labels["survey_filter"]))
    ccd_zp = _center_numpy(mean["cal/ccd_zp_raw"], parents["ccd_parent_survey_filter"], len(labels["survey_filter"]))
    sf_lookup = {label: i for i, label in enumerate(labels["survey_filter"])}
    night_lookup = {label: i for i, label in enumerate(labels["night"])}
    ccd_lookup = {label: i for i, label in enumerate(labels["ccd"])}

    fieldnames = [
        "survey",
        "night",
        "image_id",
        "ccd_id",
        "filter",
        "survey_filter_zp",
        "survey_filter_zp_std",
        "night_zp",
        "image_zp",
        "image_zp_std",
        "ccd_zp",
        "color_coeff",
        "color_coeff_std",
        "spatial_x_coeff",
        "spatial_y_coeff",
        "total_zp_at_center",
        "intrinsic_scatter",
        "intrinsic_scatter_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_i, image_label in enumerate(labels["image"]):
            survey, night, image_id, ccd_id, filter_name = image_label
            sf_i = sf_lookup[(survey, filter_name)]
            night_i = night_lookup[(survey, night, filter_name)]
            ccd_i = ccd_lookup[(survey, ccd_id, filter_name)]
            total = mean["cal/survey_filter_zp"][sf_i] + night_zp[night_i] + image_zp[image_i] + ccd_zp[ccd_i]
            writer.writerow(
                {
                    "survey": survey,
                    "night": night,
                    "image_id": image_id,
                    "ccd_id": ccd_id,
                    "filter": filter_name,
                    "survey_filter_zp": mean["cal/survey_filter_zp"][sf_i],
                    "survey_filter_zp_std": std["cal/survey_filter_zp"][sf_i],
                    "night_zp": night_zp[night_i],
                    "image_zp": image_zp[image_i],
                    "image_zp_std": std["cal/image_zp_raw"][image_i],
                    "ccd_zp": ccd_zp[ccd_i],
                    "color_coeff": mean["cal/color_coeff"][sf_i],
                    "color_coeff_std": std["cal/color_coeff"][sf_i],
                    "spatial_x_coeff": mean["cal/spatial_x_coeff"][sf_i],
                    "spatial_y_coeff": mean["cal/spatial_y_coeff"][sf_i],
                    "total_zp_at_center": total,
                    "intrinsic_scatter": float(np.asarray(mean["cal/intrinsic_scatter"])),
                    "intrinsic_scatter_std": float(np.asarray(std["cal/intrinsic_scatter"])),
                }
            )


def write_calibrated(path: Path, input_fieldnames: list[str], summary: PosteriorSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    added = [
        "ref_mag",
        "ref_color",
        "calibration_mag_offset",
        "calibration_mag_offset_std",
        "calibration_scale",
        "calibration_scale_std",
        "mag_cal",
        "calibration_residual",
        "calibration_status",
    ]
    fieldnames = input_fieldnames + [name for name in added if name not in input_fieldnames]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for measurement, offset, offset_std, scale, scale_std in zip(
            summary.arrays.measurements,
            summary.offsets,
            summary.offset_std,
            summary.scales,
            summary.scale_std,
        ):
            row = dict(measurement.row)
            mag_cal = measurement.mag_inst - offset
            row.update(
                {
                    "ref_mag": measurement.ref_mag,
                    "ref_color": measurement.color,
                    "calibration_mag_offset": offset,
                    "calibration_mag_offset_std": offset_std,
                    "calibration_scale": scale,
                    "calibration_scale_std": scale_std,
                    "mag_cal": mag_cal,
                    "calibration_residual": mag_cal - measurement.ref_mag,
                    "calibration_status": "ok",
                }
            )
            writer.writerow(row)


def write_params_npz(path: Path, summary: PosteriorSummary) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for key, value in summary.posterior_mean.items():
        payload[f"mean__{key.replace('/', '__')}"] = np.asarray(value)
    for key, value in summary.posterior_std.items():
        payload[f"std__{key.replace('/', '__')}"] = np.asarray(value)
    for name, labels in summary.arrays.labels.items():
        payload[f"labels__{name}"] = np.asarray(["|".join(label) for label in labels])
    for name, values in summary.arrays.parent_indices.items():
        payload[f"indices__{name}"] = np.asarray(values, dtype=int)
    payload["offset_mean"] = np.asarray(summary.offsets)
    payload["offset_std"] = np.asarray(summary.offset_std)
    payload["scale_mean"] = np.asarray(summary.scales)
    payload["scale_std"] = np.asarray(summary.scale_std)
    np.savez(path, **payload)


def validate_args(args: argparse.Namespace) -> None:
    positive = [
        "default_mag_err",
        "max_mag_err",
        "num_warmup",
        "num_samples",
        "num_chains",
        "student_t_df",
        "prior_survey_filter_sigma",
        "prior_night_sigma",
        "prior_image_sigma",
        "prior_ccd_sigma",
        "prior_color_sigma",
        "prior_spatial_sigma",
        "prior_intrinsic_scatter_sigma",
    ]
    for name in positive:
        if float(getattr(args, name)) <= 0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        reference = read_reference(args.reference, star_id_column=args.star_id_column)
        if not reference:
            raise RuntimeError(f"No reference stars found in {args.reference}")
        input_fieldnames, measurements = read_measurements(reference, args)
        if not measurements:
            raise RuntimeError("No matched standard-star measurements found")
        config = BayesianCalibrationConfig(
            use_spatial=args.spatial_order == 1,
            student_t_df=args.student_t_df,
            prior_survey_filter_sigma=args.prior_survey_filter_sigma,
            prior_night_sigma=args.prior_night_sigma,
            prior_image_sigma=args.prior_image_sigma,
            prior_ccd_sigma=args.prior_ccd_sigma,
            prior_color_sigma=args.prior_color_sigma,
            prior_spatial_sigma=args.prior_spatial_sigma,
            prior_intrinsic_scatter_sigma=args.prior_intrinsic_scatter_sigma,
        )
        arrays = build_calibration_arrays(measurements, use_spatial=args.spatial_order == 1)
        summary = run_bayesian_calibration(arrays, config, args)
        write_zeropoints(args.out_zeropoints, summary)
        write_calibrated(args.out_calibrated, input_fieldnames, summary)
        if args.out_params:
            write_params_npz(args.out_params, summary)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"matched standard-star measurements: {len(measurements)}")
    print(f"wrote posterior coefficients: {args.out_zeropoints}")
    print(f"wrote calibrated photometry: {args.out_calibrated}")
    if args.out_params:
        print(f"wrote parameter export: {args.out_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
