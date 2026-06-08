#!/usr/bin/env python3
"""Prepare a tiny SDSS corrected-frame smoke test around one named AGN.

The script derives the sky position from a J-name such as
``J230056.54-001711.1``, queries SDSS CAS for corrected frames that have
detections near that position, downloads a small number of full corrected CCD
frames, and writes:

- a one-object target catalog
- an SDSS image manifest for those downloaded frames

The output catalog is intentionally named generically so it can be reused by
the patch-level fast path.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_TARGET = "J230056.54-001711.1"
DEFAULT_CAS_SQL_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
DEFAULT_SAS_FRAMES_URL = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames"
DEFAULT_SAS_PHOTO_REDUX_URL = "https://data.sdss.org/sas/dr17/eboss/photo/redux"
USER_AGENT = "stripe82-agn-patch-test/1.0"
FILTERS = "ugriz"


@dataclass(frozen=True)
class FrameRecord:
    rerun: str
    run: int
    camcol: int
    field: int


def request_url(url: str, *, timeout: float, start: int | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if start is not None and start > 0:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_j_name(name: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"J(?P<rah>\d{2})(?P<ram>\d{2})(?P<ras>\d{2}(?:\.\d+)?)"
        r"(?P<sign>[+-])(?P<decd>\d{2})(?P<decm>\d{2})(?P<decs>\d{2}(?:\.\d+)?)",
        name.strip(),
    )
    if match is None:
        raise ValueError(f"Cannot parse J2000 coordinate name: {name!r}")

    rah = float(match.group("rah"))
    ram = float(match.group("ram"))
    ras = float(match.group("ras"))
    decd = float(match.group("decd"))
    decm = float(match.group("decm"))
    decs = float(match.group("decs"))
    sign = -1.0 if match.group("sign") == "-" else 1.0

    ra = 15.0 * (rah + ram / 60.0 + ras / 3600.0)
    dec = sign * (decd + decm / 60.0 + decs / 3600.0)
    return ra, dec


def query_sdss_frames(
    *,
    ra: float,
    dec: float,
    radius_arcmin: float,
    max_frames: int | None,
    cas_sql_url: str,
    timeout: float,
) -> list[FrameRecord]:
    radius_deg = radius_arcmin / 60.0
    top_clause = f"TOP {int(max_frames)} " if max_frames is not None else ""
    query = f"""
SELECT DISTINCT {top_clause}
    rerun, run, camcol, field
FROM PhotoObjAll
WHERE
    ra BETWEEN {ra - radius_deg:.10f} AND {ra + radius_deg:.10f}
    AND dec BETWEEN {dec - radius_deg:.10f} AND {dec + radius_deg:.10f}
ORDER BY run, camcol, field
"""
    url = f"{cas_sql_url}?cmd={quote(query)}&format=csv"
    data = request_url(url, timeout=timeout).decode("utf-8")
    lines = [line for line in data.splitlines() if line and not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    frames: list[FrameRecord] = []
    for row in rows:
        rerun = (row.get("rerun") or row.get("RERUN") or "").strip()
        run = (row.get("run") or row.get("RUN") or "").strip()
        camcol = (row.get("camcol") or row.get("CAMCOL") or "").strip()
        field = (row.get("field") or row.get("FIELD") or "").strip()
        if rerun and run and camcol and field:
            frames.append(
                FrameRecord(
                    rerun=rerun,
                    run=int(run),
                    camcol=int(camcol),
                    field=int(field),
                )
            )
    if not frames:
        raise RuntimeError(
            "No SDSS PhotoObjAll frames found near "
            f"ra={ra:.8f}, dec={dec:.8f}; try increasing --query-radius-arcmin"
        )
    return frames


def parse_sdss_frame(value: str) -> FrameRecord:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--sdss-frame must be RERUN,RUN,CAMCOL,FIELD")
    rerun, run, camcol, field = parts
    if not rerun:
        raise ValueError("--sdss-frame rerun must not be empty")
    return FrameRecord(rerun=rerun, run=int(run), camcol=int(camcol), field=int(field))


def frame_filename(frame: FrameRecord, filter_name: str) -> str:
    return f"frame-{filter_name}-{frame.run:06d}-{frame.camcol}-{frame.field:04d}.fits.bz2"


def frame_url(frame: FrameRecord, filter_name: str, sas_frames_url: str) -> str:
    return (
        f"{sas_frames_url.rstrip('/')}/{frame.rerun}/{frame.run}/{frame.camcol}/"
        f"{frame_filename(frame, filter_name)}"
    )


def fpm_filename(frame: FrameRecord, filter_name: str) -> str:
    return f"fpM-{frame.run:06d}-{filter_name}{frame.camcol}-{frame.field:04d}.fit.gz"


def fpm_url(frame: FrameRecord, filter_name: str, sas_photo_redux_url: str) -> str:
    return (
        f"{sas_photo_redux_url.rstrip('/')}/{frame.rerun}/{frame.run}/objcs/{frame.camcol}/"
        f"{fpm_filename(frame, filter_name)}"
    )


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    clobber: bool,
) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not clobber:
        print(f"exists: {destination}")
        return True

    temporary = destination.with_suffix(destination.suffix + ".part")
    start = 0 if clobber or not temporary.exists() else temporary.stat().st_size
    for attempt in range(1, retries + 1):
        try:
            payload = request_url(url, timeout=timeout, start=start)
            mode = "wb" if start == 0 else "ab"
            with temporary.open(mode) as handle:
                handle.write(payload)
            os.replace(temporary, destination)
            print(f"downloaded: {destination}")
            return True
        except HTTPError as exc:
            if exc.code == 404:
                print(f"missing: {url}", file=sys.stderr)
                return False
            if exc.code == 416 and temporary.exists():
                os.replace(temporary, destination)
                print(f"downloaded: {destination}")
                return True
            if attempt == retries:
                raise
        except URLError:
            if attempt == retries:
                raise
        sleep_seconds = min(60, 2**attempt)
        print(f"retrying in {sleep_seconds}s: {url}", file=sys.stderr)
        time.sleep(sleep_seconds)
    return False


def maybe_decompress(path: Path, *, clobber: bool) -> Path:
    if path.suffix != ".bz2":
        return path
    out_path = path.with_suffix("")
    if out_path.exists() and not clobber:
        return out_path
    with bz2.open(path, "rb") as source, out_path.open("wb") as destination:
        destination.write(source.read())
    return out_path


def load_catalog_table(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "--wu-shen-catalog requires pandas plus the reader dependency for "
            f"{path.suffix or 'this file type'}"
        ) from exc

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".h5", ".hdf5"}:
        return pd.read_hdf(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise RuntimeError(f"Unsupported catalog format for --wu-shen-catalog: {path}")


def match_wu_shen_catalog(
    path: Path,
    *,
    target_id: str,
    fallback_ra: float,
    fallback_dec: float,
    max_sep_arcsec: float,
) -> tuple[float, float, dict[str, str]]:
    table = load_catalog_table(path)
    for column in ("ra", "dec"):
        if column not in table.columns:
            raise RuntimeError(f"{path} must contain {column!r}")

    cos_dec = math.cos(math.radians(fallback_dec))
    best_index: int | None = None
    best_sep_deg = math.inf
    for index, (ra_value, dec_value) in enumerate(zip(table["ra"], table["dec"])):
        try:
            row_ra = float(ra_value)
            row_dec = float(dec_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(row_ra) or not math.isfinite(row_dec):
            continue
        sep_deg = math.hypot((row_ra - fallback_ra) * cos_dec, row_dec - fallback_dec)
        if sep_deg < best_sep_deg:
            best_index = index
            best_sep_deg = sep_deg
    if best_index is None:
        raise RuntimeError(f"{path} has no finite ra/dec rows")
    sep_arcsec = best_sep_deg * 3600.0
    if sep_arcsec > max_sep_arcsec:
        raise RuntimeError(
            f"No Wu/Shen catalog match for {target_id} within {max_sep_arcsec:.3f} arcsec; "
            f"nearest is {sep_arcsec:.3f} arcsec"
        )

    row = table.iloc[best_index]
    metadata = {
        "catalog": "WuShen",
        "catalog_path": str(path),
        "catalog_sep_arcsec": f"{sep_arcsec:.6f}",
    }
    for column in ("objID", "surveyID", "plateID", "parentID", "sourceID", "sMag"):
        if column in table.columns:
            metadata[f"catalog_{column}"] = str(row[column])
    return float(row["ra"]), float(row["dec"]), metadata


def write_target_catalog(path: Path, *, target_id: str, ra: float, dec: float, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["star_id", "ra", "dec", *metadata.keys()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"star_id": target_id, "ra": f"{ra:.10f}", "dec": f"{dec:.10f}", **metadata})


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> int:
    fieldnames = [
        "image_path",
        "survey",
        "night",
        "image_id",
        "ccd_id",
        "filter",
        "psf_path",
        "noise_path",
        "invvar_path",
        "variance_path",
        "mask_path",
        "zeropoint",
        "exptime",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a one-AGN SDSS corrected-frame smoke test for JAGUAR photometry."
    )
    parser.add_argument("--target-id", default=DEFAULT_TARGET, help="J2000 object id, default: J230056.54-001711.1")
    parser.add_argument("--ra", type=float, help="Override target RA in degrees.")
    parser.add_argument("--dec", type=float, help="Override target Dec in degrees.")
    parser.add_argument("--wu-shen-catalog", type=Path, help="Optional Wu/Shen catalog table to match by position.")
    parser.add_argument("--catalog-match-arcsec", type=float, default=1.0)
    parser.add_argument(
        "--sdss-frame",
        help="Optional known SDSS frame as RERUN,RUN,CAMCOL,FIELD; bypasses the CAS frame lookup.",
    )
    parser.add_argument("--query-radius-arcmin", type=float, default=2.0)
    parser.add_argument("--filters", default="r", help="Filters to download, any combination of ugriz. Default: r.")
    parser.add_argument("--max-frames", type=int, help="Maximum distinct SDSS fields to download. Default: no cap.")
    parser.add_argument("--out-dir", type=Path, default=Path("agn_patch_test"))
    parser.add_argument("--target-catalog", type=Path, help="Output one-object catalog CSV.")
    parser.add_argument("--image-manifest", type=Path, help="Output image manifest CSV.")
    parser.add_argument("--cas-sql-url", default=DEFAULT_CAS_SQL_URL)
    parser.add_argument("--sas-frames-url", default=DEFAULT_SAS_FRAMES_URL)
    parser.add_argument("--sas-photo-redux-url", default=DEFAULT_SAS_PHOTO_REDUX_URL)
    parser.add_argument(
        "--skip-sdss-masks",
        action="store_true",
        help="Do not download SDSS fpM mask files or populate manifest mask_path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-compressed", action="store_true", help="Keep manifest paths as .fits.bz2 instead of decompressed .fits.")
    parser.add_argument("--clobber", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        print("--max-frames must be greater than zero", file=sys.stderr)
        return 2
    filters = set(args.filters)
    invalid_filters = filters - set(FILTERS)
    if invalid_filters:
        print(f"invalid filters: {''.join(sorted(invalid_filters))}", file=sys.stderr)
        return 2

    try:
        seed_ra, seed_dec = (
            (args.ra, args.dec) if args.ra is not None and args.dec is not None else parse_j_name(args.target_id)
        )
        metadata = {"catalog": "designation"}
        if args.wu_shen_catalog is not None:
            ra, dec, metadata = match_wu_shen_catalog(
                args.wu_shen_catalog,
                target_id=args.target_id,
                fallback_ra=seed_ra,
                fallback_dec=seed_dec,
                max_sep_arcsec=args.catalog_match_arcsec,
            )
        else:
            ra, dec = seed_ra, seed_dec
        if not math.isfinite(ra) or not math.isfinite(dec):
            raise ValueError("RA/Dec must be finite")

        target_catalog = args.target_catalog or args.out_dir / f"{args.target_id}_catalog.csv"
        image_manifest = args.image_manifest or args.out_dir / f"{args.target_id}_sdss_manifest.csv"
        write_target_catalog(target_catalog, target_id=args.target_id, ra=ra, dec=dec, metadata=metadata)
        frames = (
            [parse_sdss_frame(args.sdss_frame)]
            if args.sdss_frame
            else query_sdss_frames(
                ra=ra,
                dec=dec,
                radius_arcmin=args.query_radius_arcmin,
                max_frames=args.max_frames,
                cas_sql_url=args.cas_sql_url,
                timeout=args.timeout,
            )
        )
        print(f"{args.target_id}: ra={ra:.8f}, dec={dec:.8f}")
        print(f"found {len(frames)} candidate SDSS frame(s)")

        manifest_rows: list[dict[str, str]] = []
        for frame in frames:
            for filter_name in sorted(filters):
                url = frame_url(frame, filter_name, args.sas_frames_url)
                mask_url = fpm_url(frame, filter_name, args.sas_photo_redux_url)
                compressed_path = args.out_dir / "sdss_frames" / frame.rerun / str(frame.run) / str(frame.camcol) / frame_filename(frame, filter_name)
                mask_path = args.out_dir / "sdss_masks" / frame.rerun / str(frame.run) / str(frame.camcol) / fpm_filename(frame, filter_name)
                image_path = compressed_path if args.keep_compressed else compressed_path.with_suffix("")
                if args.dry_run:
                    print(f"would download: {url} -> {compressed_path}")
                    if not args.skip_sdss_masks:
                        print(f"would download: {mask_url} -> {mask_path}")
                    downloaded = True
                    mask_downloaded = not args.skip_sdss_masks
                else:
                    downloaded = download_file(
                        url,
                        compressed_path,
                        timeout=args.timeout,
                        retries=args.retries,
                        clobber=args.clobber,
                    )
                    if downloaded and not args.keep_compressed:
                        image_path = maybe_decompress(compressed_path, clobber=args.clobber)
                    mask_downloaded = False
                    if downloaded and not args.skip_sdss_masks:
                        mask_downloaded = download_file(
                            mask_url,
                            mask_path,
                            timeout=args.timeout,
                            retries=args.retries,
                            clobber=args.clobber,
                        )
                if not downloaded:
                    continue
                image_id = f"sdss-{frame.run:06d}-{frame.camcol}-{frame.field:04d}"
                manifest_rows.append(
                    {
                        "image_path": str(image_path),
                        "survey": "sdss",
                        "night": str(frame.run),
                        "image_id": image_id,
                        "ccd_id": str(frame.camcol),
                        "filter": filter_name,
                        "psf_path": "",
                        "noise_path": "",
                        "invvar_path": "",
                        "variance_path": "",
                        "mask_path": str(mask_path) if mask_downloaded else "",
                        "zeropoint": "",
                        "exptime": "53.907456",
                    }
                )
        count = write_manifest(image_manifest, manifest_rows)
    except (RuntimeError, ValueError, HTTPError, URLError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote target catalog: {target_catalog}")
    print(f"wrote {count} manifest row(s): {image_manifest}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
