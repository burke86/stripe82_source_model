#!/usr/bin/env python3
"""Download full Pan-STARRS1 single-epoch warp images over Stripe 82.

Pan-STARRS does not have SDSS-style Stripe 82 frame IDs. The closest public
data product to "calibrated CCD images at each epoch" is the DR2 single-epoch
``warp`` image: an astrometrically and photometrically calibrated exposure
resampled onto a PS1 skycell. This script samples the Stripe 82 footprint,
queries the MAST PS1 image service for all matching ``warp`` files, de-duplicates
the returned filenames, and downloads the full FITS files.

Default footprint:
    RA 300..360 and 0..60 deg, Dec -1.25..1.25 deg

Start with ``--dry-run`` or ``--max-files``. Full warp images are large, and a
dense Stripe 82 grid can match many epochs.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PS1_FILENAMES_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_IMAGE_BASE_URL = "https://ps1images.stsci.edu"
USER_AGENT = "panstarrs-stripe82-warp-downloader/1.0"
FILTERS = "grizy"


@dataclass(frozen=True)
class Position:
    ra: float
    dec: float
    name: str


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    filter: str = ""
    image_type: str = ""
    mjd: str = ""


def request_url(url: str, *, timeout: float, start: int | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if start is not None and start > 0:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def format_position_name(ra: float, dec: float) -> str:
    ra_text = f"{ra:08.4f}".replace(".", "p")
    dec_prefix = "p" if dec >= 0 else "m"
    dec_text = f"{abs(dec):07.4f}".replace(".", "p")
    return f"ra{ra_text}_dec{dec_prefix}{dec_text}"


def parse_ra_ranges(value: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            start_text, stop_text = item.split(":", 1)
            start = float(start_text)
            stop = float(stop_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "RA ranges must look like '300:360,0:60'"
            ) from exc
        if start < 0 or stop > 360 or stop <= start:
            raise argparse.ArgumentTypeError(
                "Each RA range must satisfy 0 <= start < stop <= 360"
            )
        ranges.append((start, stop))
    if not ranges:
        raise argparse.ArgumentTypeError("At least one RA range is required")
    return ranges


def iter_values(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    epsilon = step / 1000.0
    while value <= stop + epsilon:
        yield round(value, 10)
        value += step


def iter_stripe82_grid(
    ra_ranges: list[tuple[float, float]],
    dec_min: float,
    dec_max: float,
    ra_step: float,
    dec_step: float,
) -> Iterable[Position]:
    for ra_min, ra_max in ra_ranges:
        for ra in iter_values(ra_min, ra_max, ra_step):
            ra_wrapped = 0.0 if abs(ra - 360.0) < 1e-9 else ra
            for dec in iter_values(dec_min, dec_max, dec_step):
                yield Position(
                    ra=ra_wrapped,
                    dec=dec,
                    name=format_position_name(ra_wrapped, dec),
                )


def read_positions_csv(path: Path) -> list[Position]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    positions: list[Position] = []
    for row in rows:
        ra = float(row["ra"])
        dec = float(row["dec"])
        name = (row.get("name") or "").strip() or format_position_name(ra, dec)
        positions.append(Position(ra=ra, dec=dec, name=name))
    if not positions:
        raise RuntimeError(f"No positions found in {path}")
    return positions


def parse_ps1_filename_table(text: str) -> list[ImageRecord]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    header_index = None
    for index, line in enumerate(lines):
        columns = line.split()
        if "filename" in columns:
            header_index = index
            break
    if header_index is None:
        raise RuntimeError("Could not find filename header in PS1 response")

    header = lines[header_index].split()
    records: list[ImageRecord] = []
    for line in lines[header_index + 1 :]:
        parts = line.split()
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        filename = row.get("filename", "").strip()
        if not filename:
            continue
        records.append(
            ImageRecord(
                filename=filename,
                filter=row.get("filter", ""),
                image_type=row.get("type", ""),
                mjd=row.get("mjd", ""),
            )
        )
    return records


def query_images(
    position: Position,
    *,
    filters: str,
    image_types: str,
    timeout: float,
) -> list[ImageRecord]:
    query = urlencode(
        {
            "ra": f"{position.ra:.8f}",
            "dec": f"{position.dec:.8f}",
            "filters": filters,
            "type": image_types,
        }
    )
    text = request_url(f"{PS1_FILENAMES_URL}?{query}", timeout=timeout).decode("utf-8")
    return parse_ps1_filename_table(text)


def make_image_url(filename: str) -> str:
    if filename.startswith("http://") or filename.startswith("https://"):
        return filename
    return f"{PS1_IMAGE_BASE_URL}/{filename.lstrip('/')}"


def destination_for_record(out_dir: Path, record: ImageRecord) -> Path:
    filename = record.filename.lstrip("/")
    if filename.startswith("data/") or filename.startswith("rings."):
        return out_dir / filename
    return out_dir / Path(filename).name


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    clobber: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not clobber:
        print(f"exists: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    start = 0 if clobber or not temporary.exists() else temporary.stat().st_size

    for attempt in range(1, retries + 1):
        try:
            data = request_url(url, timeout=timeout, start=start)
            mode = "wb" if start == 0 else "ab"
            with temporary.open(mode) as handle:
                handle.write(data)
            os.replace(temporary, destination)
            print(f"downloaded: {destination}")
            return
        except HTTPError as exc:
            if exc.code == 416 and temporary.exists():
                os.replace(temporary, destination)
                print(f"downloaded: {destination}")
                return
            if attempt == retries:
                raise
        except URLError:
            if attempt == retries:
                raise

        sleep_seconds = min(60, 2**attempt)
        print(f"retrying in {sleep_seconds}s: {url}", file=sys.stderr)
        time.sleep(sleep_seconds)


def append_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_name",
        "query_ra",
        "query_dec",
        "filter",
        "image_type",
        "mjd",
        "filename",
        "url",
        "destination",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download full Pan-STARRS1 single-epoch warp FITS images over Stripe 82."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("panstarrs_stripe82_warps"),
        help="Output directory. The PS1 server path is preserved under this root.",
    )
    parser.add_argument(
        "--positions-csv",
        type=Path,
        help="Optional CSV with ra,dec columns and optional name column.",
    )
    parser.add_argument(
        "--ra-ranges",
        type=parse_ra_ranges,
        default=parse_ra_ranges("300:360,0:60"),
        help="Comma-separated RA ranges in degrees. Default: 300:360,0:60.",
    )
    parser.add_argument(
        "--dec-min",
        type=float,
        default=-1.25,
        help="Minimum Dec for generated Stripe 82 grid. Default: -1.25.",
    )
    parser.add_argument(
        "--dec-max",
        type=float,
        default=1.25,
        help="Maximum Dec for generated Stripe 82 grid. Default: 1.25.",
    )
    parser.add_argument(
        "--ra-step",
        type=float,
        default=0.25,
        help="RA grid spacing in degrees. Default: 0.25.",
    )
    parser.add_argument(
        "--dec-step",
        type=float,
        default=0.25,
        help="Dec grid spacing in degrees. Default: 0.25.",
    )
    parser.add_argument(
        "--filters",
        default=FILTERS,
        help="Filters to download, any combination of grizy. Default: grizy.",
    )
    parser.add_argument(
        "--image-types",
        default="warp",
        help="Comma-separated PS1 image types. Use warp for calibrated single-epoch images. "
        "Common auxiliary products include warp.wt and warp.mask. Default: warp.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("panstarrs_stripe82_warps_manifest.csv"),
        help="CSV manifest path. Default: panstarrs_stripe82_warps_manifest.csv.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Download/write repeated filenames found by multiple grid positions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be downloaded without downloading them.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Stop after matching this many unique files. Useful for smoke tests.",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds. Default: 120.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of attempts per file. Default: 4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.dec_max < args.dec_min:
        print("--dec-max must be greater than or equal to --dec-min", file=sys.stderr)
        return 2
    if args.ra_step <= 0 or args.dec_step <= 0:
        print("--ra-step and --dec-step must be greater than zero", file=sys.stderr)
        return 2
    if args.max_files is not None and args.max_files < 1:
        print("--max-files must be greater than zero", file=sys.stderr)
        return 2

    filters = "".join(dict.fromkeys(args.filters.lower()))
    invalid_filters = set(filters) - set(FILTERS)
    if invalid_filters:
        print(f"invalid filters: {''.join(sorted(invalid_filters))}", file=sys.stderr)
        return 2

    positions = (
        read_positions_csv(args.positions_csv)
        if args.positions_csv
        else list(
            iter_stripe82_grid(
                args.ra_ranges,
                args.dec_min,
                args.dec_max,
                args.ra_step,
                args.dec_step,
            )
        )
    )
    print(f"checking {len(positions)} positions")

    seen: set[str] = set()
    total = 0
    manifest_rows: list[dict[str, str]] = []

    for position in positions:
        print(f"querying {position.name} ra={position.ra:.6f} dec={position.dec:.6f}")
        records = query_images(
            position,
            filters=filters,
            image_types=args.image_types,
            timeout=args.timeout,
        )
        if not records:
            print(f"no PS1 images found for {position.name}", file=sys.stderr)
            continue

        for record in records:
            if not args.no_dedupe and record.filename in seen:
                continue
            seen.add(record.filename)

            total += 1
            url = make_image_url(record.filename)
            destination = destination_for_record(args.out_dir, record)
            manifest_rows.append(
                {
                    "query_name": position.name,
                    "query_ra": f"{position.ra:.8f}",
                    "query_dec": f"{position.dec:.8f}",
                    "filter": record.filter,
                    "image_type": record.image_type,
                    "mjd": record.mjd,
                    "filename": record.filename,
                    "url": url,
                    "destination": str(destination),
                }
            )

            if args.dry_run:
                print(f"would download: {url} -> {destination}")
            else:
                download_file(
                    url,
                    destination,
                    timeout=args.timeout,
                    retries=args.retries,
                    clobber=args.clobber,
                )

            if args.max_files is not None and total >= args.max_files:
                append_manifest(args.manifest, manifest_rows)
                print(f"matched {total} Pan-STARRS files")
                return 0

    append_manifest(args.manifest, manifest_rows)
    print(f"matched {total} Pan-STARRS files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
