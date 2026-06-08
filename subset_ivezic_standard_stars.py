#!/usr/bin/env python3
"""Stream and subset the Ivezić/Thanjavur Stripe 82 standard-star catalog."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_URL = "https://faculty.washington.edu/ivezic/sdss/calib82/dataV2/stripe82calibStars_v4.2.dat"
USER_AGENT = "stripe82-source-model-standard-star-subsetter/1.0"
BANDS = ("u", "g", "r", "i", "z")


def parse_float(value: str) -> float:
    return float(value)


def angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot(dra, dec1 - dec2)


def open_text(source: str, timeout: float):
    path = Path(source)
    if path.exists():
        if path.suffix == ".gz":
            return gzip.open(path, "rt")
        return path.open()
    request = Request(source, headers={"User-Agent": USER_AGENT})
    response = urlopen(request, timeout=timeout)
    if source.endswith(".gz"):
        return gzip.open(response, "rt")
    return response


def parse_catalog_line(line: str, line_number: int) -> dict[str, str] | None:
    parts = line.split()
    if len(parts) < 37 or not parts[0].startswith("CALIBSTARS"):
        return None
    ra = parse_float(parts[1])
    dec = parse_float(parts[2])
    row: dict[str, str] = {
        "star_id": parts[0],
        "ra": f"{ra:.10f}",
        "dec": f"{dec:.10f}",
        "ntot": parts[5],
        "Ar": parts[6],
    }
    offset = 7
    for band in BANDS:
        nobs = int(float(parts[offset]))
        mmed = parse_float(parts[offset + 1])
        msig = parse_float(parts[offset + 3])
        row[f"{band}_nobs"] = str(nobs)
        row[band] = f"{mmed:.6f}"
        row[f"{band}_err"] = f"{1.25 * msig:.6f}"
        offset += 6
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subset Stripe 82 standard stars near one sky position.")
    parser.add_argument("--source", default=DEFAULT_URL, help="Catalog URL or local file path.")
    parser.add_argument("--ra", required=True, type=float)
    parser.add_argument("--dec", required=True, type=float)
    parser.add_argument("--radius-deg", type=float, default=0.05)
    parser.add_argument("--out-reference", required=True, type=Path)
    parser.add_argument("--min-r-nobs", type=int, default=4)
    parser.add_argument("--max-r-err", type=float, default=0.05)
    parser.add_argument("--max-stars", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fieldnames = [
        "star_id",
        "ra",
        "dec",
        "ntot",
        "Ar",
        *[item for band in BANDS for item in (f"{band}_nobs", band, f"{band}_err")],
    ]
    args.out_reference.parent.mkdir(parents=True, exist_ok=True)
    tmp_reference = args.out_reference.with_suffix(args.out_reference.suffix + ".tmp")
    count = 0
    scanned = 0
    print(f"opening standard-star catalog: {args.source}", flush=True)
    print(
        f"subsetting around ra={args.ra:.8f}, dec={args.dec:.8f}, radius={args.radius_deg:.4f} deg",
        flush=True,
    )
    with open_text(args.source, args.timeout) as handle, tmp_reference.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("CALIBSTARS"):
                continue
            scanned += 1
            row = parse_catalog_line(line, line_number)
            if row is None:
                continue
            ra = float(row["ra"])
            dec = float(row["dec"])
            if angular_sep_deg(ra, dec, args.ra, args.dec) > args.radius_deg:
                continue
            if int(row["r_nobs"]) < args.min_r_nobs:
                continue
            if float(row["r_err"]) > args.max_r_err:
                continue
            writer.writerow(row)
            count += 1
            if args.max_stars is not None and count >= args.max_stars:
                break
            if args.progress_every > 0 and scanned and scanned % args.progress_every == 0:
                print(f"scanned {scanned} catalog rows; matched {count}", flush=True)
    print(f"scanned {scanned} catalog rows")
    if count == 0:
        tmp_reference.unlink(missing_ok=True)
        print("no standard stars matched; output file was not replaced", flush=True)
        return 1
    tmp_reference.replace(args.out_reference)
    print(f"wrote {count} standard stars: {args.out_reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
