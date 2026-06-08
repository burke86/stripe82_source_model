#!/usr/bin/env python3
"""Apply inferred image calibration offsets to a photometry CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply calibration offsets to AGN/source photometry.")
    parser.add_argument("--photometry", required=True, type=Path)
    parser.add_argument("--zeropoints", required=True, type=Path)
    parser.add_argument("--out-calibrated", required=True, type=Path)
    return parser.parse_args()


def key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("survey", ""),
        row.get("night", ""),
        row.get("image_id", ""),
        row.get("ccd_id", ""),
        row.get("filter", ""),
    )


def main() -> int:
    args = parse_args()
    offsets: dict[tuple[str, str, str, str, str], float] = {}
    with args.zeropoints.open(newline="") as handle:
        for row in csv.DictReader(handle):
            offset = parse_float(row.get("total_zp_at_center"))
            if offset is not None:
                offsets[key(row)] = offset

    with args.photometry.open(newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames:
            raise SystemExit(f"{args.photometry} has no header")
        fieldnames = list(reader.fieldnames)
        for column in ("calibration_mag_offset", "calibration_scale", "mag_cal", "calibration_status"):
            if column not in fieldnames:
                fieldnames.append(column)
        args.out_calibrated.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        matched = 0
        with args.out_calibrated.open("w", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                count += 1
                offset = offsets.get(key(row))
                mag_inst = parse_float(row.get("mag_inst"))
                if offset is None or mag_inst is None:
                    row["calibration_status"] = "missing_calibration"
                else:
                    matched += 1
                    row["calibration_mag_offset"] = f"{offset:.9g}"
                    row["calibration_scale"] = f"{10 ** (-0.4 * offset):.9g}"
                    row["mag_cal"] = f"{mag_inst - offset:.9g}"
                    row["calibration_status"] = "ok"
                writer.writerow(row)
    print(f"wrote {count} rows, matched {matched}: {args.out_calibrated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
