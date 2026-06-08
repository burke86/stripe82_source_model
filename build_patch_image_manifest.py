#!/usr/bin/env python3
"""Build one image manifest from SDSS, PS1, and ZTF patch manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
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
    "science_hdu",
    "psf_hdu",
]


def add_row(rows: list[dict[str, str]], seen: set[tuple[str, str, str]], row: dict[str, str]) -> None:
    key = (row["survey"], row["image_path"], row["filter"])
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def append_sdss(path: Path, rows: list[dict[str, str]], seen: set[tuple[str, str, str]]) -> None:
    if not path.exists():
        return
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {key: row.get(key, "") for key in FIELDNAMES}
            normalized["science_hdu"] = normalized.get("science_hdu") or "0"
            add_row(rows, seen, normalized)


def append_ps1(path: Path, rows: list[dict[str, str]], seen: set[tuple[str, str, str]]) -> None:
    if not path.exists():
        return
    with path.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    aux: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in manifest_rows:
        name = Path(row["destination"]).name
        parts = name.split(".")
        skycell = ".".join(parts[2:4]) if len(parts) > 4 else ""
        key = (skycell, row.get("filter", ""), row.get("mjd", ""))
        aux.setdefault(key, {})[row.get("image_type", "")] = row["destination"]
    for row in manifest_rows:
        if row.get("image_type") != "warp":
            continue
        image_path = row["destination"]
        name = Path(image_path).name
        parts = name.split(".")
        skycell = ".".join(parts[2:4]) if len(parts) > 4 else ""
        key = (skycell, row.get("filter", ""), row.get("mjd", ""))
        aux_paths = aux.get(key, {})
        add_row(
            rows,
            seen,
            {
                "image_path": image_path,
                "survey": "panstarrs",
                "night": (row.get("mjd") or "").split(".")[0],
                "image_id": name.removesuffix(".fits"),
                "ccd_id": skycell,
                "filter": row.get("filter", ""),
                "psf_path": "",
                "noise_path": "",
                "invvar_path": aux_paths.get("warp.wt", ""),
                "variance_path": "",
                "mask_path": aux_paths.get("warp.mask", ""),
                "zeropoint": "",
                "exptime": "43.0",
                "science_hdu": "1",
                "psf_hdu": "",
            },
        )


def append_ztf(path: Path, rows: list[dict[str, str]], seen: set[tuple[str, str, str]]) -> None:
    if not path.exists():
        return
    with path.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    aux: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in manifest_rows:
        key = (row.get("filefracday", ""), row.get("field", ""), row.get("filtercode", ""), row.get("ccdid", ""), row.get("qid", ""))
        aux.setdefault(key, {})[row.get("suffix", "")] = row["destination"]
    for row in manifest_rows:
        if row.get("suffix") != "sciimg.fits":
            continue
        image_path = row["destination"]
        name = Path(image_path).name
        key = (row.get("filefracday", ""), row.get("field", ""), row.get("filtercode", ""), row.get("ccdid", ""), row.get("qid", ""))
        aux_paths = aux.get(key, {})
        add_row(
            rows,
            seen,
            {
                "image_path": image_path,
                "survey": "ztf",
                "night": (row.get("obsdate") or "").split()[0].replace("-", ""),
                "image_id": name.removesuffix("_sciimg.fits"),
                "ccd_id": f"{row.get('ccdid', '')}_q{row.get('qid', '')}",
                "filter": row.get("filtercode", ""),
                "psf_path": "",
                "noise_path": "",
                "invvar_path": "",
                "variance_path": "",
                "mask_path": aux_paths.get("mskimg.fits", ""),
                "zeropoint": "",
                "exptime": "30.0",
                "science_hdu": "0",
                "psf_hdu": "",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined image manifest for a patch test.")
    parser.add_argument("--sdss-manifest", type=Path)
    parser.add_argument("--panstarrs-manifest", type=Path)
    parser.add_argument("--ztf-manifest", type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    if args.sdss_manifest:
        append_sdss(args.sdss_manifest, rows, seen)
    if args.panstarrs_manifest:
        append_ps1(args.panstarrs_manifest, rows, seen)
    if args.ztf_manifest:
        append_ztf(args.ztf_manifest, rows, seen)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows: {args.out_manifest}")
    for survey in sorted({row["survey"] for row in rows}):
        filters = sorted({row["filter"] for row in rows if row["survey"] == survey})
        count = sum(1 for row in rows if row["survey"] == survey)
        print(f"{survey}: {count} rows; filters={','.join(filters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
