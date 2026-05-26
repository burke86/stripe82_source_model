#!/usr/bin/env python3
"""Download ZTF calibrated single-epoch science images over Stripe 82.

ZTF science images are CCD-quadrant products, not SDSS-style run/camcol/field
frames. This script queries the IRSA ZTF science-exposure API for images
overlapping the Stripe 82 footprint, de-duplicates returned products, and
downloads the full calibrated science images:

    ztf_FILEFRACDAY_PADDEDFIELD_FILTER_cCCD_o_qQID_sciimg.fits

Default footprint:
    RA 300..360 and 0..60 deg, Dec -1.25..1.25 deg

The default query is split into two rectangular sky regions so it can straddle
RA = 0. Start with ``--dry-run`` or ``--max-files`` before launching a large
download.
"""

from __future__ import annotations

import argparse
import base64
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


ZTF_SEARCH_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
ZTF_DATA_BASE_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"
USER_AGENT = "ztf-stripe82-science-downloader/1.0"
FILTERS = {"zg", "zr", "zi"}
DEFAULT_COLUMNS = (
    "filefracday,field,ccdid,qid,filtercode,imgtypecode,obsdate,ra,dec"
)


@dataclass(frozen=True)
class Region:
    name: str
    ra: float
    dec: float
    width: float
    height: float


@dataclass(frozen=True)
class ScienceRecord:
    filefracday: str
    field: str
    ccdid: str
    qid: str
    filtercode: str
    imgtypecode: str
    obsdate: str = ""
    ra: str = ""
    dec: str = ""

    @property
    def paddedfield(self) -> str:
        return f"{int(self.field):06d}"

    @property
    def paddedccdid(self) -> str:
        return f"{int(self.ccdid):02d}"

    @property
    def normalized_imgtypecode(self) -> str:
        return self.imgtypecode or "o"


def auth_header(user: str | None, password: str | None) -> str | None:
    if not user and not password:
        return None
    if not user or password is None:
        raise ValueError("--irsa-user and --irsa-password must be supplied together")
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def request_url(
    url: str,
    *,
    timeout: float,
    start: int | None = None,
    authorization: str | None = None,
) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if authorization:
        headers["Authorization"] = authorization
    if start is not None and start > 0:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_regions(value: str) -> list[Region]:
    regions: list[Region] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        parts = item.split(",")
        if len(parts) not in {4, 5}:
            raise argparse.ArgumentTypeError(
                "Regions must look like 'name,ra,dec,width,height' or 'ra,dec,width,height'"
            )
        if len(parts) == 4:
            name = f"region{len(regions) + 1}"
            ra_text, dec_text, width_text, height_text = parts
        else:
            name, ra_text, dec_text, width_text, height_text = parts
        try:
            region = Region(
                name=name.strip(),
                ra=float(ra_text),
                dec=float(dec_text),
                width=float(width_text),
                height=float(height_text),
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Region coordinates and sizes must be numeric"
            ) from exc
        if not 0 <= region.ra <= 360:
            raise argparse.ArgumentTypeError("Region RA must be between 0 and 360")
        if region.width < 0 or region.height < 0:
            raise argparse.ArgumentTypeError("Region width/height must be non-negative")
        regions.append(region)
    if not regions:
        raise argparse.ArgumentTypeError("At least one region is required")
    return regions


def make_query_url(
    region: Region,
    *,
    filters: list[str],
    where: str | None,
    intersect: str,
    columns: str,
) -> str:
    clauses = ["imgtypecode='o'"]
    if filters:
        quoted_filters = ",".join(f"'{filtercode}'" for filtercode in filters)
        clauses.append(f"filtercode IN ({quoted_filters})")
    if where:
        clauses.append(f"({where})")

    params = {
        "POS": f"{region.ra:.8f},{region.dec:.8f}",
        "SIZE": f"{region.width:.8f},{region.height:.8f}",
        "INTERSECT": intersect,
        "COLUMNS": columns,
        "WHERE": " AND ".join(clauses),
        "ct": "csv",
    }
    return f"{ZTF_SEARCH_URL}?{urlencode(params)}"


def parse_csv_response(text: str) -> list[dict[str, str]]:
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith("\\") and not line.startswith("#")
    ]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def records_from_rows(rows: Iterable[dict[str, str]]) -> list[ScienceRecord]:
    records: list[ScienceRecord] = []
    for row in rows:
        normalized = {key.lower(): (value or "").strip() for key, value in row.items()}
        filefracday = normalized.get("filefracday", "")
        field = normalized.get("field", "")
        ccdid = normalized.get("ccdid", "")
        qid = normalized.get("qid", "")
        filtercode = normalized.get("filtercode", "")
        if not all([filefracday, field, ccdid, qid, filtercode]):
            continue
        records.append(
            ScienceRecord(
                filefracday=filefracday,
                field=field,
                ccdid=ccdid,
                qid=qid,
                filtercode=filtercode,
                imgtypecode=normalized.get("imgtypecode", "o"),
                obsdate=normalized.get("obsdate", ""),
                ra=normalized.get("ra", ""),
                dec=normalized.get("dec", ""),
            )
        )
    return records


def query_region(
    region: Region,
    *,
    filters: list[str],
    where: str | None,
    intersect: str,
    columns: str,
    timeout: float,
    authorization: str | None,
) -> list[ScienceRecord]:
    url = make_query_url(
        region,
        filters=filters,
        where=where,
        intersect=intersect,
        columns=columns,
    )
    text = request_url(url, timeout=timeout, authorization=authorization).decode("utf-8")
    return records_from_rows(parse_csv_response(text))


def product_url(record: ScienceRecord, suffix: str) -> str:
    filefracday = record.filefracday
    year = filefracday[:4]
    month_day = filefracday[4:8]
    fracday = filefracday[8:14]
    filename = (
        f"ztf_{filefracday}_{record.paddedfield}_{record.filtercode}_"
        f"c{record.paddedccdid}_{record.normalized_imgtypecode}_q{record.qid}_{suffix}"
    )
    return f"{ZTF_DATA_BASE_URL}/{year}/{month_day}/{fracday}/{filename}"


def destination_for_url(out_dir: Path, url: str) -> Path:
    marker = "/products/sci/"
    if marker in url:
        return out_dir / url.split(marker, 1)[1]
    return out_dir / url.rsplit("/", 1)[-1]


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    clobber: bool,
    authorization: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not clobber:
        print(f"exists: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    start = 0 if clobber or not temporary.exists() else temporary.stat().st_size

    for attempt in range(1, retries + 1):
        try:
            data = request_url(
                url,
                timeout=timeout,
                start=start,
                authorization=authorization,
            )
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
        "region",
        "filefracday",
        "obsdate",
        "field",
        "ccdid",
        "qid",
        "filtercode",
        "imgtypecode",
        "ra",
        "dec",
        "suffix",
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
        description="Download ZTF calibrated single-epoch science images over Stripe 82."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ztf_stripe82_science"),
        help="Output directory. The IRSA products/sci path is preserved under this root.",
    )
    parser.add_argument(
        "--regions",
        type=parse_regions,
        default=parse_regions("west,330,0,60,2.5;east,30,0,60,2.5"),
        help="Semicolon-separated search regions: name,ra,dec,width,height in degrees. "
        "Default covers RA 300..360 and 0..60, Dec -1.25..1.25.",
    )
    parser.add_argument(
        "--filters",
        default="zg,zr,zi",
        help="Comma-separated ZTF filters to download. Default: zg,zr,zi.",
    )
    parser.add_argument(
        "--suffixes",
        default="sciimg.fits",
        help="Comma-separated science products to download. Default: sciimg.fits. "
        "Examples: sciimg.fits,mskimg.fits,psfcat.fits,scimrefdiffimg.fits.fz.",
    )
    parser.add_argument(
        "--where",
        help="Additional IRSA SQL WHERE clause, e.g. \"obsdate >= '2020-01-01'\".",
    )
    parser.add_argument(
        "--intersect",
        choices=["COVERS", "ENCLOSED", "CENTER", "OVERLAPS"],
        default="OVERLAPS",
        help="IRSA spatial predicate. Default: OVERLAPS.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ztf_stripe82_science_manifest.csv"),
        help="CSV manifest path. Default: ztf_stripe82_science_manifest.csv.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Download/write repeated products returned by multiple regions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be downloaded without downloading them.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Stop after matching this many unique products. Useful for smoke tests.",
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
    parser.add_argument(
        "--irsa-user",
        default=os.environ.get("IRSA_USER"),
        help="Optional IRSA username for proprietary data. Defaults to IRSA_USER.",
    )
    parser.add_argument(
        "--irsa-password",
        default=os.environ.get("IRSA_PASSWORD"),
        help="Optional IRSA password for proprietary data. Defaults to IRSA_PASSWORD.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    filters = [item.strip() for item in args.filters.split(",") if item.strip()]
    invalid_filters = sorted(set(filters) - FILTERS)
    if invalid_filters:
        print(f"invalid filters: {','.join(invalid_filters)}", file=sys.stderr)
        return 2

    suffixes = [item.strip() for item in args.suffixes.split(",") if item.strip()]
    if not suffixes:
        print("--suffixes must include at least one product suffix", file=sys.stderr)
        return 2
    if args.max_files is not None and args.max_files < 1:
        print("--max-files must be greater than zero", file=sys.stderr)
        return 2

    try:
        authorization = auth_header(args.irsa_user, args.irsa_password)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    seen: set[str] = set()
    total = 0
    manifest_rows: list[dict[str, str]] = []

    for region in args.regions:
        print(
            f"querying {region.name} POS={region.ra:.6f},{region.dec:.6f} "
            f"SIZE={region.width:.3f},{region.height:.3f}"
        )
        records = query_region(
            region,
            filters=filters,
            where=args.where,
            intersect=args.intersect,
            columns=DEFAULT_COLUMNS,
            timeout=args.timeout,
            authorization=authorization,
        )
        print(f"found {len(records)} science metadata rows in {region.name}")

        for record in records:
            for suffix in suffixes:
                url = product_url(record, suffix)
                if not args.no_dedupe and url in seen:
                    continue
                seen.add(url)

                total += 1
                destination = destination_for_url(args.out_dir, url)
                manifest_rows.append(
                    {
                        "region": region.name,
                        "filefracday": record.filefracday,
                        "obsdate": record.obsdate,
                        "field": record.field,
                        "ccdid": record.ccdid,
                        "qid": record.qid,
                        "filtercode": record.filtercode,
                        "imgtypecode": record.normalized_imgtypecode,
                        "ra": record.ra,
                        "dec": record.dec,
                        "suffix": suffix,
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
                        authorization=authorization,
                    )

                if args.max_files is not None and total >= args.max_files:
                    append_manifest(args.manifest, manifest_rows)
                    print(f"matched {total} ZTF products")
                    return 0

    append_manifest(args.manifest, manifest_rows)
    print(f"matched {total} ZTF products")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
