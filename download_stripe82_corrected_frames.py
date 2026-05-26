#!/usr/bin/env python3
"""Download SDSS corrected frames for Stripe 82.

The script queries SkyServer CAS for all imaging runs with ``stripe = 82`` and
then downloads every corrected frame file from SDSS SAS:

    frame-[ugriz]-RUN-CAMCOL-FIELD.fits.bz2

These files are calibrated, sky-subtracted SDSS corrected frames. The full
Stripe 82 data set is large, so start with ``--dry-run`` or a restricted set of
filters before launching a full download.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


DEFAULT_CAS_SQL_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
DEFAULT_SAS_FRAMES_URL = (
    "https://data.sdss.org/sas/dr17/eboss/photoObj/frames"
)

USER_AGENT = "stripe82-corrected-frame-downloader/1.0"
FILTERS = "ugriz"


@dataclass(frozen=True)
class Run:
    rerun: str
    run: int


class LinkParser(HTMLParser):
    """Collect href values from a simple SAS directory listing."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def request_url(url: str, *, timeout: float, start: int | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if start is not None and start > 0:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def query_stripe82_runs(cas_sql_url: str, timeout: float) -> list[Run]:
    query = "SELECT rerun, run FROM Run WHERE stripe = 82 ORDER BY run"
    url = f"{cas_sql_url}?cmd={quote(query)}&format=csv"
    data = request_url(url, timeout=timeout).decode("utf-8")
    lines = [line for line in data.splitlines() if line and not line.startswith("#")]

    runs: list[Run] = []
    for row in csv.DictReader(lines):
        if not row:
            continue
        rerun = (row.get("rerun") or row.get("RERUN") or "").strip()
        run_value = (row.get("run") or row.get("RUN") or "").strip()
        if rerun and run_value:
            runs.append(Run(rerun=rerun, run=int(run_value)))
    if not runs:
        raise RuntimeError(f"No Stripe 82 runs found from CAS query: {url}")
    return runs


def read_runs_csv(path: Path) -> list[Run]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    runs: list[Run] = []
    for row in rows:
        rerun = (row.get("rerun") or row.get("RERUN") or "").strip()
        run_value = (row.get("run") or row.get("RUN") or "").strip()
        if rerun and run_value:
            runs.append(Run(rerun=rerun, run=int(run_value)))
    if not runs:
        raise RuntimeError(f"No rows with rerun,run columns found in {path}")
    return runs


def iter_frame_urls(
    run: Run,
    *,
    sas_frames_url: str,
    camcols: Iterable[int],
    filters: set[str],
    timeout: float,
) -> Iterable[tuple[str, str]]:
    for camcol in camcols:
        directory_url = f"{sas_frames_url.rstrip('/')}/{run.rerun}/{run.run}/{camcol}/"
        try:
            html = request_url(directory_url, timeout=timeout).decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 404:
                print(f"missing directory: {directory_url}", file=sys.stderr)
                continue
            raise

        parser = LinkParser()
        parser.feed(html)
        for href in parser.hrefs:
            filename = href.rsplit("/", 1)[-1]
            if not filename.startswith("frame-") or not filename.endswith(".fits.bz2"):
                continue
            parts = filename.split("-")
            if len(parts) < 5 or parts[1] not in filters:
                continue
            yield urljoin(directory_url, href), f"{run.rerun}/{run.run}/{camcol}/{filename}"


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
            chunk = request_url(url, timeout=timeout, start=start)
            mode = "wb" if start == 0 else "ab"
            with temporary.open(mode) as handle:
                handle.write(chunk)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download every SDSS corrected frame in Stripe 82."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stripe82_corrected_frames"),
        help="Output directory. Files are stored as rerun/run/camcol/frame-*.fits.bz2.",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        help="Optional CSV with rerun,run columns. If omitted, query SkyServer CAS.",
    )
    parser.add_argument(
        "--cas-sql-url",
        default=DEFAULT_CAS_SQL_URL,
        help="SkyServer SQL endpoint used to discover Stripe 82 runs.",
    )
    parser.add_argument(
        "--sas-frames-url",
        default=DEFAULT_SAS_FRAMES_URL,
        help="Base SAS URL ending at photoObj/frames.",
    )
    parser.add_argument(
        "--filters",
        default=FILTERS,
        help="Filters to download, any combination of ugriz. Default: ugriz.",
    )
    parser.add_argument(
        "--camcols",
        default="1,2,3,4,5,6",
        help="Comma-separated camera columns to download. Default: 1,2,3,4,5,6.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be downloaded without downloading them.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Stop after matching this many frame files. Useful for smoke tests.",
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
        help="Number of attempts for each file. Default: 4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    filters = set(args.filters)
    invalid_filters = filters - set(FILTERS)
    if invalid_filters:
        print(f"invalid filters: {''.join(sorted(invalid_filters))}", file=sys.stderr)
        return 2

    try:
        camcols = [int(value) for value in args.camcols.split(",") if value]
    except ValueError:
        print("--camcols must be a comma-separated list of integers", file=sys.stderr)
        return 2
    invalid_camcols = [value for value in camcols if value < 1 or value > 6]
    if invalid_camcols:
        print(f"invalid camcols: {invalid_camcols}", file=sys.stderr)
        return 2
    if args.max_files is not None and args.max_files < 1:
        print("--max-files must be greater than zero", file=sys.stderr)
        return 2

    runs = (
        read_runs_csv(args.runs_csv)
        if args.runs_csv
        else query_stripe82_runs(args.cas_sql_url, args.timeout)
    )
    print(f"found {len(runs)} Stripe 82 runs")

    total = 0
    for run in runs:
        print(f"scanning rerun={run.rerun} run={run.run}")
        for url, relative_path in iter_frame_urls(
            run,
            sas_frames_url=args.sas_frames_url,
            camcols=camcols,
            filters=filters,
            timeout=args.timeout,
        ):
            total += 1
            destination = args.out_dir / relative_path
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
                print(f"matched {total} corrected frame files")
                return 0

    print(f"matched {total} corrected frame files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
