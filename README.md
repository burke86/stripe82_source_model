# Stripe 82 Image Downloaders

Scripts for bulk downloading full image products over the Stripe 82 footprint.
They use only the Python standard library.

Default Stripe 82 footprint:

- RA: `300..360` deg and `0..60` deg
- Dec: `-1.25..1.25` deg

Start with a smoke test before running a full download. These data sets are
large.

## Setup

Clone the repository and run commands from the repo root:

```bash
git clone https://github.com/burke86/stripe82_source_model.git
cd stripe82_source_model
```

Check that the scripts run:

```bash
python3 download_sdss_stripe82_corrected_frames.py --help
python3 download_panstarrs_warps_stripe82.py --help
python3 download_ztf_stripe82_science.py --help
```

## SDSS Corrected Frames

Script:

```bash
download_sdss_stripe82_corrected_frames.py
```

This downloads SDSS corrected frames:

```text
frame-[u|g|r|i|z]-RUN-CAMCOL-FIELD.fits.bz2
```

Smoke test with two `r`-band frames:

```bash
python3 download_sdss_stripe82_corrected_frames.py \
  --filters r \
  --camcols 1 \
  --max-files 2 \
  --out-dir sdss_stripe82_corrected_frames
```

Bulk download all SDSS Stripe 82 corrected frames in all five filters:

```bash
python3 download_sdss_stripe82_corrected_frames.py \
  --filters ugriz \
  --camcols 1,2,3,4,5,6 \
  --out-dir sdss_stripe82_corrected_frames
```

Files are written as:

```text
sdss_stripe82_corrected_frames/RERUN/RUN/CAMCOL/frame-*.fits.bz2
```

## Pan-STARRS1 Single-Epoch Warps

Script:

```bash
download_panstarrs_warps_stripe82.py
```

This downloads full Pan-STARRS1 DR2 single-epoch `warp` FITS images from MAST.
These are calibrated skycell images, not cutouts.

Smoke test with two `r`-band warp images:

```bash
python3 download_panstarrs_warps_stripe82.py \
  --filters r \
  --max-files 2 \
  --out-dir panstarrs_stripe82_warps
```

Bulk download all matched single-epoch warp images in all PS1 filters:

```bash
python3 download_panstarrs_warps_stripe82.py \
  --filters grizy \
  --image-types warp \
  --out-dir panstarrs_stripe82_warps \
  --manifest panstarrs_stripe82_warps_manifest.csv
```

To also download weight and mask images:

```bash
python3 download_panstarrs_warps_stripe82.py \
  --filters grizy \
  --image-types warp,warp.wt,warp.mask \
  --out-dir panstarrs_stripe82_warps \
  --manifest panstarrs_stripe82_warps_manifest.csv
```

The script samples the Stripe 82 footprint on a grid and de-duplicates repeated
filenames returned by adjacent grid points.

## ZTF Single-Epoch Science Images

Script:

```bash
download_ztf_stripe82_science.py
```

This downloads full ZTF calibrated single-epoch science images from IRSA. ZTF
products are CCD-quadrant images:

```text
ztf_FILEFRACDAY_FIELD_FILTER_cCCD_o_qQID_sciimg.fits
```

Smoke test with two `zr` science images:

```bash
python3 download_ztf_stripe82_science.py \
  --filters zr \
  --max-files 2 \
  --out-dir ztf_stripe82_science \
  --manifest ztf_stripe82_science_manifest.csv
```

Bulk download all public calibrated science images in `zg`, `zr`, and `zi`:

```bash
python3 download_ztf_stripe82_science.py \
  --filters zg,zr,zi \
  --suffixes sciimg.fits \
  --out-dir ztf_stripe82_science \
  --manifest ztf_stripe82_science_manifest.csv
```

To include masks and PSF catalogs:

```bash
python3 download_ztf_stripe82_science.py \
  --filters zg,zr,zi \
  --suffixes sciimg.fits,mskimg.fits,psfcat.fits \
  --out-dir ztf_stripe82_science \
  --manifest ztf_stripe82_science_manifest.csv
```

To restrict the epoch range, pass an additional IRSA SQL clause:

```bash
python3 download_ztf_stripe82_science.py \
  --filters zr \
  --where "obsdate >= '2020-01-01' AND obsdate < '2021-01-01'" \
  --out-dir ztf_stripe82_science_2020
```

For proprietary ZTF data, set IRSA credentials:

```bash
export IRSA_USER="your_username"
export IRSA_PASSWORD="your_password"
python3 download_ztf_stripe82_science.py --filters zr
```

## General Options

All scripts support:

- `--dry-run`: print URLs and destinations without downloading
- `--max-files N`: stop after matching `N` files
- `--clobber`: overwrite existing files
- `--timeout SECONDS`: HTTP timeout
- `--retries N`: retry failed downloads

Recommended pattern for bulk runs:

```bash
# 1. Inspect what would be downloaded.
python3 SCRIPT.py --dry-run --max-files 10

# 2. Download a tiny sample.
python3 SCRIPT.py --max-files 2

# 3. Launch the full run.
python3 SCRIPT.py --out-dir OUTPUT_DIR
```

