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

## Standard-Star Recalibration

Step 1 script:

```bash
run_jaguar_standard_star_photometry.py
```

Step 2 script:

```bash
calibrate_standard_star_photometry.py
```

The first script runs JAGUAR spatial PSF fitting at the Ivezić standard-star
positions in each CCD image. The second script uses that PSF photometry to infer
Bayesian hierarchical calibration parameters.

It infers hierarchical calibration terms:

```text
m_inst - m_ref =
    survey_filter_zp[survey, filter]
  + night_zp[survey, night, filter]
  + image_zp[image_id, ccd_id, filter]
  + ccd_zp[survey, ccd_id, filter]
  + color_coeff[survey, filter] * color
  + optional spatial terms
  + residual
```

Then it writes:

- a posterior zeropoint/coefficient table
- a calibrated photometry table with `mag_cal`, residuals, and calibration scales
- an optional `.npz` export for later JAGUAR integration

The JAGUAR PSF runner requires JAGUAR's runtime dependencies, including
`photutils`. The Bayesian calibration script requires `jax` and `numpyro`.

```bash
pip install "jax>=0.4.30" "jaxlib>=0.4.30" "numpyro>=0.15" photutils
```

### Reference Catalog Input

The reference CSV must contain a stable star identifier and reference
magnitudes. Minimal columns:

```text
star_id,ra,dec,u,g,r,i,z
```

Optional uncertainty columns can be included as:

```text
u_err,g_err,r_err,i_err,z_err
```

If your catalog uses prefixed columns such as `sdss_g`, use
`--ref-mag-prefix sdss_`.

### PSF Photometry Input

`run_jaguar_standard_star_photometry.py` writes the photometry CSV expected by
`calibrate_standard_star_photometry.py`. If you provide your own photometry, it
must contain the same `star_id` plus survey/image metadata and either
instrumental magnitudes or fluxes:

```text
star_id,survey,night,image_id,ccd_id,filter,mag_inst,mag_err,x,y
```

or:

```text
star_id,survey,night,image_id,ccd_id,filter,flux,flux_err,exptime,x,y
```

Recommended group identifiers:

- SDSS: `survey,image_id,ccd_id,filter`, where `ccd_id` can be camcol
- Pan-STARRS: `survey,image_id,ccd_id,filter`, where `ccd_id` can be skycell
- ZTF: `survey,image_id,ccd_id,filter`, where `ccd_id` can be `ccdid_qid`

### Image Manifest Input

The JAGUAR runner reads one row per CCD image:

```text
image_path,survey,night,image_id,ccd_id,filter,psf_path,noise_path,invvar_path,variance_path,zeropoint,exptime
```

Required columns:

```text
image_path,survey,night,image_id,ccd_id,filter
```

Optional columns:

- `psf_path`: FITS PSF image. If omitted, the script uses a Gaussian fallback.
- `noise_path`: per-pixel 1-sigma noise image.
- `invvar_path`: inverse-variance image.
- `variance_path`: variance image.
- `zeropoint`: AB zeropoint used to populate JAGUAR `counts_per_mjy`.
- `exptime`: exposure time for instrumental magnitude output.
- `science_hdu`, `psf_hdu`: optional FITS HDU selectors.

Run JAGUAR PSF photometry:

```bash
PYTHONPATH=/path/to/jaguar/src python3 run_jaguar_standard_star_photometry.py \
  --standards stripe82_standards.csv \
  --image-manifest image_manifest.csv \
  --out-photometry standard_star_jaguar_psf_photometry.csv \
  --stamp-radius 15 \
  --fit-method map_only \
  --map-steps 500
```

For a small smoke test:

```bash
PYTHONPATH=/path/to/jaguar/src python3 run_jaguar_standard_star_photometry.py \
  --standards stripe82_standards.csv \
  --image-manifest image_manifest.csv \
  --out-photometry standard_star_jaguar_psf_photometry_smoke.csv \
  --max-stars-per-image 5 \
  --map-steps 50
```

### Example Calibration Commands

SDSS-like filters:

```bash
python3 calibrate_standard_star_photometry.py \
  --reference stripe82_standards.csv \
  --photometry standard_star_jaguar_psf_photometry.csv \
  --out-zeropoints sdss_zeropoints.csv \
  --out-calibrated sdss_calibrated_standard_star_photometry.csv \
  --out-params sdss_calibration_params.npz \
  --ref-filter-map u:u,g:g,r:r,i:i,z:z \
  --color g-r \
  --num-warmup 1000 \
  --num-samples 1000
```

Pan-STARRS single-epoch warps, using SDSS Stripe 82 standards as the reference
system with fitted color terms:

```bash
python3 calibrate_standard_star_photometry.py \
  --reference stripe82_standards.csv \
  --photometry standard_star_jaguar_psf_photometry.csv \
  --out-zeropoints panstarrs_zeropoints.csv \
  --out-calibrated panstarrs_calibrated_standard_star_photometry.csv \
  --out-params panstarrs_calibration_params.npz \
  --ref-filter-map g:g,r:r,i:i,z:z,y:z \
  --color g-r \
  --num-warmup 1000 \
  --num-samples 1000
```

ZTF:

```bash
python3 calibrate_standard_star_photometry.py \
  --reference stripe82_standards.csv \
  --photometry standard_star_jaguar_psf_photometry.csv \
  --out-zeropoints ztf_zeropoints.csv \
  --out-calibrated ztf_calibrated_standard_star_photometry.csv \
  --out-params ztf_calibration_params.npz \
  --ref-filter-map zg:g,zr:r,zi:i \
  --color g-r \
  --num-warmup 1000 \
  --num-samples 1000
```

Include first-order detector-position corrections after the simple zeropoint
fit looks stable:

```bash
python3 calibrate_standard_star_photometry.py \
  --reference stripe82_standards.csv \
  --photometry standard_star_jaguar_psf_photometry.csv \
  --out-zeropoints ztf_zeropoints_spatial.csv \
  --out-calibrated ztf_calibrated_standard_star_photometry_spatial.csv \
  --out-params ztf_calibration_params_spatial.npz \
  --ref-filter-map zg:g,zr:r,zi:i \
  --color g-r \
  --spatial-order 1
```

The `calibration_scale` column is:

```text
calibration_scale = 10 ** (-0.4 * calibration_mag_offset)
```

This is the factor JAGUAR should multiply into `counts_per_mjy` or the rendered
count model for a matching survey/night/image/CCD/filter.
