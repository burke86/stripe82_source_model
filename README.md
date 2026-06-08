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
python3 prepare_agn_patch_test.py --help
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

## One-AGN Patch Test

Script:

```bash
prepare_agn_patch_test.py
```

One-command full pipeline test for `J230056.54-001711.1`:

```bash
cd /Users/colinburke/research/stripe82_source_model
conda run --no-capture-output -n jaxcpu python -u run_j230056_pipeline_test.py
```

The same runner can be used for other Wu/Shen Stripe 82 quasars by passing an
explicit target id, coordinate, and output directory. A useful extended-source
test case is `J012000.78+005154.0` at `z = 0.33705`:

```bash
cd /Users/colinburke/research/stripe82_source_model
conda run --no-capture-output -n jaxcpu python -u run_j230056_pipeline_test.py \
  --target-id J012000.78+005154.0 \
  --ra 20.00327415 \
  --dec 0.86502325 \
  --out-dir /private/tmp/j012000_extended_patch
```

That command uses the default fast path: it downloads/queries the relevant
SDSS, Pan-STARRS, and ZTF `ugri`-equivalent images, subsets nearby Ivezić
standards, compresses each standard-star stamp to a local marginalized Gaussian
Moffat matched-filter flux likelihood, compresses each AGN stamp to a joint
Moffat point-source plus Sérsic-host flux likelihood, jointly infers the
standard-star calibration and AGN light curve, and writes the light-curve and
image-diagnostic products under `/private/tmp/j012000_extended_patch`. The
calibration is placed on the Pan-STARRS1 magnitude system for `g/r/i`; `u`
remains SDSS-native because PS1 has no `u` band.

This fast joint Moffat model is the default and only light-curve/calibration
path used by `run_j230056_pipeline_test.py`.

The compressor evaluates each source stamp on a small grid in Moffat FWHM and
centroid shift. For standard stars, it analytically profiles over point-source
flux and constant background at every grid point. For the AGN, it profiles over
point-source flux, a constant host-galaxy flux, and constant background using a
fixed Sérsic host basis convolved with the same Moffat PSF. It then marginalizes
that local grid to one Gaussian likelihood per source/image stamp; for AGN rows
this Gaussian includes the point-host flux covariance. NumPyro samples the
calibration hierarchy, latent constant standard-star magnitudes, AGN epoch
magnitudes, and one constant host magnitude per AGN/band against those
compressed flux likelihoods. Local FWHM and centroid are not global NUTS
parameters in the fast path. ZTF uses `SEEING` converted to pixels as the FWHM
grid center, Pan-STARRS uses `CHIP.SEEING`, and `--moffat-fwhm-pix` is the
fallback for images without usable seeing metadata, currently including the SDSS
corrected frames in this test. The default centroid grid is
`--centroid-grid-size 5 --centroid-grid-radius-pix 1.0`. The output tables
record both the original WCS fractional center and the local best-fit center:
`wcs_center_x_pix`, `wcs_center_y_pix`, `fit_center_x_pix`,
`fit_center_y_pix`, `fit_centroid_shift_x_pix`, and
`fit_centroid_shift_y_pix`; `fit_centroid_scope` is
`local_marginalized_grid`.

The AGN host component is always included in the fast path. By default it is an
exponential Sérsic profile (`--host-sersic-n 1.0`) with fixed effective radius
`--host-reff-pix 5.0`, axis ratio `--host-axis-ratio 0.8`, and position angle
`--host-pa-deg 0.0`. If the host is not detected, its latent magnitude is pushed
by the likelihood and the weak `--prior-host-mag-loc 22.0`
`--prior-host-mag-sigma 5.0` prior rather than being removed from the model.
AGN output rows include `host_mag_cal`, `host_mag_err`, and
`host_flux_calibration_scale`; compressed rows include `host_flux`,
`host_flux_err`, `point_host_flux_cov`, `host_rate`, `host_rate_err`,
`point_host_rate_cov`, and the fixed host-shape settings.

The fast compressor also uses survey uncertainty and mask products when they
are present in the image manifest. New full pipeline downloads request
Pan-STARRS `warp.wt`/`warp.mask` products and ZTF `mskimg.fits` products, and
`build_patch_image_manifest.py` connects those paths to the science images.
Existing science-only downloads will still fall back to robust scalar image
noise until the auxiliary products are downloaded.
Compressed photometry rows include `mask_fraction`, the fraction of each stamp
excluded by the survey mask or non-finite pixels. The light-curve plotter hides
points with `mask_fraction > 0.10` by default; adjust this with
`--max-mask-fraction`.

Standard-star constancy is enforced in the flux-space NUTS model with one
latent `star/true_mag` parameter per standard star and band. Each
`star/true_mag` is shared across all images and anchored to the Ivezić/PS1
reference magnitude with the catalog uncertainty, floored by
`--prior-star-mag-floor`. Standard-star output rows include `star_true_mag`,
`star_true_mag_err`, `catalog_residual`, and `calibration_residual`, where
`calibration_residual` is measured against the inferred constant star magnitude.

Default outputs are written under:

```text
/private/tmp/j230056_full_patch
```

Key products:

```text
J230056.54-001711.1_lightcurves_ugri.png
J230056.54-001711.1_joint_moffat_compressed_ugri.csv
J230056.54-001711.1_agn_calibrated_ugri.csv
J230056.54-001711.1_standards_calibrated_ugri.csv
J230056.54-001711.1_zeropoints_ugri.csv
J230056.54-001711.1_quality_diagnostics_ugri.csv
quality_diagnostics/
```

To inspect why a particular image has bad standard-star residuals, make stamp
and context diagnostics for that image:

```bash
conda run --no-capture-output -n jaxcpu python plot_image_fit_diagnostics.py \
  --star-photometry /private/tmp/j230056_full_patch/J230056.54-001711.1_standards_calibrated_ugri.csv \
  --image-manifest /private/tmp/j230056_full_patch/J230056.54-001711.1_existing_manifest_ugri.csv \
  --image rings.v3.skycell.1318.091.wrp.i.56148_52428.fits \
  --out-dir /private/tmp/j230056_full_patch \
  --label clean \
  --prefer-full-stamps
```

This writes `<image>_standard_star_fit_diagnostics.png` and
`<image>_context_diagnostics.png`. The image argument can be either the full
science image path or just its basename.

The download stage is uncapped over the small AGN patch. The joint fitting stage
uses up to 40 standards per image by default. To remove that star cap:

```bash
conda run --no-capture-output -n jaxcpu python -u run_j230056_pipeline_test.py \
  --photometry-max-stars-per-image 0
```

Useful resume/debug options:

```bash
# Reuse already downloaded images/manifests and rerun downstream products.
conda run --no-capture-output -n jaxcpu python -u run_j230056_pipeline_test.py --skip-downloads

# Use only the first 25 existing images for a faster local smoke run.
conda run --no-capture-output -n jaxcpu python -u run_j230056_pipeline_test.py \
  --skip-downloads \
  --photometry-max-images 25

# Regenerate outputs from scratch, preserving already downloaded image files
# that the lower-level downloaders can skip.
conda run --no-capture-output -n jaxcpu python -u run_j230056_pipeline_test.py --force
```

The runner also writes a live log file:

```text
/private/tmp/j230056_full_patch/pipeline.log
```

Equivalent filters are plotted as common bands: SDSS/PS1 `g` and ZTF `zg` are
green `g`, SDSS/PS1 `r` and ZTF `zr` are red `r`, and SDSS/PS1 `i` and ZTF `zi`
are purple `i`. The standard-star panel subtracts each star's median in each
common band, so residual survey jumps should be visible if the inferred
cross-survey calibration has not flattened the standards.

For the current full-pipeline patch test, use only `ugri` equivalents:

- SDSS: `u,g,r,i`
- Pan-STARRS: `g,r,i`
- ZTF: `zg,zr,zi`

Do not include SDSS `z` or Pan-STARRS `z/y` in the calibration/light-curve
manifests.

## Standard-Star Recalibration

Script:

```bash
calibrate_standard_star_photometry.py
```

This script consumes standard-star photometry and infers Bayesian hierarchical
calibration parameters. It is separate from the default one-AGN fast path, which
does its own compressed Moffat photometry and calibration in
`run_joint_moffat_patch_model.py`.

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
- an optional `.npz` export of posterior samples and label tables

The Bayesian calibration script requires `jax` and `numpyro`.

```bash
pip install "jax>=0.4.30" "jaxlib>=0.4.30" "numpyro>=0.15"
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

### Photometry Input

The photometry CSV must contain the same `star_id` plus survey/image metadata
and either instrumental magnitudes or fluxes:

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

### Example Calibration Commands

SDSS-like filters:

```bash
python3 calibrate_standard_star_photometry.py \
  --reference stripe82_standards.csv \
  --photometry standard_star_psf_photometry.csv \
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
  --photometry standard_star_psf_photometry.csv \
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
  --photometry standard_star_psf_photometry.csv \
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
  --photometry standard_star_psf_photometry.csv \
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

This is the multiplicative flux scale for a matching
survey/night/image/CCD/filter.
