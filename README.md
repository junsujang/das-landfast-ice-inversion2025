# das-landfast-ice-inversion2025

Code and processed results for distributed acoustic sensing (DAS) on landfast sea
ice near Utqiagvik, Alaska (PISCES 2025): inversion of ice properties from hammer
(tap-test) dispersion curves and from ambient flexural-gravity waves, with the
seismometer and DAS spectral analyses and the paper's figures.

## What you can run

| | what | where | needs |
|---|---|---|---|
| 1 | Beamforming + inversion of ambient flexural-gravity waves | `scripts/bf_inv_disp/` | hourly 5 Hz DAS files (Zenodo) |
| 2 | Hammer (tap-test) dispersion inversion | `notebooks/tap_inv_batch/` | raw hammer records (Zenodo) |
| 3 | Seismometer decimation and polarization; DAS and seismometer spectrograms | `notebooks/seismometer/`, `scripts/das_spectrogram/` | seismometer data (IRIS), hourly DAS files (Zenodo) |
| 4 | Tap localization: which channel sits where on the cable | `notebooks/tap_localization/` | raw hammer records (Zenodo) |
| 5 | Figures of the main text and the supplementary material | `figures/` | depends on the figure, see [Figures](#5-figures) |

The processed beamforming results that Figs. 7-9 plot are included in
`outputs/bf_inv_disp/`, so those figures run from a fresh clone without
re-running pipeline 1 (Figs. 8 and 9 also need the seismometer results of
pipeline 3 and the SWIFT buoy record).

## Repository layout

```text
das_tools/                 the Python package (installed by pip, see below)
├── readers/               open_das(): Onyx raw files and decimated hourly files
├── processing/            filtering, decimation to hourly files, bend schedule
│                          (pisces25), hammer/tap-test helpers (hammer_utils)
├── physics/               P (QS0), SH0 and flexural-gravity dispersion relations
├── beamforming/           dispersion grid, steering cache, covariance, beamformer
├── hammer_inversion/      Bayesian inversion of hammer dispersion curves
├── tap_localization/      tap-test channel picks and channel positions on the walked route
├── maps/                  KML route, lon/lat <-> metres, satellite basemaps, scale bars
├── seismometer/           seismometer reading helpers (ObsPy)
└── utils/                 project-root lookup, posterior stats, script helpers
scripts/
├── bf_inv_disp/           ambient-wave beamforming pipeline (steps 1-4 + diagnostics, SLURM jobs)
└── das_spectrogram/       30-min DAS spectrogram of the decimated files
notebooks/
├── tap_inv_batch/         hammer-test preprocessing, residual diagnostics, inversion
├── seismometer/           decimation and swell direction from the 3C geophones
└── tap_localization/      tap-test review -> channel positions along the cable
figures/
├── main_text/             one folder per figure (1_..., 2_..., ...), notebook + its PDF
└── supplementary/         one folder per figure (S1_..., S2_..., ...)
data/                      small inputs (geometry, bend schedule, tap-test picks, cable route);
                           downloaded records go in data/DAS/ and data/seismometer*/
outputs/                   everything the code writes (only outputs/bf_inv_disp/ is tracked)
```

Scripts and notebooks are named in run order: `step1_*`, `step2_*`, ... form a
pipeline; `diag_*` are optional checks run after it (shorthand for diagnostics).

---

# Setup

## Python environment

Use a Python virtual environment (Python 3.9 or newer). From the root of the repo:

```bash
cd /path/to/das-landfast-ice-inversion2025
python3 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

`-e` installs the package in editable mode: changes to `das_tools/` are picked up
without reinstalling (in notebooks, restart the kernel or use
`%load_ext autoreload` / `%autoreload 2`). `[dev]` adds Jupyter, pytest and the
formatters; on a cluster where you only run the scripts, `pip install -e .` is
enough.

To use the environment from Jupyter or VS Code, register it as a kernel and
select it in the notebook:

```bash
python -m ipykernel install --user --name das-landfast-ice-inversion2025 --display-name "Python (das-landfast-ice-inversion2025)"
```

Some figure notebooks set `text.usetex = True` (Fig. 5), which needs a LaTeX
installation.

## Where files are read and written

Every default path is relative to the project folder, through
`das_tools/utils/proj_path.py`:

| location | default | environment variable |
|---|---|---|
| inputs | `data/` | `DAS_LFI_DATA_DIR` |
| raw hammer records | `data/DAS/pisces25_hammer_signals/` | `DAS_LFI_HAMMER_DIR` |
| hourly decimated DAS | `data/DAS/pisces_2025_Fs5Hz/` | `DAS_LFI_HOURLY_DIR` |
| everything the code writes | `outputs/<pipeline>/` | `DAS_LFI_OUTPUT_DIR` |
| paper figures | `figures/` | `DAS_LFI_FIGURES_DIR` |

LFI is short for LandFast Ice. To move a location for every script and notebook
at once (for example to a cluster's scratch disk), set its variable first:

```bash
export DAS_LFI_HOURLY_DIR=/path/to/data/pisces25/hourly
export DAS_LFI_OUTPUT_DIR=/path/to/outputs
```

A single run can also be pointed elsewhere with the script's own flags
(`--data-root`, `--results-dir`, ...), which take precedence over both.

Every script documents itself: `--help` lists the options, and the docstring at
the top of each file explains what it reads and writes. Notebooks can be opened
in Jupyter or run headless:

```bash
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

## Data

| data | source | goes in | tracked in git |
|---|---|---|---|
| hourly decimated DAS (5 Hz) | Zenodo, [10.5281/zenodo.23221461](https://doi.org/10.5281/zenodo.23221461) | `data/DAS/pisces_2025_Fs5Hz/` | no |
| raw hammer-shot records (500 Hz, 1 min) | Zenodo, same record | `data/DAS/pisces25_hammer_signals/` | no |
| seismometers N2, N4, N5 (network X1) | IRIS, [10.7914/Z4FP-3J26](https://doi.org/10.7914/Z4FP-3J26) | `data/seismometer/` | no |
| channel geometry, bend schedule, tap-test picks, hammer event list, cable route (KML), drill-hole ice thickness | this repo | `data/` | yes |
| SWIFT 19 buoy telemetry (wind speed, position) | University of Washington, [SWIFT data](https://faculty.washington.edu/jmt3rd/SWIFTdata/DynamicDataLinks.html) | `data/SWIFT19_telemetry.nc` | no |
| satellite images (Planet, Sentinel-2) | third party, not redistributed | `data/satellite/` | no |
| EM ice-thickness surveys | third party, not redistributed | `data/ice_thickness/EM*.csv` | no |

```text
./data
├── bend_change_shortest.csv          bend (hairpin apex) channel over time
├── channel_geometry_20260908.npz     channel x/y positions (ch_xy) and origin
├── channel_geometry_20260908_L.npz   ch_xy of the 90 channels around the bend kept in the hourly files
├── tapTestInv_all.csv                hammer events for the inversion: location, time/distance/frequency windows
├── tapTestLoc_picks.csv              taps for the channel localization, with the reviewed arrival channel
├── CableRoute2025_onice_traced_smoothed.kml   cable route traced on the ice
├── ice_thickness/                    drill-hole thickness (EM*.csv surveys not included)
└── DAS/                              from Zenodo
    ├── pisces25_hammer_signals/      decimator_2025-04-24_00.25.00_UTC_000279.h5, ...
    └── pisces_2025_Fs5Hz/            das_20250424_000000_5Hz.h5, ...
```

The DAS data can be found on Zenodo, <https://doi.org/10.5281/zenodo.23221461>
(Smith & Jang, 2026). The seismometer data can be found at
<https://doi.org/10.7914/Z4FP-3J26> (Averbuch, 2025). The measurements from
SWIFT 19, including the wind speed, can be downloaded from
<https://faculty.washington.edu/jmt3rd/SWIFTdata/DynamicDataLinks.html>; save
the record as `data/SWIFT19_telemetry.nc`.
The ARM data can also be accessed via <https://arctic.labworks.org/data>.

---

# Pipelines

## 1. Ambient flexural-gravity waves: beamforming and inversion (`scripts/bf_inv_disp`)

Beamforms the hourly DAS files against flexural-gravity dispersion curves
k(f; E, P) and turns the beam power into posteriors of Young's modulus E, the
in-plane compression P and the arrival direction theta. Reads
`data/DAS/pisces_2025_Fs5Hz/` and writes to `outputs/bf_inv_disp/`.

| step | script | writes to `outputs/bf_inv_disp/` |
|---|---|---|
| 1 | `step1_steer_vec_disp.py` | `disp_curves/` (dispersion grid) and `steering/` (per-frequency steering cache, can reach tens of GB) |
| 2 | `step2_bf_disp_hourly.py` | `<band>/<raw\|coh>/bfdisp_*.npz`: beam power per covariance block (expensive) |
| 3 | `step3_analyze_bf_disp_posterior.py` | `posterior/<run>/`: per-block (E, P, theta) posteriors, time series, stacked p(E, P), figures |
| 4 | `step4_stitch_bf_disp_series.py` | `summary/`: the per-block posteriors stitched into one small file |
| - | `diag_crossleg_coh.py` | `crossleg/`: do the two cable legs agree? |
| - | `diag_sensitivity_ice_params.py` | `sensitivity/`: how one block's (E, P) moves with nu, h_w, rho_i, rho_w (data of Fig. S7) |

Each step is resumable: finished outputs are skipped on a rerun. Give every step
the same directional setting (`--directional-floor`, `--directional-power`); it
names the steering cache. The frequency band (`--fmin`, `--fmax`, `--window-s`,
`--cov-duration-h`; default 0.10-1.00 Hz, 180 s snapshots, 0.5 h covariance
blocks) and the frequency decimation (`--f-bin-skip`) are set **only in step 1**,
which records them with the steering cache; later steps read them from there.

```bash
cd scripts/bf_inv_disp
DIR="--directional-floor 1.0 --directional-power 0"   # omnidirectional

python step1_steer_vec_disp.py $DIR --fmin 0.10 --fmax 1.00 --f-bin-skip 2 --dry-run   # check the cache size
python step1_steer_vec_disp.py $DIR --fmin 0.10 --fmax 1.00 --f-bin-skip 2
python step2_bf_disp_hourly.py $DIR --is-coherence-norm true
python step3_analyze_bf_disp_posterior.py $DIR --is-coherence-norm true
python step4_stitch_bf_disp_series.py $DIR --is-coherence-norm true

# diagnostics, reusing the step 1-2 outputs
python diag_crossleg_coh.py $DIR --is-coherence-norm true
python diag_sensitivity_ice_params.py --list-tasks
python diag_sensitivity_ice_params.py --data-file <hourly file>.h5
```

Keep the scripts in their folder; they import each other.

**Included results.** The paper's run (omnidirectional, coherence-normalized,
0.10-1.00 Hz, `--f-bin-skip 2`) is tracked in git, so Figs. 7-9 can be drawn
without steps 1-4:

```text
outputs/bf_inv_disp/
├── disp_curves/disp_cache_0.10-1.00Hz.npz
├── posterior/omni_skip2_coh_f0.10-1.00Hz_llnone/
│   ├── config.json
│   ├── posterior_timeseries.{csv,npz}
│   └── posterior_stacked_EP.npz
└── summary/timeseries_stats_omni_skip2_coh_f0.10-1.00Hz_llnone.npz
```

We used slurm to process the data on the remote server.


## 2. Hammer (tap-test) inversion (`notebooks/tap_inv_batch`)

Picks the dispersion curves (QS, QS0, SH0) of every hammer shot from its f-k
spectrum and inverts them for Young's modulus E, Poisson's ratio nu and ice
thickness h_i. Reads the raw hammer records in `data/DAS/pisces25_hammer_signals/`,
`data/tapTestInv_all.csv` and `data/channel_geometry_20260908.npz`. Run in order;
each notebook reads the previous one's output from `outputs/tap_inv_batch/`:

| step | notebook | writes |
|---|---|---|
| 1 | `step1_preprocess.ipynb` | `dispersion_events_0908.npz`: f-k ridge picks per hammer event |
| 2 | `step2_residual_diagnostics.ipynb` | `dispersion_events_0908_clean.npz` (outlier picks removed), the per-mode sigma for step 3, `residuals/`; Fig. S4 |
| 3 | `step3_invert_fixed.ipynb` | forward-model cache and `inversion_results_fixed_0908_customSTD_noMod.npz`: per-event and global posteriors of (E, nu, h_i) |
| 4 | `figures/main_text/6_hammer_inv/step4_plot_fixed.ipynb` | Fig. 6 and Fig. S5 from step 3 |
| - | `invert_sensitivity_test.ipynb` | posteriors under swept h_w, rho_i, rho_w; Fig. S6 |

```bash
for nb in step1_preprocess step2_residual_diagnostics step3_invert_fixed; do
    jupyter nbconvert --to notebook --execute --inplace notebooks/tap_inv_batch/$nb.ipynb
done
jupyter nbconvert --to notebook --execute --inplace figures/main_text/6_hammer_inv/step4_plot_fixed.ipynb
```

## 3. Seismometer and spectrograms

**Seismometer** (`notebooks/seismometer/`), run in order:

| step | notebook | reads | writes |
|---|---|---|---|
| 1 | `main_decimate.ipynb` | raw 1000 Hz day files `data/seismometer/N?/{GP1,GP2,GPZ}.D/` | `data/seismometer_decimated/X1.N?.10Hz.mseed` |
| 2 | `mainSwellDir_seism_Welch.ipynb` | `X1.N?.10Hz.mseed` and the StationXML `X1.N?.xml` in `data/seismometer_decimated/` | `outputs/seismometer/welch_pca_results_N?.csv` |

Step 2 is a Welch-averaged horizontal PCA per 30-min block and frequency: the
swell azimuth, rectilinearity and the spectra that Figs. 3, 8 and 9 use. Copy each
station's `X1.N?.xml` next to its mseed; the sensor number and time range are set
in the notebook's parameters cell.

**DAS spectrogram** (`scripts/das_spectrogram/main_spectrogram.py`): Welch PSD
of every channel of the hourly files in 30-min blocks, written to
`outputs/das_spectrogram/das_spectrogram.npz` (Fig. 3).

```bash
python scripts/das_spectrogram/main_spectrogram.py
python scripts/das_spectrogram/main_spectrogram.py --start 2025-04-26 --end 2025-04-27
```

## 4. Tap localization (`notebooks/tap_localization`)

Registers the DAS channel numbers onto the cable route. A hammer was struck next
to the cable at GPS-logged points; the channel nearest a tap hears it first, so
each tap ties a channel to a position, and the channels in between are placed
along the traced route (`data/CableRoute2025_onice_traced_smoothed.kml`).

`manual_first_arrival_review.ipynb` shows a waterfall of each tap with candidate
first-arrival channels, asks a person to choose one, and writes the channel
positions to `outputs/tap_localization/` (`channel_geometry*.npz`/`.csv`,
`tap_anchors.csv`, `anchor_segments.csv`, `summary.json`).

The reviewed arrival channel of every tap is in the `arrival_channel` column of
`data/tapTestLoc_picks.csv`, and the resulting geometry is in
`data/channel_geometry_20260908*.npz`, which is what the other pipelines read.
The raw hammer records on Zenodo cover only the focus study area, i.e. 6 of the
13 taps (recorded channels 496-644). The 7 taps nearer the start of the cable
(recorded channels 108-382) have no record, so no waterfall is drawn for them
and the review skips them; they keep their pick from the CSV, so the geometry
still uses all 13 taps.

## 5. Figures

Each figure has a folder under `figures/main_text/` or `figures/supplementary/`
with the notebook that draws it, and writes its PDF into that folder.

What each figure needs, beyond the files tracked in this repo:

| fig | notebook | needs | reproducible from public data |
|---|---|---|---|
| 1 | `1_fieldwork_context/satellite_cable_icethickness.ipynb` | satellite image, EM ice thickness, SWIFT | no |
| 2 | `2_zoomin_waterfall/combined_layout_2waterfalls_3col_annotated.ipynb` | `das_20250505_230000_5Hz.h5` and `decimator_2025-04-24_00.27.00_UTC_000281.h5` (Zenodo), SWIFT; the map is drawn on a plain background without the satellite image | yes, with SWIFT |
| 3 | `3_spectrogram_wind/combined_DAS_seismo_wind_3row.ipynb` | pipeline 3 (DAS spectrogram, seismometer), SWIFT | yes, with SWIFT |
| 4 | `4_fg_disp_ex/main_combine_sketch_with_dispersion_curves.ipynb` | nothing | yes |
| 5 | `5_fk_plots/main_hammer_fk_ex.ipynb` | (a) hammer records (Zenodo); (b) a 2-D beamformer output, `outputs/bf2d/*_BF2D_grid_result.npz`
| 6 | `6_hammer_inv/step4_plot_fixed.ipynb` | pipeline 2 | yes |
| 7 | `7_fg_joint_posterior/main_plot_EP.ipynb` | included `outputs/bf_inv_disp/` | yes, from the clone |
| 8 | `8_directional/main_plot_aoa_das_and_seismometer_combined3.ipynb` | included `outputs/bf_inv_disp/`, pipeline 3 (seismometer), SWIFT | yes, with SWIFT |
| 9 | `9_fg_inv_timeseries/mainSeriesAndStat.ipynb` | included `outputs/bf_inv_disp/`, pipeline 3 (seismometer), SWIFT | yes, with SWIFT |
| S1 | `S1_tap_location/main_tap_location.ipynb` | pipeline 4 | yes |
| S2 | `S2_channel_spacing/main_channel_spacing.ipynb` | pipeline 4 | yes |
| S3 | `S3_hammer_inv_loc/main_cable_hammer_layout.ipynb` | satellite image, pipeline 2 step 1 | no |
| S4 | written by `notebooks/tap_inv_batch/step2_residual_diagnostics.ipynb` | pipeline 2 | yes |
| S5 | written by `figures/main_text/6_hammer_inv/step4_plot_fixed.ipynb` | pipeline 2 | yes |
| S6 | written by `notebooks/tap_inv_batch/invert_sensitivity_test.ipynb` | pipeline 2 | yes |
| S7 | data from `scripts/bf_inv_disp/diag_sensitivity_ice_params.py` (`outputs/bf_inv_disp/sensitivity/`) | hourly files (Zenodo) | data yes; plotting notebook not included |
| S8 | `S8_ice_drill_location/main_ice_thickness.ipynb` | satellite image, EM ice thickness | no |
| S9 | `S9_legwise_fk_plot/main_fk_eachLeg.ipynb` | hourly files (Zenodo) | yes |

"Pipeline n" means its outputs in `outputs/`, which are not tracked: run that
pipeline first.

---

# Data citations

Smith, M., & Jang, J. (2026). Distributed acoustic sensing (DAS) data for property
inversion from landfast ice near Utqiagvik, Alaska, April-May 2025. Zenodo.
doi: 10.5281/zenodo.23221461

Averbuch, G. (2025). Mechanical interaction of ocean waves and sea ice observed by
DAS, seismometers, and hydrophone. International Federation of Digital Seismograph
Networks. doi: 10.7914/Z4FP-3J26

# License

Code: MIT. Files under `data/`: CC BY 4.0.
