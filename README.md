# das-landfast-ice-inversion2025
Processing tools for DAS data collected on landfast sea ice (PISCES 2025): hammer
(tap-test) dispersion inversion and ambient-wave dispersion beamforming.

## Repository layout

```text
das_tools/                 the Python package (installed by pip, see below)
├── readers/               open_das(): Onyx raw files and decimated hourly files
├── processing/            filtering, decimation to hourly files, bend schedule
│                          (pisces25), hammer/tap-test helpers (hammer_utils)
├── physics/               P (QS0), SH0 and flexural-gravity dispersion relations
├── beamforming/           dispersion grid, steering cache, covariance, beamformer
├── ice_inversion/         Bayesian inversion of hammer dispersion curves
├── seismometer/           seismometer reading helpers (ObsPy)
└── utils/                 project-root lookup, posterior stats, script helpers
scripts/
├── bf_inv_disp/           ambient-wave beamforming pipeline (stages 1-4 + diagnostics)
└── seismometer/           swell direction from the 3C geophones
notebooks/
└── tap_inv_batch/         hammer-test preprocessing, inversion and figures
data/                      small inputs (geometry, bend schedule, tap-test picks);
                           raw DAS records go in data/DAS/ and are not tracked
```


# Repository setup

Below is how to set up the Python environment for this repository and run notebooks using the local `das_tools` package.

## Python environment setup

This repo is intended to be used inside a Python virtual environment (Python 3.10 or newer).

A virtual environment, or `venv`, is a local isolated Python environment. It lets this project have its own Python packages without interfering with your system Python or other projects.

From the root of the repo:

```bash
cd /path/to/das-landfast-ice-inversion2025
python3 -m venv .venv
```

Activate the environment.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade the basic package tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Install the package in editable mode

Install the repo as an editable Python package:

```bash
pip install -e ".[dev]"
```

The `-e` means editable install. This is useful during development. `[dev]` adds
Jupyter, pytest and the formatters; on a cluster where you only run the scripts,
`pip install -e .` is enough.

For example, after editable install, this works from notebooks and scripts:

```python
from das_tools.readers import open_das
```

Changes made to files such as:

```text
das_tools/readers/onyx.py
das_tools/processing/filtering.py
das_tools/physics/dispersion_relations.py
```

are picked up without reinstalling the package.

In notebooks, restart the kernel after changing source files, or use:

```python
%load_ext autoreload
%autoreload 2
```

## VS Code setup

Open the repository folder in VS Code.

Select the virtual environment as the Python interpreter:

```text
Cmd/Ctrl + Shift + P
Python: Select Interpreter
Choose .venv
```

For notebooks, register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name das-landfast-ice-inversion2025 --display-name "Python (das-landfast-ice-inversion2025)"
```

Then open a notebook and select:

```text
Python (das-landfast-ice-inversion2025)
```

as the kernel.

---

# Using it from the terminal

Start every terminal session by activating the environment:

```bash
cd /path/to/das-landfast-ice-inversion2025
source .venv/bin/activate
```

Every script documents itself: `--help` lists the options, and the docstring at
the top of each file explains what it does and what it writes.

```bash
python scripts/bf_inv_disp/main_steer_vec_disp.py --help
```

The scripts can be run from any directory; keep the `bf_inv_disp` stages in their
folder, since they import each other.

## Where files are read and written

Every default path is relative to the project folder, through
`das_tools/utils/proj_path.py`:

| location | default | environment variable |
|---|---|---|
| inputs | `data/` | `DAS_LFI_DATA_DIR` |
| raw hammer records | `data/DAS/pisces25_hammer_signals/` | `DAS_LFI_HAMMER_DIR` |
| hourly decimated DAS | `data/DAS/pisces_das_data_10Hz/` | `DAS_LFI_HOURLY_DIR` |
| everything the code writes | `outputs/<pipeline>/` (git-ignored) | `DAS_LFI_OUTPUT_DIR` |
| paper figures | `figures/` | `DAS_LFI_FIGURES_DIR` |

To move a location for every script and notebook at once (for example to a
cluster's scratch disk), set its variable before running:

```bash
export DAS_LFI_HOURLY_DIR=/data/pisces25/hourly
export DAS_LFI_OUTPUT_DIR=/scratch/$USER/das_lfi_outputs
```

A single run can also be pointed elsewhere with the script's own flags
(`--data-root`, `--results-dir`, `--events`, ...), which take precedence over both.

## 1. Hammer (tap-test) inversion

Inputs: the raw Onyx hammer records in `data/DAS/pisces25_hammer_signals/`,
`data/tapTestInv_all.csv` and `data/channel_geometry_20260908.npz`.
Run the notebooks in `notebooks/tap_inv_batch/` in this order; each reads the
previous one's output from `outputs/tap_inv_batch/`:

| step | notebook | writes to `outputs/tap_inv_batch/` |
|---|---|---|
| 1 | `main_preprocess.ipynb` | `dispersion_events_0908.npz`: FK ridge picks per hammer event |
| 2 | `main_invert_fixed.ipynb` | forward-model cache, per-event + global posteriors of (E, nu, h_i) |
| 3 | `main_plot_fixed.ipynb` | `figures/` from step 2 |
| - | `main_invert_sensitivity.ipynb` | `sensitivity_fixed/`: posteriors under swept h_w, rho_i, rho_w |

Open them in Jupyter (`jupyter lab`), or execute them headless from the terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/tap_inv_batch/main_preprocess.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/tap_inv_batch/main_invert_fixed.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/tap_inv_batch/main_plot_fixed.ipynb
```

After step 2, check the error model (residual correlation, effective sample size,
chi2/dof, Gaussianity); it writes to `outputs/tap_inv_batch/residuals/`:

```bash
python notebooks/tap_inv_batch/main_residual_diagnostics.py
```

## 2. Decimate raw DAS to hourly files

The beamforming pipeline reads hourly decimated files, not the raw one-minute
Onyx files. They are made with `das_tools.processing.decimate_and_merge_to_hourly_padded`;
with `bend_csv_path` the channel axis is bend-relative, which the pipeline expects:

```bash
python - <<'EOF'
from pathlib import Path
from das_tools.processing import decimate_and_merge_to_hourly_padded
from das_tools.utils import proj_path

decimate_and_merge_to_hourly_padded(
    das_data_dirs=sorted(Path("/path/to/raw_onyx").glob("2025*")),  # every date folder at once
    output_dir=proj_path.hourly_das_dir(),
    new_fs=10.0,
    bend_csv_path=proj_path.data_dir("bend_change_shortest.csv"),
)
EOF
```

## 3. Ambient-wave dispersion beamforming (`scripts/bf_inv_disp`)

Reads the hourly files from `data/DAS/pisces_das_data_10Hz/` and writes everything
to `outputs/bf_inv_disp/`. Each stage is resumable: finished outputs are skipped on a
rerun. The directional setting (`--directional-floor`, `--directional-power`) and
`--f-bin-skip` name the steering cache, so give every stage the same values.

```bash
cd scripts/bf_inv_disp
DIR="--directional-floor 1.0 --directional-power 0"   # omnidirectional

# Stage 1: dispersion grid + per-frequency steering cache (check the size first)
python main_steer_vec_disp.py $DIR --dry-run
python main_steer_vec_disp.py $DIR

# Stage 2: beamform every covariance block (expensive)
python main_bf_disp_hourly.py $DIR --is-coherence-norm true

# Stage 3: posteriors, time series and figures
python analyze_bf_disp_posterior.py $DIR --is-coherence-norm true

# Stage 4: stitch the per-block posteriors into one small file to copy home
python stitch_bf_disp_series.py --is-coherence-norm true
```

Diagnostics that reuse the stage 1-2 outputs:

```bash
# do the two cable legs agree? (--m1 = channels in leg 1)
python main_bf_crossleg_coh.py --m1 <M1> $DIR --is-coherence-norm true

# how the (E, P) estimate of one block moves with nu, h_w, rho_i, rho_w
python sensitivity_ice_params.py --list-tasks
python sensitivity_ice_params.py --data-file <hourly file>.h5
```

**SLURM.** Stages 2 and 3 and the cross-leg script split their blocks into shards:
launched as a job array they read `SLURM_ARRAY_TASK_ID` / `SLURM_ARRAY_TASK_COUNT`
themselves, or take `--num-shards N --shard-index i`. `sensitivity_ice_params.py`
runs one parameter per array element (`--params`).

## 4. Seismometer swell direction

```bash
python scripts/seismometer/mainSwellDir_seism_Welch.py \
    --data-dir /path/to/seismometer --sensor-no 1 \
    --start 2025-04-20 --end 2025-05-10 --output-dir outputs/seismometer
```

---

## Data directory structure

The notebooks and scripts expect `./data` to look like:

```text
./data
├── bend_change_shortest.csv          bend (hairpin apex) channel over time
├── channel_geometry_20260908.npz     channel x/y positions (ch_xy) and origin
├── channel_geometry_20260908_L.npz
├── tapTestInv_all.csv                hammer events: location, time/distance/frequency windows
└── DAS                               (not tracked; from Zenodo)
    ├── pisces25_hammer_signals       raw Onyx records of the hammer tests
    │   ├── decimator_2025-04-24_00.25.00_UTC_000279.h5
    │   └── ...
    └── pisces_das_data_10Hz          hourly decimated records for the beamforming
        ├── das_20250424_000000_10Hz.h5
        └── ...
```

The small files at the top level are tracked in git; everything under `data/DAS/`
is ignored (`.gitignore`) and has to be copied in separately. Data kept elsewhere
can stay there: point the `DAS_LFI_*` variables above at it.

The DAS data of both the ambient flexural-gravity waves (decimated to 5 Hz) and one-minute long records (sampling frequency at 500 Hz) of hammer shots can be found on Zenodo server <https://doi.org/10.5281/zenodo.23221461>. The seismometer data can be found at <https://doi.org/10.7914/Z4FP-3J26>.

Smith, M., & Jang, J. (2026). Distributed acoustic sensing (DAS) data for property inversion from landfast ice near Utqiagvik, Alaska, April-May 2025. Zenodo. doi: 10.5281/zenodo.23221461

Averbuch, G. (2025). Mechanical interaction of ocean waves and sea ice observed by das, seismometers, and hydrophone. International Federation of Digital Seismograph Networks. doi: 10.7914/Z4FP-3J26


code: MIT; files under data/: CC BY 4.0
