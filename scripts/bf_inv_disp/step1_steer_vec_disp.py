"""Step 1: build the dispersion-curve grid and the per-frequency steering cache
that step2_bf_disp_hourly.py beamforms against.

The flexural-gravity curve k(f; E, P) is solved on the band's rFFT bins for every
(E, P) in ICE_CFG, and for each kept bin a steering matrix over (valid curve,
theta) is cached with the fiber's directional weighting
(das_beamforming.directional_weight_table) baked in. The band, --f-bin-skip and
directional setting are set here only and recorded in steering_meta_<postfix>.json
for the later steps. The first hourly file supplies fs and the channel axis;
channel coordinates come from data/channel_geometry_20260908_L.npz.

Writes, under --results-dir (default outputs/bf_inv_disp):
    disp_curves/disp_cache_<band>.npz              full band, independent of skip
    steering/<band>/A_f_XXXX_<postfix>.npz         one per kept frequency bin
    steering/<band>/steering_meta_<postfix>.json

<postfix> follows das_beamforming.steering_postfix (omni, dirSen_025_pwr150_skip2, ...).
The cache can reach tens of GB; check with --dry-run (--max-cache-gb guards it).

    python step1_steer_vec_disp.py --directional-floor 1.0 --directional-power 0 --dry-run
    python step1_steer_vec_disp.py --directional-floor 0.25 --directional-power 1.5 \
        --fmin 0.10 --fmax 0.60 --f-bin-skip 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from das_tools.utils import proj_path
from das_tools.readers import decimated
from das_tools.processing import pisces25, spectral
from das_tools.beamforming import das_beamforming, steering

# --------------------------------------------------------------------- #
# Config. Paths default via das_tools.utils.proj_path (DAS_LFI_* env vars).
# --------------------------------------------------------------------- #

DAS_DATA_PATH = proj_path.hourly_das_dir()
CH_GEO_PATH = proj_path.data_dir("channel_geometry_20260908_L.npz")
RESULTS_DIR = proj_path.output_dir("bf_inv_disp")

THETA_STEP_DEG = 2.0

# Ice / hypothesis grid. arange is half-open, so the maxima are nudged by half a
# step to make E = 10.0 GPa and P = 2.5 MPa inclusive.
ICE_CFG = steering.IceConfig(
    nu=0.22,
    rho_i=910.0,
    rho_w=1025.0,
    h_i=0.98,
    h_w=10.0,
    E_min=0.3e9, E_max=10.0e9 + 0.05e9, E_step=0.1e9,
    C_min=0.0,   C_max=2.5e6 + 0.05e6,  C_step=0.1e6,
)

# Channels on each side used to estimate the local fiber tangent.
ORIENTATION_HALF_WINDOW = 2

F_BIN_SKIP = 2


# --------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------- #

steering_postfix = das_beamforming.steering_postfix


def disp_cache_path(results_dir: Path, band_name: str) -> Path:
    return Path(results_dir) / "disp_curves" / f"disp_cache_{band_name.replace(' ', '_')}.npz"


# --------------------------------------------------------------------- #
# Setup read from the data
# --------------------------------------------------------------------- #

def load_reference_geometry(das_data_path: Path, ch_geo_path: Path):
    """(ch_xy, channel, fs, run_start) from the earliest hourly file.

    The bend-relative channel axis is the same in every hour, so one file suffices.
    """
    hourly_map = decimated.index_hourly_files(das_data_path)
    if not hourly_map:
        raise FileNotFoundError(f"No hourly decimated files found in {das_data_path}")

    run_start = min(hourly_map)
    first_ds = decimated.open_decimated(hourly_map[run_start])

    if not bool(first_ds.attrs.get("channel_is_bend_relative", False)):
        raise ValueError(
            f"{hourly_map[run_start]} has an absolute channel axis; this stage "
            "needs the bend-relative axis written by decimate_merge."
        )

    fs = float(first_ds.attrs["fs"])
    channel = first_ds["channel"].values

    ch_xy, _m1 = pisces25.load_leg_geometry(ch_geo_path, channel)

    print(f"Reference hour      : {run_start} ({hourly_map[run_start].name})")
    print(f"fs                  : {fs} Hz")
    print(f"Channels            : {channel.size} "
          f"(offsets {channel.min()} .. {channel.max()} from the bend)")

    return ch_xy, channel, fs, run_start


def band_frequencies(fs: float, band: dict) -> np.ndarray:
    """The band's rFFT bins, identical to those of step 2's window_s snapshots."""
    n_win_samples = round(band["window_s"] * fs)
    return spectral.bandlimited_rfft_freqs(fs, band["fmin"], band["fmax"], n_win_samples)


# --------------------------------------------------------------------- #
# Size estimate
# --------------------------------------------------------------------- #

def report_cache_size(n_channels: int, n_curves: int, n_theta: int,
                      n_freq_kept: int) -> float:
    """Print and return the projected steering-cache size in GB (uncompressed complex128).

    The per-file figure is also roughly step 2's peak memory per frequency.
    """
    bytes_per_file = n_channels * n_curves * n_theta * 16
    total_gb = bytes_per_file * n_freq_kept / 1024 ** 3

    print(f"  steering matrix / freq : ({n_channels}, {n_curves} curves x {n_theta} theta) complex128")
    print(f"  per file               : {bytes_per_file / 1024 ** 3:.2f} GB")
    print(f"  files                  : {n_freq_kept}")
    print(f"  TOTAL                  : {total_gb:.1f} GB")

    return total_gb


# --------------------------------------------------------------------- #
# Per-band build
# --------------------------------------------------------------------- #

def build_band(band: dict, ch_xy: np.ndarray, fs: float, args) -> None:
    band_name = band["name"]
    postfix = steering_postfix(args.directional_floor, args.directional_power, args.f_bin_skip)

    print(f"\n=== {band_name}  [{postfix}] ===")

    freq = band_frequencies(fs, band)
    if freq.size == 0:
        raise ValueError(
            f"[{band_name}] no rFFT bins land in [{band['fmin']}, {band['fmax']}] Hz "
            f"for a {band['window_s']} s window at {fs} Hz"
        )

    freq_kept = freq[::args.f_bin_skip]
    span = f"{freq[0]:.4f} .. {freq[-1]:.4f} Hz" if freq.size > 1 else f"{freq[0]:.4f} Hz"
    df = f", df = {freq[1] - freq[0]:.5f} Hz" if freq.size > 1 else ""
    print(f"  band bins              : {freq.size} ({span}{df})")
    print(f"  kept (f_bin_skip={args.f_bin_skip}): {freq_kept.size}")

    theta_deg = np.arange(0.0, 360.0, args.theta_step_deg)
    theta_rad = np.deg2rad(theta_deg)

    # Built on the full band so changing f_bin_skip never invalidates it.
    cache_path = disp_cache_path(args.results_dir, band_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  dispersion cache       : {cache_path}")
    disp = steering.load_or_build_disp_curves(cache_path, freq, theta_rad, ICE_CFG)

    k_disp_valid = disp["k_disp_valid"]          # (nCurves_valid, nF)
    n_curves = k_disp_valid.shape[0]
    n_total = disp["E_all"].size * disp["P_all"].size
    print(f"  (E, P) grid            : {disp['E_all'].size} x {disp['P_all'].size} = {n_total}, "
          f"{n_curves} valid (subcritical, converged)")

    total_gb = report_cache_size(
        n_channels=ch_xy.shape[1],
        n_curves=n_curves,
        n_theta=theta_rad.size,
        n_freq_kept=freq_kept.size,
    )

    if args.dry_run:
        print("  --dry-run: nothing written")
        return

    if total_gb > args.max_cache_gb:
        raise RuntimeError(
            f"[{band_name}] projected steering cache is {total_gb:.1f} GB, over the "
            f"--max-cache-gb limit of {args.max_cache_gb:.1f} GB. Raise the limit if "
            "that is really what you want, or cut it down with a larger --f-bin-skip "
            "/ --theta-step-deg, or a coarser (E, P) grid in ICE_CFG."
        )

    cache_dir = das_beamforming.steering_cache_dir(args.results_dir, band_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  steering cache         : {cache_dir}")

    das_beamforming.compute_steering_per_freq_directional(
        cache_dir=cache_dir,
        k_disp_valid=k_disp_valid,
        theta=theta_rad,
        ch_xy=ch_xy,
        f_bin_skip=args.f_bin_skip,
        postfix=postfix,
        directional_floor=args.directional_floor,
        directional_power=args.directional_power,
        orientation_half_window=args.orientation_half_window,
    )

    # Read by later steps (das_beamforming.resolve_cache) to get the band and
    # check the frequency grid and channel geometry.
    meta = {
        "band": band,
        "postfix": postfix,
        "fs": fs,
        "n_channels": int(ch_xy.shape[1]),
        "f_bin_skip": int(args.f_bin_skip),
        "n_freq_band": int(freq.size),
        "n_freq_kept": int(freq_kept.size),
        "freq_band": freq.tolist(),
        "freq_kept": freq_kept.tolist(),
        "theta_step_deg": float(args.theta_step_deg),
        "n_theta": int(theta_rad.size),
        "n_curves_valid": int(n_curves),
        "directional_floor": float(args.directional_floor),
        "directional_power": float(args.directional_power),
        "orientation_half_window": int(args.orientation_half_window),
        "disp_cache_path": str(cache_path),
        "ch_geo_path": str(args.ch_geo_path),
        "ice_cfg": vars(ICE_CFG),
    }
    out_meta = das_beamforming.steering_meta_path(args.results_dir, band_name, postfix)
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_meta}")


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the dispersion grid + per-frequency steering cache for "
                    "the bf_inv_disp pipeline (step 1)."
    )
    parser.add_argument("--data-root", type=Path, default=DAS_DATA_PATH,
                        help="Directory of hourly 5 Hz decimated h5 files (decimate_merge output)")
    parser.add_argument("--ch-geo-path", type=Path, default=CH_GEO_PATH,
                        help="Leg geometry file: ch_xy per bend-relative channel offset")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Root for steering/ and disp_curves/; stage 2 reads from here")
    # The band is defined only here; later steps read it from steering_meta.
    band =das_beamforming.DEFAULT_BAND
    parser.add_argument("--fmin", type=float, default=band["fmin"], help="Band low edge, Hz")
    parser.add_argument("--fmax", type=float, default=band["fmax"], help="Band high edge, Hz")
    parser.add_argument("--window-s", type=float, default=band["window_s"],
                        help="Snapshot (FFT) window length, s")
    parser.add_argument("--cov-duration-h", type=float, default=band["cov_duration_h"],
                        help="Covariance block length stage 2 averages snapshots over, h")
    parser.add_argument("--f-bin-skip", type=int, default=F_BIN_SKIP,
                        help="Keep every Nth frequency bin; later steps read it from the cache.")
    parser.add_argument("--theta-step-deg", type=float, default=THETA_STEP_DEG)
    parser.add_argument("--directional-floor", type=float, required=True)
    parser.add_argument("--directional-power", type=float, required=True,
                        help="0 builds an omnidirectional cache")
    parser.add_argument("--orientation-half-window", type=int, default=ORIENTATION_HALF_WINDOW)
    parser.add_argument("--max-cache-gb", type=float, default=80.0,
                        help="Refuse to build a cache projected larger than this")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the projected cache size and exit without writing steering files")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    band = das_beamforming.make_band(args.fmin, args.fmax, args.window_s, args.cov_duration_h)

    ch_xy, _, fs, _ = load_reference_geometry(
        das_data_path=args.data_root,
        ch_geo_path=args.ch_geo_path,
    )

    build_band(band, ch_xy=ch_xy, fs=fs, args=args)


if __name__ == "__main__":
    main()
