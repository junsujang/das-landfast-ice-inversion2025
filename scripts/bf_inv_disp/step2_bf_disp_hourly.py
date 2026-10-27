"""Step 2: beamform the 5 Hz decimated hourly files against the per-frequency
steering cache built by step1_steer_vec_disp.py.

The record is cut into consecutive cov_duration_h covariance blocks from the
first hour. In each block, window_s snapshots containing any NaN are dropped and
the rest are band-passed and averaged into R(f) (blocks with fewer than
MIN_VALID_SNAPSHOTS are skipped), optionally normalized to coherence
(--is-coherence-norm), decimated by f_bin_skip and beamformed.
The band, f_bin_skip and cache come from step 1 via das_beamforming.resolve_cache;
--directional-floor/--directional-power only select which cache to use.

Writes, under --results-dir (default outputs/bf_inv_disp):
    <band>/<raw|coh>/bfdisp_<block_start>_<band>_<postfix>.npz
holding B and E3, each (nFreqKept, nCurves, nTheta). Axis 1 is the dispersion
curve: column c is (E_all[valid_pairs[c, 0]], C_all[valid_pairs[c, 1]]), at that
bin's exact k(f; E, P); E_all, C_all and valid_pairs are saved in every file.

Blocks are split round-robin over shards (SLURM array variables or
--shard-index/--num-shards); existing outputs are skipped, so failed tasks can
be requeued. See slurms/step2_bf_disp.slurm.

    python step2_bf_disp_hourly.py --directional-floor 1.0 --directional-power 0
    python step2_bf_disp_hourly.py --directional-floor 0.25 --directional-power 1.5 \
        --num-shards 10 --shard-index 3
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from das_tools.utils import proj_path
from das_tools.readers import decimated
from das_tools.processing import pisces25, spectral
from das_tools.beamforming import cov_mat, das_beamforming, steering
from das_tools.utils.helpers import savez_atomic, str2bool

# --------------------------------------------------------------------- #
# Config. Paths default via das_tools.utils.proj_path (DAS_LFI_* env vars).
# --------------------------------------------------------------------- #

DAS_DATA_PATH = proj_path.hourly_das_dir()
RESULTS_DIR = proj_path.output_dir("bf_inv_disp")

MIN_VALID_SNAPSHOTS = 6

FILTER_ORDER = 4


# R(f) -> D(f)^-1/2 R(f) D(f)^-1/2, the complex coherence matrix (unit diagonal).
IS_COHERENCE_NORM = False
EPS_D = 1e-12   # protects low-power channels in the coherence normalizer's diagonal

EPS = 1e-12


# --------------------------------------------------------------------- #
# Output layout and cache checks
# --------------------------------------------------------------------- #

def block_dir(results_dir: Path, band_name: str, norm_tag: str) -> Path:
    """<results>/<band>/<raw|coh>/; a directory per normalization keeps runs apart for step 3."""
    return Path(results_dir) / band_name.replace(" ", "_") / norm_tag


def check_cache_matches(meta: dict, band: dict, fs: float, freq: np.ndarray,
                        n_channels: int, f_bin_skip: int) -> None:
    """Fail fast if the data's fs, frequency grid or channel count differ from the cache's.

    A mismatch would otherwise beamform silently against the wrong bins or geometry.
    """
    if not np.isclose(float(meta["fs"]), fs):
        raise ValueError(f"Steering cache built at fs={meta['fs']}, data is fs={fs}")

    if int(meta["f_bin_skip"]) != int(f_bin_skip):
        raise ValueError(
            f"Steering cache was built with f_bin_skip={meta['f_bin_skip']}, "
            f"this run uses {f_bin_skip}. They index the same frequency axis and "
            "must agree."
        )

    if int(meta["n_channels"]) != int(n_channels):
        raise ValueError(
            f"Steering cache built for {meta['n_channels']} channels, "
            f"data has {n_channels}"
        )

    cached_freq = np.asarray(meta["freq_band"], dtype=float)
    if cached_freq.shape != freq.shape or not np.allclose(cached_freq, freq):
        raise ValueError(
            "Steering cache frequency grid differs from this data's (different fs?). "
            "Rebuild the cache with step1_steer_vec_disp.py."
        )


# --------------------------------------------------------------------- #
# Per-block covariance with this sweep's settings; the diag_* scripts reuse
# these so every script forms R the same way.
# --------------------------------------------------------------------- #

def block_covariance(block: np.ndarray, fs: float, band: dict):
    """cov_mat.block_covariance with MIN_VALID_SNAPSHOTS and FILTER_ORDER."""
    return cov_mat.block_covariance(block, fs, band, MIN_VALID_SNAPSHOTS, FILTER_ORDER)


def normalize_to_coherence(Rk: np.ndarray) -> np.ndarray:
    """cov_mat.normalize_to_coherence with EPS_D."""
    return cov_mat.normalize_to_coherence(Rk, EPS_D)


# --------------------------------------------------------------------- #
# Sweep plumbing -- blocks, shards, atomic writes
# --------------------------------------------------------------------- #

def enumerate_blocks(band: dict, run_start: pd.Timestamp, run_end: pd.Timestamp):
    """Every (block_start, block_end) this band covers, oldest first."""
    delta = pd.Timedelta(hours=band["cov_duration_h"])

    blocks = []
    block_start = run_start
    while block_start < run_end:
        blocks.append((block_start, block_start + delta))
        block_start = block_start + delta

    return blocks


def assign_shard(items: list, shard_index: int, num_shards: int) -> list:
    """This task's round-robin slice of the work list.

    Stride rather than contiguous chunks, since data gaps (cheap blocks) cluster in time.
    """
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards}), got {shard_index}")

    return items[shard_index::num_shards]


def slurm_shard_defaults() -> tuple[int, int]:
    """(shard_index, num_shards) from the SLURM array environment, else (0, 1).

    Relative to SLURM_ARRAY_TASK_MIN, so --array=0-9 and --array=1-10 both give shards 0..9.
    """
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return 0, 1

    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    count = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if count is None:
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", task_min))
        count = task_max - task_min + 1

    return int(task_id) - task_min, int(count)


# --------------------------------------------------------------------- #
# One block
# --------------------------------------------------------------------- #

def process_block(hourly_map, block_start, block_end, band, args, ctx) -> str:
    """Beamform one covariance block. Returns a one-word status for the log."""
    out_path = ctx["band_dir"] / (
        f"bfdisp_{block_start:%Y%m%d_%H%M%S}_{band['name']}_{ctx['postfix']}.npz"
    )
    if out_path.exists():
        return "exists"

    loaded = decimated.load_block(hourly_map, block_start, block_end)
    if loaded is None:
        return "no-data"
    block, channel, fs = loaded

    cov_result = block_covariance(block, fs, band)
    if cov_result is None:
        return "too-few-snapshots"
    Rk_avg, freq, n_valid, n_expected = cov_result

    if args.is_coherence_norm:
        Rk_avg = normalize_to_coherence(Rk_avg)

    # The cache holds one file per kept bin, so decimate R the same way.
    Rk_use = Rk_avg[::args.f_bin_skip, :, :]
    freq_kept = freq[::args.f_bin_skip]

    # Directional weights are already baked into the cached A.
    E3, B = das_beamforming.beamform_k_aoa_implicit_vec_2Dk_cached_from_cov(
        R=Rk_use,
        theta=ctx["theta_rad"],
        cache_dir=ctx["cache_dir"],
        cache_postfix=ctx["postfix"],
        eps=EPS,
        return_E3_sum=False,
    )

    savez_atomic(
        out_path,
        band_name=band["name"],
        fmin=band["fmin"],
        fmax=band["fmax"],
        window_s=band["window_s"],
        cov_duration_h=band["cov_duration_h"],
        block_start=block_start.isoformat(),
        block_end=block_end.isoformat(),
        n_valid_snapshots=n_valid,
        n_expected_snapshots=n_expected,
        min_valid_snapshots=MIN_VALID_SNAPSHOTS,
        delta_f=1.0 / float(band["window_s"]),
        freq=freq_kept,
        f_bin_skip=args.f_bin_skip,
        theta_rad=ctx["theta_rad"],
        theta_deg=np.rad2deg(ctx["theta_rad"]),
        # Curve axis: column c of E3/B is (E_all[vp[c,0]], C_all[vp[c,1]]).
        E_all=ctx["E_all"],
        C_all=ctx["C_all"],
        valid_pairs=ctx["valid_pairs"],
        is_coherence_norm=args.is_coherence_norm,
        # Directional settings from the cache's metadata, i.e. what was applied.
        steering_postfix=ctx["postfix"],
        steering_cache_dir=str(ctx["cache_dir"]),
        directional_floor=ctx["meta"]["directional_floor"],
        directional_power=ctx["meta"]["directional_power"],
        orientation_half_window=ctx["meta"]["orientation_half_window"],
        channel=channel,
        ch_xy=ctx["ch_xy"],
        E3=E3,
        B=B,
    )

    return f"saved ({n_valid}/{n_expected} snapshots)"


# --------------------------------------------------------------------- #
# Per-band sweep
# --------------------------------------------------------------------- #

def run_band(hourly_map: dict, args) -> None:
    # Band, f_bin_skip and cache name all come from what step 1 built.
    band, meta, postfix = das_beamforming.resolve_cache(
        args.results_dir, args.directional_floor, args.directional_power,
        args.f_bin_skip, args.band,
    )
    args.f_bin_skip = int(meta["f_bin_skip"])
    band_name = band["name"]
    print(f"\n=== {band_name}  [{postfix}] ===")

    run_start = min(hourly_map)
    run_end = max(hourly_map) + pd.Timedelta(hours=1)

    first_ds = decimated.open_decimated(hourly_map[run_start])
    if not bool(first_ds.attrs.get("channel_is_bend_relative", False)):
        raise ValueError(
            f"{hourly_map[run_start]} has an absolute channel axis; this sweep "
            "needs the bend-relative axis written by decimate_merge."
        )
    fs = float(first_ds.attrs["fs"])
    channel = first_ds["channel"].values

    freq = spectral.bandlimited_rfft_freqs(
        fs, band["fmin"], band["fmax"], round(band["window_s"] * fs)
    )
    check_cache_matches(meta, band, fs, freq, channel.size, args.f_bin_skip)

    # Saved for reference only; the geometry is already in the cached A.
    ch_xy, _m1 = pisces25.load_leg_geometry(Path(meta["ch_geo_path"]), channel)

    disp = steering.load_dispersion_cache(Path(meta["disp_cache_path"]))

    tag = das_beamforming.normalization_tag(args.is_coherence_norm)
    band_dir = block_dir(args.results_dir, band_name, tag)
    band_dir.mkdir(parents=True, exist_ok=True)

    ctx = {
        "band_dir": band_dir,
        "cache_dir": das_beamforming.steering_cache_dir(args.results_dir, band_name),
        "postfix": postfix,
        "theta_rad": np.asarray(disp["theta"], dtype=float),
        "ch_xy": ch_xy,
        "E_all": disp["E_all"],
        "C_all": disp["C_all"],
        "valid_pairs": disp["valid_pairs"],
        "meta": meta,
    }

    blocks = enumerate_blocks(band, run_start, run_end)
    mine = assign_shard(blocks, args.shard_index, args.num_shards)

    print(f"  {len(blocks)} blocks total, {len(mine)} on shard "
          f"{args.shard_index}/{args.num_shards}")
    print(f"  curves {ctx['valid_pairs'].shape[0]}, theta {ctx['theta_rad'].size}, "
          f"kept bins {meta['n_freq_kept']}")
    print(f"  covariance [{tag}]: "
          f"coherence_norm={args.is_coherence_norm}")
    print(f"  -> {band_dir}")

    for block_start, block_end in mine:
        t0 = time.time()
        status = process_block(hourly_map, block_start, block_end, band, args, ctx)
        if status == "exists":
            continue
        print(f"  [{block_start:%Y-%m-%d %H:%M}] {status} [{time.time() - t0:.1f}s]",
              flush=True)


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def make_parser() -> argparse.ArgumentParser:
    shard_index, num_shards = slurm_shard_defaults()

    parser = argparse.ArgumentParser(
        description="Beamform the decimated hourly files against a cached "
                    "per-frequency dispersion steering grid (step 2)."
    )
    parser.add_argument("--data-root", type=Path, default=DAS_DATA_PATH,
                        help="Directory of hourly decimated h5 files")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Same --results-dir stage 1 was given")
    parser.add_argument("--band", type=str, default=None,
                        help="Band stage 1 built, e.g. 0.10-1.00Hz; only needed when it built several")
    parser.add_argument("--f-bin-skip", type=int, default=None,
                        help="Decimate R to every Nth bin AND name the cache to use "
                             "(default: the skip stage 1 built the cache with; only needed when it built several)")
    parser.add_argument("--is-coherence-norm", type=str2bool, default=IS_COHERENCE_NORM,
                        help="Normalize the averaged covariance to a coherence matrix")
    # Required (no default) so a forgotten pair cannot silently pick some cache.
    parser.add_argument("--directional-floor", type=float, required=True,
                        help="Names the cache to beamform against; must match stage 1")
    parser.add_argument("--directional-power", type=float, required=True,
                        help="Names the cache to beamform against; 0 selects the omni cache")
    parser.add_argument("--num-shards", type=int, default=num_shards)
    parser.add_argument("--shard-index", type=int, default=shard_index)
    return parser


def main() -> None:
    args = make_parser().parse_args()

    hourly_map = decimated.index_hourly_files(args.data_root)
    if not hourly_map:
        raise FileNotFoundError(f"No hourly decimated files found in {args.data_root}")

    print(f"{len(hourly_map)} hourly files, {min(hourly_map)} .. {max(hourly_map)}")

    run_band(hourly_map, args)


if __name__ == "__main__":
    main()
