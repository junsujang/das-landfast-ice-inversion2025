"""Cross-leg coherence diagnostic, run beside step2_bf_disp_hourly.py.

A full-array beam peak in step 2 can be produced by one leg alone. With the
channel axis split as [leg 1, leg 2] at M1 and the cached steering vector split
as a = [a1; a2], this script computes per (f, curve, theta):

    gamma_cross = a1^H C12 a2 / sqrt((a1^H C11 a1)(a2^H C22 a2))
    gamma_leg_l = a_l^H C_ll a_l / (tr(C_ll) |a_l|^2),        l = 1, 2
    B = share_1 + share_2 + share_x,
        share_l = a_l^H C_ll a_l / (tr(R) |a|^2)
        share_x = 2 Re{a1^H C12 a2} / (tr(R) |a|^2)

A two-leg coherent arrival gives |gamma_cross| ~ 1, arg ~ 0; a one-leg peak
gives |gamma_cross| ~ 0. gamma_leg_l is leg l beamformed alone, normalized as B
is; share_x < 0 means the legs disagree. The cross blocks are contracted
directly rather than beamforming a cross-only covariance, which would be
indefinite with zero trace. Runs on the raw or coherence-normalized covariance;
intended for --is-coherence-norm true with the omni cache.

M1 comes from the leg geometry file the steering cache was built with
(pisces25.load_leg_geometry): leg 1 is the channels before the bend (offset < 0).
--m1 overrides it and is part of the output directory name.

Phase 1 (per block, sharded, resumable) reads the hourly decimated files and
the step-1 steering cache, and writes (<results-dir> = outputs/bf_inv_disp)
    <results-dir>/crossleg/<band>/<norm>_<postfix>_m1<M1>/gxleg_<block_start>.npz
holding gamma_cross, gamma_leg1/2, share_leg1/2 (share_x = B - shares) and the
step-2 axes. The full-array B is taken from a matching step-2 block (only its
path stored) or computed here with the same call and stored inline.

Phase 2 (re-runnable over any [fmin, fmax]) scores each block with
step3_analyze_bf_disp_posterior.block_log_likelihood on B; its argmax is the MAP
(theta, E, P). gamma_cross is averaged over frequency as a complex number before
Re/|.|^2/arg are taken, so a cross-leg phase rotating with f cancels. Per block
it reports gamma_bar, gamma_leg_l and the B shares (also as fractions of B) at
the MAP, and where the cross-leg and each leg's own map peak (diagnostic only;
the estimator stays the full-array beam). Writes, beside the gxleg files,
crossleg_summary_<window>.csv/.npz, crossleg_stats_<window>.csv (across-block
statistics) and crossleg_coh_{tf,freq}_<window>.png/.pdf.

Usage (SLURM: slurms/diag_crossleg.slurm; sharded runs skip phase 2):
    python diag_crossleg_coh.py --directional-floor 1.0 --directional-power 0
    python diag_crossleg_coh.py --directional-floor 1.0 --directional-power 0 \
        --summary-only --fmin 0.10 --fmax 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Steps 2 and 3 are imported, not copied, so the covariance and likelihood are
# the same code that produced the blocks compared against.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import step2_bf_disp_hourly as stage2          # noqa: E402
import step3_analyze_bf_disp_posterior as stage3    # noqa: E402

from das_tools.readers import decimated       # noqa: E402
from das_tools.processing import pisces25, spectral  # noqa: E402
from das_tools.beamforming import das_beamforming, steering  # noqa: E402
from das_tools.utils.helpers import savez_atomic, str2bool  # noqa: E402

# --------------------------------------------------------------------- #
# Config (defaults; override on the command line)
# --------------------------------------------------------------------- #

EPS = stage2.EPS

# "re" ranks the cross-leg map by Re{gamma_bar} (coherent and in phase);
# "abs2" by |gamma_bar|^2, which also rewards a constant cross-leg phase offset.
CROSS_METRIC = "re"

AGREE_THETA_DEG = 10.0

# Per-block scalars the across-block statistics are taken over.
STATS_COLUMNS = (
    "re_gamma_map", "abs2_gamma_map",
    "gamma_leg1_map", "gamma_leg2_map",
    "gamma_leg1_legmax", "gamma_leg2_legmax",
    "frac_leg1_map", "frac_leg2_map", "frac_cross_map",
)


# --------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------- #

def crossleg_dir(results_dir: Path, band_name: str, norm_tag: str,
                 postfix: str, m1: int) -> Path:
    """<results>/crossleg/<band>/<norm>_<postfix>_m1<M1>/.

    Every setting that changes gamma_cross is in the name, since phase 1 skips
    blocks whose file already exists.
    """
    return (Path(results_dir) / "crossleg" / band_name.replace(" ", "_") /
            f"{norm_tag}_{postfix}_m1{int(m1)}")


def window_tag(fmin: float, fmax: float) -> str:
    return f"f{fmin:.2f}-{fmax:.2f}Hz"


def check_config(out_dir: Path, cfg: dict) -> None:
    """Refuse to mix a different phase-1 config into an existing directory."""
    path = out_dir / "config.json"
    if not path.exists():
        path.write_text(json.dumps(cfg, indent=2))
        return

    old = json.loads(path.read_text())
    differs = {k: (old.get(k), cfg[k]) for k in cfg if k in old and old[k] != cfg[k]}
    if differs:
        detail = ", ".join(f"{k}: {was!r} -> {now!r}" for k, (was, now) in differs.items())
        raise SystemExit(
            f"{out_dir} already holds gamma blocks computed with a different config "
            f"({detail}).\nThose files would be kept as-is and reported under the new "
            "config. Delete the directory to recompute, or change --m1 / --band / "
            "--directional-* so the run gets its own directory."
        )


# --------------------------------------------------------------------- #
# Reusing a step-2 block
# --------------------------------------------------------------------- #

def stage2_block_path(results_dir: Path, band_name: str, norm_tag: str,
                      postfix: str, block_start: pd.Timestamp) -> Path | None:
    """The step-2 block for this time, or None.

    Also accepts a block directly under <results>/<band>/ for raw runs, as
    step3_analyze_bf_disp_posterior.find_block_dir does; its flags are checked
    by stage2_block_mismatch either way.
    """
    name = f"bfdisp_{block_start:%Y%m%d_%H%M%S}_{band_name}_{postfix}.npz"

    tagged = stage2.block_dir(results_dir, band_name, norm_tag) / name
    if tagged.exists():
        return tagged

    flat = Path(results_dir) / band_name.replace(" ", "_") / name
    if norm_tag == "raw" and flat.exists():
        return flat

    return None


def stage2_block_mismatch(path: Path, args, ctx, freq_kept: np.ndarray) -> str | None:
    """None if that block's B is the one this run would compute, else the reason.

    Reads only the small axes, not B.
    """
    try:
        with np.load(path) as d:
            if bool(d["is_coherence_norm"]) != bool(args.is_coherence_norm):
                return (f"coherence_norm={bool(d['is_coherence_norm'])}, "
                        f"this run uses {bool(args.is_coherence_norm)}")
            if str(d["steering_postfix"]) != ctx["postfix"]:
                return f"steering {str(d['steering_postfix'])!r} != {ctx['postfix']!r}"
            if int(d["f_bin_skip"]) != int(args.f_bin_skip):
                return f"f_bin_skip={int(d['f_bin_skip'])} != {int(args.f_bin_skip)}"

            freq = np.asarray(d["freq"], dtype=float)
            if freq.shape != freq_kept.shape or not np.allclose(freq, freq_kept):
                return "different frequency grid"

            theta = np.asarray(d["theta_rad"], dtype=float)
            if theta.shape != ctx["theta_rad"].shape or not np.allclose(theta, ctx["theta_rad"]):
                return "different theta grid"

            vp = np.asarray(d["valid_pairs"])
            if vp.shape != ctx["valid_pairs"].shape or not np.array_equal(vp, ctx["valid_pairs"]):
                return "different (E, P) curve list"

            if "B" not in d.files:
                return "holds no B array"
    except Exception as exc:
        # Most likely a step-2 write killed mid-flight; recompute rather than trust it.
        return f"unreadable ({exc})"

    return None


# --------------------------------------------------------------------- #
# The cross-leg kernel
# --------------------------------------------------------------------- #

def crossleg_gamma_from_cov(R, theta, cache_dir, cache_postfix, M1,
                            eps=EPS, dtype=np.complex64):
    """gamma_cross, gamma_leg1/2 and share_leg1/2 per (f, curve, theta).

    Follows das_beamforming.beamform_k_aoa_implicit_vec_2Dk_cached_from_cov bin
    for bin (same cache, A_f = A.conj(), same scatter through valid_mask), so
    every output is aligned element-wise with that function's B and E3.

    R : (F, M, M) covariance per kept bin, after normalization.
    M1 : channels in leg 1; the channel axis is ordered [leg 1, leg 2].
    dtype : storage type of gamma_cross (real outputs use the matching float);
        contractions are done in complex128.

    Returns a dict of (F, K_len, A_len) arrays, NaN where the cache marks the
    (f, curve) cell invalid, i.e. where B is NaN.
    """
    R = np.asarray(R)
    if R.ndim != 3:
        raise ValueError(f"Expected R to be (F, M, M). Got {R.shape}")

    F, M, M2 = R.shape
    if M2 != M:
        raise ValueError(f"R must be square in its last dims. Got {R.shape}")

    M1 = int(M1)
    if not 0 < M1 < M:
        raise ValueError(
            f"--m1 must be between 1 and M-1 for an array of M={M} channels, got {M1}"
        )

    A_len = len(theta)
    cache_dir = Path(cache_dir)

    first = np.load(cache_dir / f"A_f_0000_{cache_postfix}.npz")
    K_len = len(first["valid_mask"])

    dtype = np.dtype(dtype)
    rdtype = np.float32 if dtype == np.complex64 else np.float64
    shape = (int(F), K_len, A_len)

    out = {
        "gamma_cross": np.full(shape, np.nan, dtype=dtype),
        "gamma_leg1": np.full(shape, np.nan, dtype=rdtype),
        "gamma_leg2": np.full(shape, np.nan, dtype=rdtype),
        "share_leg1": np.full(shape, np.nan, dtype=rdtype),
        "share_leg2": np.full(shape, np.nan, dtype=rdtype),
    }

    for out_idx in range(F):
        cache = np.load(cache_dir / f"A_f_{out_idx:04d}_{cache_postfix}.npz")
        A = cache["A"]                       # (M, K_valid * A_len)
        valid_mask = cache["valid_mask"]
        K_valid = int(cache["K_valid"])

        if K_valid == 0:
            continue

        if A.shape[0] != M:
            raise ValueError(
                f"Steering cache bin {out_idx} has {A.shape[0]} channels, "
                f"covariance has {M}"
            )

        A_f = A.conj()
        a1 = A_f[:M1, :]
        a2 = A_f[M1:, :]

        Rf = R[out_idx, :, :]
        C11 = Rf[:M1, :M1]
        C12 = Rf[:M1, M1:]
        C22 = Rf[M1:, M1:]

        q12 = np.sum(a1.conj() * (C12 @ a2), axis=0)                 # complex
        p1 = np.real(np.sum(a1.conj() * (C11 @ a1), axis=0))         # >= 0
        p2 = np.real(np.sum(a2.conj() * (C22 @ a2), axis=0))         # >= 0

        a1_norm2 = np.sum(np.abs(a1) ** 2, axis=0)
        a2_norm2 = np.sum(np.abs(a2) ** 2, axis=0)
        tr11 = np.real(np.trace(C11))
        tr22 = np.real(np.trace(C22))
        denom_full = np.maximum((tr11 + tr22) * (a1_norm2 + a2_norm2), eps)

        # q12 is not clipped: its sign and phase are the diagnostic. eps only
        # guards a dead leg against division by zero.
        terms = {
            "gamma_cross": q12 / np.sqrt(np.maximum(p1 * p2, eps)),
            "gamma_leg1": p1 / np.maximum(tr11 * a1_norm2, eps),
            "gamma_leg2": p2 / np.maximum(tr22 * a2_norm2, eps),
            "share_leg1": p1 / denom_full,
            "share_leg2": p2 / denom_full,
        }

        for key, val in terms.items():
            out[key][out_idx, valid_mask, :] = val.reshape(K_valid, A_len, order="C")

    return out


# --------------------------------------------------------------------- #
# Phase 1 -- one block
# --------------------------------------------------------------------- #

def resolve_beam_source(block_start, band, args, ctx) -> tuple[str, str] | None:
    """(path, source) of this block's full-array beam; None if --reuse require
    finds no usable step-2 block. Decided before any data is read."""
    if args.reuse == "never":
        return "", "computed-here"

    cand = stage2_block_path(args.results_dir, band["name"], ctx["norm_tag"],
                             ctx["postfix"], block_start)

    if cand is None:
        return None if args.reuse == "require" else ("", "computed-here")

    why = stage2_block_mismatch(cand, args, ctx, ctx["freq_kept"])
    if why is None:
        return str(cand), "stage2-reuse"

    print(f"    NOTE: not reusing {cand.name}: {why}")
    return None if args.reuse == "require" else ("", "computed-here")


# Arrays phase 1 writes per (f, curve, theta); a block missing any is recomputed.
LEG_KEYS = ("gamma_cross", "gamma_leg1", "gamma_leg2", "share_leg1", "share_leg2")


def block_is_complete(path: Path) -> bool:
    """True if `path` holds every LEG_KEYS array. Reads only the zip directory."""
    try:
        with np.load(path) as d:
            return all(k in d.files for k in LEG_KEYS)
    except Exception:
        return False


def process_block(hourly_map, block_start, block_end, band, args, ctx) -> str:
    """Compute and save one block's gxleg file; returns a status for the log."""
    out_path = ctx["out_dir"] / f"gxleg_{block_start:%Y%m%d_%H%M%S}.npz"
    if out_path.exists():
        if block_is_complete(out_path):
            return "exists"
        print(f"    NOTE: {out_path.name} predates the leg-wise outputs; recomputing")

    beam_src = resolve_beam_source(block_start, band, args, ctx)
    if beam_src is None:
        return "no-matching-stage2"
    full_beam_path, source = beam_src

    loaded = decimated.load_block(hourly_map, block_start, block_end)
    if loaded is None:
        return "no-data"
    block, channel, fs = loaded

    cov_result = stage2.block_covariance(block, fs, band)
    if cov_result is None:
        return "too-few-snapshots"
    Rk_avg, freq, n_valid, n_expected = cov_result

    if args.is_coherence_norm:
        Rk_avg = stage2.normalize_to_coherence(Rk_avg)

    Rk_use = Rk_avg[::args.f_bin_skip, :, :]
    freq_kept = freq[::args.f_bin_skip]

    # Same as ctx["freq_kept"] by construction; if not, a reused beam would be
    # paired with gamma from different bins.
    if freq_kept.shape != ctx["freq_kept"].shape or not np.allclose(freq_kept, ctx["freq_kept"]):
        raise ValueError(
            f"{block_start}: this block's frequency grid differs from the one the "
            "band and fs imply; the steering cache and the covariance would be "
            "indexed against each other wrongly."
        )

    # --- the full-array beam: reuse step 2's, or run the same call here ----
    beam = {}

    if source == "computed-here":
        E3, B = das_beamforming.beamform_k_aoa_implicit_vec_2Dk_cached_from_cov(
            R=Rk_use,
            theta=ctx["theta_rad"],
            cache_dir=ctx["cache_dir"],
            cache_postfix=ctx["postfix"],
            eps=EPS,
            return_E3_sum=False,
        )
        beam = {"B": B, "E3": E3}

    # --- the diagnostic -----------------------------------------------------
    legs = crossleg_gamma_from_cov(
        R=Rk_use,
        theta=ctx["theta_rad"],
        cache_dir=ctx["cache_dir"],
        cache_postfix=ctx["postfix"],
        M1=args.m1,
        eps=EPS,
        dtype=np.dtype(args.gamma_dtype),
    )

    savez_atomic(
        out_path,
        **legs,
        M1=int(args.m1),
        n_leg1_channels=int(args.m1),
        n_leg2_channels=int(channel.size - args.m1),
        cov_treatment=ctx["norm_tag"],
        gamma_note=(
            f"gamma_cross = a1^H C12 a2 / sqrt((a1^H C11 a1)(a2^H C22 a2)), computed "
            f"from the POST-normalization covariance ({ctx['norm_tag']}); channel axis "
            f"split as [leg1 = channels 0:{int(args.m1)}, leg2 = {int(args.m1)}:]"
        ),
        leg_note=(
            "gamma_leg<l> = a_l^H C_ll a_l / (tr(C_ll) ||a_l||^2): leg l beamformed "
            "alone, normalized as B is. share_leg<l> = a_l^H C_ll a_l / (tr(R) ||a||^2): "
            "leg l's additive part of B; B - share_leg1 - share_leg2 = "
            "2 Re{a1^H C12 a2} / (tr(R) ||a||^2) is the cross-leg part."
        ),
        full_beam_source=source,
        full_beam_path=full_beam_path,
        band_name=band["name"],
        fmin=band["fmin"],
        fmax=band["fmax"],
        window_s=band["window_s"],
        cov_duration_h=band["cov_duration_h"],
        delta_f=1.0 / float(band["window_s"]),
        block_start=block_start.isoformat(),
        block_end=block_end.isoformat(),
        n_valid_snapshots=n_valid,
        n_expected_snapshots=n_expected,
        min_valid_snapshots=stage2.MIN_VALID_SNAPSHOTS,
        freq=freq_kept,
        f_bin_skip=args.f_bin_skip,
        theta_rad=ctx["theta_rad"],
        theta_deg=np.rad2deg(ctx["theta_rad"]),
        E_all=ctx["E_all"],
        C_all=ctx["C_all"],
        valid_pairs=ctx["valid_pairs"],
        is_coherence_norm=args.is_coherence_norm,
        steering_postfix=ctx["postfix"],
        steering_cache_dir=str(ctx["cache_dir"]),
        directional_floor=ctx["meta"]["directional_floor"],
        directional_power=ctx["meta"]["directional_power"],
        orientation_half_window=ctx["meta"]["orientation_half_window"],
        channel=channel,
        ch_xy=ctx["ch_xy"],
        **beam,
    )

    return f"saved ({n_valid}/{n_expected} snapshots, beam {source})"


# --------------------------------------------------------------------- #
# Phase 1 -- sweep
# --------------------------------------------------------------------- #

def run_phase1(band: dict, hourly_map: dict, args, ctx) -> None:
    run_start = min(hourly_map)
    run_end = max(hourly_map) + pd.Timedelta(hours=1)

    blocks = stage2.enumerate_blocks(band, run_start, run_end)
    mine = stage2.assign_shard(blocks, args.shard_index, args.num_shards)

    print(f"  {len(blocks)} blocks total, {len(mine)} on shard "
          f"{args.shard_index}/{args.num_shards}")
    print(f"  -> {ctx['out_dir']}")

    for block_start, block_end in mine:
        t0 = time.time()
        status = process_block(hourly_map, block_start, block_end, band, args, ctx)
        if status == "exists":
            continue
        print(f"  [{block_start:%Y-%m-%d %H:%M}] {status} [{time.time() - t0:.1f}s]",
              flush=True)


# --------------------------------------------------------------------- #
# Phase 2 -- frequency integration and the two maxima
# --------------------------------------------------------------------- #

def integrate_gamma(gamma: np.ndarray, f_sel: np.ndarray):
    """Uniform mean of gamma (F, curves, theta) over the finite bins in f_sel.

    Complex input stays complex (the mean is taken before Re/|.|^2/arg).
    Accumulated per bin to avoid a second full-size array. Returns
    (gamma_bar, n_f_used), the latter the bin count per cell.
    """
    n_curves, n_theta = gamma.shape[1], gamma.shape[2]
    acc_dtype = np.complex128 if np.iscomplexobj(gamma) else np.float64

    acc = np.zeros((n_curves, n_theta), dtype=acc_dtype)
    n_used = np.zeros((n_curves, n_theta), dtype=np.float64)

    for i in np.flatnonzero(f_sel):
        g = gamma[i].astype(acc_dtype, copy=False)
        ok = np.isfinite(g)
        acc[ok] += g[ok]
        n_used[ok] += 1.0

    gamma_bar = np.full((n_curves, n_theta), np.nan, dtype=acc_dtype)
    nz = n_used > 0
    gamma_bar[nz] = acc[nz] / n_used[nz]

    return gamma_bar, n_used.astype(np.int32)


def circular_sep_deg(a: float, b: float) -> float:
    """Smallest angular separation between two bearings, in degrees."""
    return float(np.abs(((a - b + 180.0) % 360.0) - 180.0))


def load_full_beam(gamma_file: Path):
    """(B, source path) for a gamma block: inline, or from the step-2 block."""
    with np.load(gamma_file) as d:
        if "B" in d.files:
            return d["B"], str(gamma_file)
        full_path = str(d["full_beam_path"])

    if not full_path:
        raise SystemExit(
            f"{gamma_file.name} holds neither B nor a stage-2 path. Re-run phase 1 "
            "for it with --reuse never so the beam is stored inline."
        )

    src = Path(full_path)
    if not src.exists():
        raise SystemExit(
            f"{gamma_file.name} points at {src} for its full-array beam and that file "
            "is gone. Restore the stage-2 run, or re-run phase 1 with --reuse never."
        )

    with np.load(src) as d:
        return d["B"], full_path


def load_array(gamma_file: Path, key: str) -> np.ndarray:
    """One array from a gamma block, so only one is held in memory at a time."""
    with np.load(gamma_file) as d:
        if key not in d.files:
            raise SystemExit(
                f"{gamma_file.name} has no {key!r}: it predates the leg-wise "
                "outputs. Re-run phase 1 (without --summary-only) and it will be "
                "recomputed in place."
            )
        return d[key]


def masked_argmax(score: np.ndarray, valid: np.ndarray):
    """(curve, theta) of the largest finite `score` over `valid`, or None."""
    score = np.where(valid & np.isfinite(score), score, -np.inf)
    if not np.isfinite(score).any():
        return None
    return np.unravel_index(int(np.argmax(score)), score.shape)


def summarize_block(gamma_file: Path, args) -> dict | None:
    """One block's summary row and per-theta / per-frequency profiles at the MAP.

    All maxima are searched over the cells where the full-beam likelihood is
    finite, so they share one candidate set. None if no such cell exists.
    """
    with np.load(gamma_file) as d:
        freq = d["freq"]
        theta_deg = d["theta_deg"]
        valid_pairs = d["valid_pairs"]
        E_all = d["E_all"]
        C_all = d["C_all"]
        block_start = str(d["block_start"])
        n_valid_snapshots = int(d["n_valid_snapshots"])
        m1 = int(d["M1"])
        cov_treatment = str(d["cov_treatment"])

    f_sel = (freq >= args.fmin) & (freq <= args.fmax)
    if not f_sel.any():
        raise SystemExit(
            f"No saved bin of {gamma_file.name} lies in [{args.fmin}, {args.fmax}] Hz; "
            f"it holds {freq.size} bins spanning {freq.min():.3f}-{freq.max():.3f} Hz."
        )

    B, beam_src = load_full_beam(gamma_file)

    LL =stage3.block_log_likelihood(B, f_sel)

    cell_valid = np.isfinite(LL)
    if not cell_valid.any():
        del B
        return None

    c_map, t_map = masked_argmax(LL, cell_valid)
    e_map, p_map = valid_pairs[c_map]

    # B at the MAP cell: the total the leg shares add up to.
    B_map_f = np.asarray(B[f_sel, c_map, t_map], dtype=np.float64)
    del B
    B_bar_map = float(np.nanmean(B_map_f)) if np.isfinite(B_map_f).any() else np.nan

    def at(idx, arr):
        return arr[idx[0], idx[1]]

    def where(idx):
        e, p = valid_pairs[idx[0]]
        return float(theta_deg[idx[1]]), float(E_all[e]), float(C_all[p])

    # --- cross-leg --------------------------------------------------------
    g_cross = load_array(gamma_file, "gamma_cross")
    gamma_bar, n_f_used = integrate_gamma(g_cross, f_sel)
    g_cross_f = g_cross[f_sel, c_map, t_map].astype(np.complex64)
    del g_cross
    g_map = gamma_bar[c_map, t_map]

    metric = (np.real(gamma_bar) if args.cross_metric == "re"
              else np.abs(gamma_bar) ** 2)
    x_idx = masked_argmax(metric, cell_valid)
    if x_idx is None:
        return None
    g_x = at(x_idx, gamma_bar)
    x_theta, x_E, x_P = where(x_idx)
    sep = circular_sep_deg(x_theta, float(theta_deg[t_map]))

    row = {
        "block_start": block_start,
        "n_valid_snapshots": n_valid_snapshots,
        "M1": m1,
        "cov_treatment": cov_treatment,
        "n_freq_used": int(f_sel.sum()),
        # Full-array MAP: the parameter estimate.
        "theta_map_deg": float(theta_deg[t_map]),
        "E_map": float(E_all[e_map]),
        "P_map": float(C_all[p_map]),
        "curve_map": int(c_map),
        # gamma_bar at the MAP: the primary diagnostic.
        "re_gamma_map": float(np.real(g_map)),
        "abs2_gamma_map": float(np.abs(g_map) ** 2),
        "arg_gamma_map_deg": float(np.rad2deg(np.angle(g_map))),
        "n_f_used_map": int(n_f_used[c_map, t_map]),
        # The cross-leg map's own maximum: secondary.
        "xmax_theta_deg": x_theta,
        "xmax_E": x_E,
        "xmax_P": x_P,
        "re_gamma_xmax": float(np.real(g_x)),
        "abs2_gamma_xmax": float(np.abs(g_x) ** 2),
        "arg_gamma_xmax_deg": float(np.rad2deg(np.angle(g_x))),
        "theta_sep_deg": sep,
        "agree_theta": bool(sep <= args.agree_theta_deg),
    }
    profiles = {"gamma_bar_theta": gamma_bar[c_map, :].astype(np.complex64),
                "gamma_cross_f": g_cross_f}
    del gamma_bar

    # --- each leg on its own ----------------------------------------------
    # gamma_leg<l> has B's form, so it is scored like B; its LL argmax is what
    # leg l alone would estimate.
    for leg in (1, 2):
        g_leg = load_array(gamma_file, f"gamma_leg{leg}")
        LL_leg = stage3.block_log_likelihood(g_leg, f_sel)
        g_leg_bar, _ = integrate_gamma(g_leg, f_sel)
        profiles[f"gamma_leg{leg}_f"] = g_leg[f_sel, c_map, t_map].astype(np.float32)
        del g_leg

        l_idx = masked_argmax(LL_leg, cell_valid)
        if l_idx is None:
            return None
        l_theta, l_E, l_P = where(l_idx)
        l_sep = circular_sep_deg(l_theta, float(theta_deg[t_map]))

        row[f"gamma_leg{leg}_map"] = float(g_leg_bar[c_map, t_map])
        row[f"leg{leg}max_theta_deg"] = l_theta
        row[f"leg{leg}max_E"] = l_E
        row[f"leg{leg}max_P"] = l_P
        row[f"gamma_leg{leg}_legmax"] = float(at(l_idx, g_leg_bar))
        row[f"leg{leg}_theta_sep_deg"] = l_sep
        row[f"leg{leg}_agree_theta"] = bool(l_sep <= args.agree_theta_deg)
        profiles[f"gamma_leg{leg}_bar_theta"] = g_leg_bar[c_map, :].astype(np.float32)
        del g_leg_bar

    # --- how B at the estimate splits between the legs --------------------
    # Window means at the MAP cell; they add up to B_bar_map, and frac_* are
    # fractions of it. share_cross can be negative.
    shares = {}
    for leg in (1, 2):
        s_f = load_array(gamma_file, f"share_leg{leg}")[f_sel, c_map, t_map]
        shares[leg] = float(np.nanmean(s_f)) if np.isfinite(s_f).any() else np.nan
    share_cross = B_bar_map - shares[1] - shares[2]

    row["B_bar_map"] = B_bar_map
    row["share_leg1_map"] = shares[1]
    row["share_leg2_map"] = shares[2]
    row["share_cross_map"] = share_cross
    denom = B_bar_map if B_bar_map > 0 else np.nan
    row["frac_leg1_map"] = shares[1] / denom
    row["frac_leg2_map"] = shares[2] / denom
    row["frac_cross_map"] = share_cross / denom

    row["beam_source"] = beam_src

    return {
        "row": row,
        "profiles": profiles,
        "theta_deg": theta_deg,
        "freq": freq[f_sel],
    }


def across_block_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, median and IQR over blocks of each STATS_COLUMNS scalar."""
    out = []
    for col in STATS_COLUMNS:
        x = df[col].to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        n = x.size
        row = {"metric": col, "n_blocks": n}
        if n > 0:
            q25, q75 = np.percentile(x, [25, 75])
            row.update(
                mean=x.mean(),
                std=x.std(ddof=1) if n > 1 else np.nan,
                median=np.median(x),
                q25=q25,
                q75=q75,
                iqr=q75 - q25,
            )
        out.append(row)

    return pd.DataFrame(out)


# Fixed series colors for the coherence plots: leg 1, leg 2, cross-leg.
SERIES_COLORS = {"leg1": "#2a78d6", "leg2": "#eb6834", "cross": "#1baf7a"}


def cross_metric_values(g: np.ndarray, cross_metric: str):
    """gamma_cross reduced to the real number the run ranks by, and its label."""
    if cross_metric == "re":
        return np.real(g), r"Re{$\gamma_\times$}"
    return np.abs(g) ** 2, r"$|\gamma_\times|^2$"


def regular_time_grid(block_start: np.ndarray):
    """Regular time grid so missing blocks show as gaps. Returns (edges as
    matplotlib date numbers, column index of each block, number of columns)."""
    t = pd.to_datetime(block_start)
    if t.size > 1:
        dt = pd.Series(t).diff().dropna().median()
    else:
        dt = pd.Timedelta(hours=1)
    col = np.rint((t - t[0]) / dt).astype(int)
    n_col = int(col.max()) + 1
    edges = pd.DatetimeIndex([t[0] + dt * k for k in range(n_col + 1)])
    return mdates.date2num(edges.to_pydatetime()), col, n_col


def freq_edges(freq: np.ndarray) -> np.ndarray:
    if freq.size == 1:
        return np.array([freq[0] - 0.5, freq[0] + 0.5])
    mid = 0.5 * (freq[1:] + freq[:-1])
    return np.concatenate([[2 * freq[0] - mid[0]], mid, [2 * freq[-1] - mid[-1]]])


def plot_coherence(block_start, freq, g1_f, g2_f, gx_f, args, out_dir: Path,
                   tag: str) -> list[Path]:
    """Per-frequency coherence at each block's MAP: a time x frequency figure
    and a median/IQR-vs-frequency figure.

    g1_f, g2_f: (n_blocks, n_freq) real; gx_f: (n_blocks, n_freq) complex,
    shown as the quantity --cross-metric selects.
    """
    gx_val, gx_label = cross_metric_values(gx_f, args.cross_metric)
    series = [
        ("leg1", r"$\gamma_1$ (leg 1)", g1_f),
        ("leg2", r"$\gamma_2$ (leg 2)", g2_f),
        ("cross", gx_label + " (cross-leg)", gx_val),
    ]
    paths = []

    # --- time x frequency --------------------------------------------------
    t_edges, col, n_col = regular_time_grid(block_start)
    f_edges = freq_edges(freq)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             constrained_layout=True)
    for ax, (key, label, val) in zip(axes, series):
        grid = np.full((freq.size, n_col), np.nan)
        grid[:, col] = val.T
        if key == "cross" and args.cross_metric == "re":
            cmap, vmin, vmax = "coolwarm", -1.0, 1.0
        else:
            cmap, vmin, vmax = "Blues", 0.0, 1.0
        cmap = matplotlib.colormaps[cmap].copy()
        cmap.set_bad("#e0e0e0")
        mesh = ax.pcolormesh(t_edges, f_edges, grid, cmap=cmap, vmin=vmin,
                             vmax=vmax, shading="flat", rasterized=True)
        fig.colorbar(mesh, ax=ax, pad=0.01, label=label)
        ax.set_ylabel("Frequency [Hz]")
    axes[-1].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    axes[-1].set_xlabel("Time (UTC)")
    axes[0].set_title(f"Coherence at the full-array MAP estimate, {tag}  "
                      f"(gray: no data)", loc="left", fontsize=10)
    for ext in ("png", "pdf"):
        path = out_dir / f"crossleg_coh_tf_{tag}.{ext}"
        fig.savefig(path, dpi=200)
        paths.append(path)
    plt.close(fig)

    # --- coherence vs frequency ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for key, label, val in series:
        with np.errstate(all="ignore"):
            q25, med, q75 = np.nanpercentile(val, [25, 50, 75], axis=0)
        color = SERIES_COLORS[key]
        ax.fill_between(freq, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(freq, med, color=color, linewidth=2, label=label)
    if args.cross_metric == "re":
        ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_ylim(top=1.0)
    ax.set_xlim(freq[0], freq[-1])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence")
    ax.set_title(f"Median over {len(block_start)} blocks, IQR shaded",
                 loc="left", fontsize=10)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        path = out_dir / f"crossleg_coh_freq_{tag}.{ext}"
        fig.savefig(path, dpi=200)
        paths.append(path)
    plt.close(fig)

    return paths


def run_phase2(args, ctx) -> None:
    files = sorted(ctx["out_dir"].glob("gxleg_*.npz"))
    if not files:
        print(f"  no gamma blocks in {ctx['out_dir']}; nothing to summarize")
        return

    print(f"\n--- summary over {window_tag(args.fmin, args.fmax)}: "
          f"{len(files)} blocks ---")

    rows, theta_deg, freq_win = [], None, None
    profiles = {"gamma_bar_theta": [], "gamma_leg1_bar_theta": [],
                "gamma_leg2_bar_theta": [], "gamma_cross_f": [],
                "gamma_leg1_f": [], "gamma_leg2_f": []}

    for path in files:
        out = summarize_block(path, args)
        if out is None:
            print(f"  SKIP {path.name}: no finite likelihood cell")
            continue

        if freq_win is None:
            freq_win = out["freq"]
        elif not np.array_equal(out["freq"], freq_win):
            raise SystemExit(
                f"{path.name} has a different frequency grid in "
                f"[{args.fmin}, {args.fmax}] Hz than the blocks before it; the "
                "per-frequency coherence cannot be stacked across blocks."
            )

        rows.append(out["row"])
        for k in profiles:
            profiles[k].append(out["profiles"][k])
        theta_deg = out["theta_deg"]

        r = out["row"]
        print(f"  [{r['block_start'][:16]}] "
              f"theta={r['theta_map_deg']:6.1f}  E={r['E_map']:.3g}  P={r['P_map']:.3g} "
              f"| Re(g)={r['re_gamma_map']:+.3f} |g|^2={r['abs2_gamma_map']:.3f} "
              f"arg={r['arg_gamma_map_deg']:+6.1f}deg "
              f"| xmax {r['xmax_theta_deg']:6.1f} (dtheta={r['theta_sep_deg']:5.1f}) "
              f"| g1={r['gamma_leg1_map']:.3f} g2={r['gamma_leg2_map']:.3f} "
              f"B=[{r['frac_leg1_map']:.0%} L1, {r['frac_leg2_map']:.0%} L2, "
              f"{r['frac_cross_map']:+.0%} X] "
              f"| leg max {r['leg1max_theta_deg']:6.1f} / {r['leg2max_theta_deg']:6.1f}",
              flush=True)

    if not rows:
        print("  every block was skipped; nothing written")
        return

    df = pd.DataFrame(rows).sort_values("block_start").reset_index(drop=True)

    tag = window_tag(args.fmin, args.fmax)
    csv_path = ctx["out_dir"] / f"crossleg_summary_{tag}.csv"
    npz_path = ctx["out_dir"] / f"crossleg_summary_{tag}.npz"

    df.to_csv(csv_path, index=False)
    savez_atomic(
        npz_path,
        theta_deg=theta_deg,
        gamma_bar_theta_at_map_curve=np.asarray(profiles["gamma_bar_theta"]),
        gamma_leg1_bar_theta_at_map_curve=np.asarray(profiles["gamma_leg1_bar_theta"]),
        gamma_leg2_bar_theta_at_map_curve=np.asarray(profiles["gamma_leg2_bar_theta"]),
        freq_window=freq_win,
        gamma_cross_f_at_map=np.asarray(profiles["gamma_cross_f"]),
        gamma_leg1_f_at_map=np.asarray(profiles["gamma_leg1_f"]),
        gamma_leg2_f_at_map=np.asarray(profiles["gamma_leg2_f"]),
        block_start=df["block_start"].to_numpy(dtype=str),
        fmin=args.fmin,
        fmax=args.fmax,
        cross_metric=args.cross_metric,
        agree_theta_deg=args.agree_theta_deg,
        M1=int(args.m1),
        cov_treatment=ctx["norm_tag"],
        steering_postfix=ctx["postfix"],
        gamma_note=(
            "gamma_bar = complex mean of gamma_cross over the finite bins of "
            "[fmin, fmax] with uniform weights; Re/|.|^2/arg are taken AFTER that "
            "average. gamma_leg<l>_bar is the same uniform mean of the real "
            "leg-wise beam. Columns of the csv carry the per-block scalars."
        ),
        **{k: df[k].to_numpy() for k in df.columns
           if k not in ("block_start", "M1", "cov_treatment")},
    )

    def med_mean_std(col, fmt="+.3f"):
        x = df[col]
        return (f"{x.median():{fmt}} (mean {x.mean():{fmt}} "
                f"+/- {x.std():{fmt.lstrip('+')}})")

    agree = df["agree_theta"].mean()
    print("\n  median (mean +/- std) over blocks")
    print(f"  Re(gamma) at the full-beamformer MAP : {med_mean_std('re_gamma_map')}")
    print(f"  |gamma|^2 at the MAP                 : {med_mean_std('abs2_gamma_map')}")
    print(f"  gamma_leg1 at the MAP                : {med_mean_std('gamma_leg1_map')}")
    print(f"  gamma_leg2 at the MAP                : {med_mean_std('gamma_leg2_map')}")
    print(f"  share of B at the MAP, leg 1         : {med_mean_std('frac_leg1_map', '+.0%')}")
    print(f"  share of B at the MAP, leg 2         : {med_mean_std('frac_leg2_map', '+.0%')}")
    print(f"  share of B at the MAP, cross         : {med_mean_std('frac_cross_map', '+.0%')}")
    print(f"\n  blocks whose cross-leg max agrees in theta  : {agree:.0%} "
          f"(within {args.agree_theta_deg:.0f} deg, metric '{args.cross_metric}')")
    print(f"  blocks whose leg-1 / leg-2 max agrees       : "
          f"{df['leg1_agree_theta'].mean():.0%} / {df['leg2_agree_theta'].mean():.0%}")

    stats = across_block_stats(df)
    stats_path = ctx["out_dir"] / f"crossleg_stats_{tag}.csv"
    stats.to_csv(stats_path, index=False)

    print("\n  across-block statistics")
    print(f"  {'metric':<20} {'n':>5} {'mean':>7} {'std':>7} {'median':>7} "
          f"{'IQR':>6} {'[q25, q75]':>16}")
    for _, s in stats.iterrows():
        if s["n_blocks"] == 0:
            print(f"  {s['metric']:<20} {0:>5}  (no finite values)")
            continue
        print(f"  {s['metric']:<20} {int(s['n_blocks']):>5} {s['mean']:>+7.3f} "
              f"{s['std']:>7.3f} {s['median']:>+7.3f} {s['iqr']:>6.3f} "
              f"[{s['q25']:+.3f}, {s['q75']:+.3f}]")

    fig_paths = plot_coherence(
        df["block_start"].to_numpy(dtype=str), freq_win,
        np.asarray(profiles["gamma_leg1_f"]), np.asarray(profiles["gamma_leg2_f"]),
        np.asarray(profiles["gamma_cross_f"]), args, ctx["out_dir"], tag,
    )

    print(f"  -> {csv_path}")
    print(f"  -> {npz_path}")
    print(f"  -> {stats_path}")
    for path in fig_paths:
        print(f"  -> {path}")


# --------------------------------------------------------------------- #
# Per-band setup (mirrors step2_bf_disp_hourly.run_band)
# --------------------------------------------------------------------- #

def build_context(band: dict, hourly_map: dict, args) -> dict:
    band_name = band["name"]

    postfix = das_beamforming.steering_postfix(
        args.directional_floor, args.directional_power, args.f_bin_skip
    )
    norm_tag = das_beamforming.normalization_tag(args.is_coherence_norm)

    print(f"\n=== {band_name}  [{postfix}, {norm_tag}] ===")

    if postfix != "omni" or not args.is_coherence_norm:
        print("  NOTE: this diagnostic is meant for --is-coherence-norm true against "
              "the omni cache.\n        A directional cache tilts both legs' beam "
              "powers as well as the cross term, and\n        a raw covariance lets "
              "one loud channel set a leg's power. Still well defined,\n        but "
              "read |gamma| as relative between blocks rather than absolute.")

    first_ds = decimated.open_decimated(hourly_map[min(hourly_map)])
    if not bool(first_ds.attrs.get("channel_is_bend_relative", False)):
        raise ValueError(
            f"{hourly_map[min(hourly_map)]} has an absolute channel axis; this sweep "
            "needs the bend-relative axis written by decimate_merge."
        )
    fs = float(first_ds.attrs["fs"])
    channel = first_ds["channel"].values

    _, meta, _ = das_beamforming.resolve_cache(
        args.results_dir, args.directional_floor, args.directional_power,
        args.f_bin_skip, band_name,
    )

    freq = spectral.bandlimited_rfft_freqs(
        fs, band["fmin"], band["fmax"], round(band["window_s"] * fs)
    )
    stage2.check_cache_matches(meta, band, fs, freq, channel.size, args.f_bin_skip)
    freq_kept = freq[::args.f_bin_skip]

    ch_xy, m1 = pisces25.load_leg_geometry(Path(meta["ch_geo_path"]), channel)
    if args.m1 is None:
        args.m1 = m1
    if not 0 < args.m1 < channel.size:
        raise SystemExit(f"--m1 {args.m1} is not a valid split of {channel.size} channels.")

    disp = steering.load_dispersion_cache(Path(meta["disp_cache_path"]))

    out_dir = crossleg_dir(args.results_dir, band_name, norm_tag, postfix, args.m1)
    out_dir.mkdir(parents=True, exist_ok=True)

    check_config(out_dir, {
        "band": band_name,
        "postfix": postfix,
        "norm_tag": norm_tag,
        "m1": int(args.m1),
        "f_bin_skip": int(args.f_bin_skip),
        "is_coherence_norm": bool(args.is_coherence_norm),
        "window_s": float(band["window_s"]),
        "cov_duration_h": float(band["cov_duration_h"]),
    })

    print(f"  leg1 = channels {channel[0]}..{channel[args.m1 - 1]} ({args.m1}), "
          f"leg2 = {channel[args.m1]}..{channel[-1]} ({channel.size - args.m1})")
    print(f"  curves {disp['valid_pairs'].shape[0]}, theta {len(disp['theta'])}, "
          f"kept bins {meta['n_freq_kept']}")

    return {
        "out_dir": out_dir,
        "cache_dir": das_beamforming.steering_cache_dir(args.results_dir, band_name),
        "freq_kept": freq_kept,
        "postfix": postfix,
        "norm_tag": norm_tag,
        "theta_rad": np.asarray(disp["theta"], dtype=float),
        "ch_xy": ch_xy,
        "E_all": disp["E_all"],
        "C_all": disp["C_all"],
        "valid_pairs": disp["valid_pairs"],
        "meta": meta,
    }


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def make_parser() -> argparse.ArgumentParser:
    shard_index, num_shards = stage2.slurm_shard_defaults()

    parser = argparse.ArgumentParser(
        description="Cross-leg coherence diagnostic beside the stage-2 dispersion "
                    "beamformer. Raw or coherence-normalized covariance only."
    )
    parser.add_argument("--data-root", type=Path, default=stage2.DAS_DATA_PATH,
                        help="Directory of hourly 5 Hz decimated h5 files")
    parser.add_argument("--results-dir", type=Path, default=stage2.RESULTS_DIR,
                        help="The --results-dir stages 1 and 2 were given")
    parser.add_argument("--band", type=str, default=None,
                        help="Band stage 1 built, e.g. 0.10-1.00Hz; only needed when it built several")
    parser.add_argument("--m1", type=int, default=None,
                        help="Channels in leg 1; the axis is ordered [leg 1, leg 2]. "
                             "Default: the channels before the bend in the leg geometry file.")
    parser.add_argument("--f-bin-skip", type=int, default=None,
                        help="Cache to use (default: the skip stage 1 built the cache with; only needed when it built several)")
    parser.add_argument("--is-coherence-norm", type=str2bool, default=True,
                        help="Normalize the averaged covariance to a coherence matrix "
                             "before the diagnostic (default: true)")
    parser.add_argument("--directional-floor", type=float, required=True,
                        help="Names the steering cache; must match stage 1")
    parser.add_argument("--directional-power", type=float, required=True,
                        help="Names the steering cache; 0 selects the omni cache")
    parser.add_argument("--reuse", choices=("auto", "require", "never"), default="auto",
                        help="auto: reuse a matching stage-2 beam, else compute it here. "
                             "require: skip blocks with no matching stage-2 beam. "
                             "never: always beamform here and store B inline.")
    parser.add_argument("--gamma-dtype", choices=("complex64", "complex128"),
                        default="complex64",
                        help="Storage type of gamma_cross (the contractions are always "
                             "done in complex128)")
    parser.add_argument("--fmin", type=float, default=None,
                        help="Phase-2 integration window (default: the band's fmin)")
    parser.add_argument("--fmax", type=float, default=None,
                        help="Phase-2 integration window (default: the band's fmax)")
    parser.add_argument("--cross-metric", choices=("re", "abs2"), default=CROSS_METRIC,
                        help="What the cross-leg map's own maximum ranks by")
    parser.add_argument("--agree-theta-deg", type=float, default=AGREE_THETA_DEG,
                        help="Circular separation under which the two maxima agree")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip phase 1 and re-summarize existing gamma blocks")
    parser.add_argument("--no-summary", action="store_true",
                        help="Run phase 1 only")
    parser.add_argument("--num-shards", type=int, default=num_shards)
    parser.add_argument("--shard-index", type=int, default=shard_index)
    return parser


def main() -> None:
    args = make_parser().parse_args()

    # Band and f_bin_skip come from step 1's steering metadata.
    band, meta, _ = das_beamforming.resolve_cache(
        args.results_dir, args.directional_floor, args.directional_power,
        args.f_bin_skip, args.band,
    )
    args.f_bin_skip = int(meta["f_bin_skip"])

    hourly_map = decimated.index_hourly_files(args.data_root)
    if not hourly_map:
        raise FileNotFoundError(f"No hourly decimated files found in {args.data_root}")

    print(f"{len(hourly_map)} hourly files, {min(hourly_map)} .. {max(hourly_map)}")

    ctx = build_context(band, hourly_map, args)

    if args.fmin is None:
        args.fmin = float(band["fmin"])
    if args.fmax is None:
        args.fmax = float(band["fmax"])

    if not args.summary_only:
        run_phase1(band, hourly_map, args, ctx)

    if args.no_summary:
        return

    # A shard holds only a strided slice of the blocks, and every shard would
    # write the same summary file.
    if args.num_shards > 1 and not args.summary_only:
        print(f"\n  sharded run ({args.num_shards} shards): skipping the summary. "
              f"After the array finishes:\n      python {Path(__file__).name} "
              f"--directional-floor {args.directional_floor} "
              f"--directional-power {args.directional_power} --band {band['name']} "
              f"--f-bin-skip {args.f_bin_skip} "
              f"--is-coherence-norm {args.is_coherence_norm} --summary-only")
        return

    run_phase2(args, ctx)


if __name__ == "__main__":
    main()
