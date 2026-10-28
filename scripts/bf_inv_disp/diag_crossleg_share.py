"""Cross-leg share of the beamformer response, run beside step2_bf_disp_hourly.py.

A full-array beam peak in step 2 can be produced by one leg alone. With the
channel axis split as [leg 1, leg 2] at M1 and the cached steering vector split
as a = [a1; a2], the normalized beam splits exactly into three parts,

    B = share_1 + share_2 + share_x,
        share_l = a_l^H C_ll a_l / (tr(R) |a|^2),                 l = 1, 2
        share_x = 2 Re{a1^H C12 a2} / (tr(R) |a|^2) = B - share_1 - share_2

share_l is what leg l alone adds to B; share_x is the cross-leg interference,
signed (< 0 when the legs disagree under the tested hypothesis). For a single
plane wave coherent across both legs of equal length, share_x / B = 0.5.
Intended for --is-coherence-norm true with the omni cache.

M1 comes from the leg geometry file the steering cache was built with
(pisces25.load_leg_geometry): leg 1 is the channels before the bend (offset < 0).
--m1 overrides it and is part of the output directory name.

Phase 1 (per block, sharded, resumable) reads the hourly decimated files and
the step-1 steering cache, and writes (<results-dir> = outputs/bf_inv_disp)
    <results-dir>/crossleg/<band>/<norm>_<postfix>_m1<M1>/gxleg_<block_start>.npz
holding share_leg1/2 per (f, curve, theta) and the step-2 axes. The full-array
B is taken from a matching step-2 block (only its path stored) or computed here
with the same call and stored inline.

Phase 2 (re-runnable over any [fmin, fmax] and theta window) scores each block
with step3_analyze_bf_disp_posterior.block_log_likelihood on B and picks the
(E, P, theta) cell exactly as step 3's --theta-fix run does
(fixed_theta_record): the MAP curve of LL averaged over the theta bins within
--theta-half-width of --theta-fix, at that curve's best bin in the window. Every
block is evaluated there, wherever its unconstrained peak lies, so the shares
belong to the same estimate as the Fig. 9 time series. The three parts are
averaged over the frequency window at that cell and reported as fractions of B.
Writes, beside the gxleg files, crossleg_share_<theta tag>_<window>.csv/.npz and
crossleg_share_stats_<theta tag>_<window>.csv (across-block statistics).

Usage (SLURM: slurms/diag_crossleg.slurm; sharded runs skip phase 2):
    python diag_crossleg_share.py --directional-floor 1.0 --directional-power 0
    python diag_crossleg_share.py --directional-floor 1.0 --directional-power 0 \
        --summary-only --theta-fix 294 --theta-half-width 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# Steps 2 and 3 are imported, not copied, so the covariance, likelihood and
# theta window are the same code that produced the blocks compared against.
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

# The Fig. 9 conditioning: theta within THETA_HALF_WIDTH of THETA_FIX [deg].
THETA_FIX = 294.0
THETA_HALF_WIDTH = 5.0

# Per-block scalars the across-block statistics are taken over.
STATS_COLUMNS = ("frac_leg1", "frac_leg2", "frac_cross")


# --------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------- #

def crossleg_dir(results_dir: Path, band_name: str, norm_tag: str,
                 postfix: str, m1: int) -> Path:
    """<results>/crossleg/<band>/<norm>_<postfix>_m1<M1>/.

    Every setting that changes the shares is in the name, since phase 1 skips
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
            f"{out_dir} already holds blocks computed with a different config "
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
# The leg-wise shares
# --------------------------------------------------------------------- #

def leg_shares_from_cov(R, theta, cache_dir, cache_postfix, M1, eps=EPS):
    """share_leg1/2 per (f, curve, theta).

    Follows das_beamforming.beamform_k_aoa_implicit_vec_2Dk_cached_from_cov bin
    for bin (same cache, A_f = A.conj(), same normalization tr(R) |a|^2, same
    scatter through valid_mask), so both outputs are aligned element-wise with
    that function's B and B - share_leg1 - share_leg2 is the cross-leg part.

    R : (F, M, M) covariance per kept bin, after normalization.
    M1 : channels in leg 1; the channel axis is ordered [leg 1, leg 2].

    Returns a dict of (F, K_len, A_len) float32 arrays, NaN where the cache marks
    the (f, curve) cell invalid, i.e. where B is NaN.
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
    shape = (int(F), K_len, A_len)

    out = {key: np.full(shape, np.nan, dtype=np.float32) for key in SHARE_KEYS}

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
        p1 = np.real(np.sum(a1.conj() * (Rf[:M1, :M1] @ a1), axis=0))   # >= 0
        p2 = np.real(np.sum(a2.conj() * (Rf[M1:, M1:] @ a2), axis=0))   # >= 0

        a_norm2 = np.sum(np.abs(A_f) ** 2, axis=0)
        denom = np.maximum(np.real(np.trace(Rf)) * a_norm2, eps)

        for key, val in (("share_leg1", p1 / denom), ("share_leg2", p2 / denom)):
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
# Files from the earlier coherence version also hold gamma_* arrays; they are
# complete here and the extra arrays are ignored.
SHARE_KEYS = ("share_leg1", "share_leg2")


def block_is_complete(path: Path) -> bool:
    """True if `path` holds every SHARE_KEYS array. Reads only the zip directory."""
    try:
        with np.load(path) as d:
            return all(k in d.files for k in SHARE_KEYS)
    except Exception:
        return False


def process_block(hourly_map, block_start, block_end, band, args, ctx) -> str:
    """Compute and save one block's gxleg file; returns a status for the log."""
    out_path = ctx["out_dir"] / f"gxleg_{block_start:%Y%m%d_%H%M%S}.npz"
    if out_path.exists():
        if block_is_complete(out_path):
            return "exists"
        print(f"    NOTE: {out_path.name} is incomplete; recomputing")

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
    # paired with shares from different bins.
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

    # --- the leg-wise shares --------------------------------------------------
    shares = leg_shares_from_cov(
        R=Rk_use,
        theta=ctx["theta_rad"],
        cache_dir=ctx["cache_dir"],
        cache_postfix=ctx["postfix"],
        M1=args.m1,
        eps=EPS,
    )

    savez_atomic(
        out_path,
        **shares,
        M1=int(args.m1),
        n_leg1_channels=int(args.m1),
        n_leg2_channels=int(channel.size - args.m1),
        cov_treatment=ctx["norm_tag"],
        share_note=(
            f"share_leg<l> = a_l^H C_ll a_l / (tr(R) ||a||^2), computed from the "
            f"POST-normalization covariance ({ctx['norm_tag']}): leg l's additive part "
            f"of B. B - share_leg1 - share_leg2 = 2 Re{{a1^H C12 a2}} / (tr(R) ||a||^2) "
            f"is the cross-leg part. Channel axis split as [leg1 = channels "
            f"0:{int(args.m1)}, leg2 = {int(args.m1)}:]"
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
# Phase 2 -- the shares at the theta-window estimate
# --------------------------------------------------------------------- #

def window_map_cell(LL: np.ndarray, theta_deg: np.ndarray, theta_fix: float,
                    half_width: float):
    """(curve, theta index, theta_mass) of the theta-window estimate, or None.

    The same cell as step3.fixed_theta_record: the MAP curve of
    LL_W(curve) = log (1/|W|) sum_{theta in W} exp LL(curve, theta), at the bin
    of W that curve scores best. theta_mass is the share of p(theta) inside W.
    """
    bins = stage3.theta_window_bins(theta_deg, theta_fix, half_width)
    safe = np.where(np.isfinite(LL), LL, -np.inf)

    ll_w = logsumexp(safe[:, bins], axis=1) - np.log(bins.size)
    if not np.isfinite(ll_w).any():
        return None

    c_map = int(np.argmax(ll_w))
    t_map = int(bins[np.argmax(safe[c_map, bins])])
    theta_mass = float(np.exp(logsumexp(safe[:, bins]) - logsumexp(safe)))
    return c_map, t_map, theta_mass


def load_full_beam(share_file: Path):
    """(B, source path) for a share block: inline, or from the step-2 block."""
    with np.load(share_file) as d:
        if "B" in d.files:
            return d["B"], str(share_file)
        full_path = str(d["full_beam_path"])

    if not full_path:
        raise SystemExit(
            f"{share_file.name} holds neither B nor a stage-2 path. Re-run phase 1 "
            "for it with --reuse never so the beam is stored inline."
        )

    src = Path(full_path)
    if not src.exists():
        raise SystemExit(
            f"{share_file.name} points at {src} for its full-array beam and that file "
            "is gone. Restore the stage-2 run, or re-run phase 1 with --reuse never."
        )

    with np.load(src) as d:
        return d["B"], full_path


def summarize_block(share_file: Path, args) -> dict | None:
    """One block's row: B and its three parts, frequency-averaged at the
    theta-window estimate, and their fractions of B. None if the block has no
    finite likelihood cell."""
    with np.load(share_file) as d:
        freq = d["freq"]
        theta_deg = d["theta_deg"]
        valid_pairs = d["valid_pairs"]
        E_all = d["E_all"]
        C_all = d["C_all"]
        block_start = str(d["block_start"])
        n_valid_snapshots = int(d["n_valid_snapshots"])

    f_sel = (freq >= args.fmin) & (freq <= args.fmax)
    if not f_sel.any():
        raise SystemExit(
            f"No saved bin of {share_file.name} lies in [{args.fmin}, {args.fmax}] Hz; "
            f"it holds {freq.size} bins spanning {freq.min():.3f}-{freq.max():.3f} Hz."
        )

    B, beam_src = load_full_beam(share_file)
    LL = stage3.block_log_likelihood(B, f_sel)

    cell = window_map_cell(LL, theta_deg, args.theta_fix, args.theta_half_width)
    if cell is None:
        return None
    c_map, t_map, theta_mass = cell
    e_map, p_map = valid_pairs[c_map]

    # The unconstrained peak, only to record whether it fell inside the window.
    safe = np.where(np.isfinite(LL), LL, -np.inf)
    t_global = int(np.unravel_index(int(np.argmax(safe)), safe.shape)[1])
    bins = stage3.theta_window_bins(theta_deg, args.theta_fix, args.theta_half_width)

    B_f = np.asarray(B[f_sel, c_map, t_map], dtype=np.float64)
    del B

    # Window means at the cell; they add up to B_bar, and frac_* are fractions
    # of it. share_cross can be negative.
    def wmean(x):
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    B_bar = wmean(B_f)
    with np.load(share_file) as d:
        share1 = wmean(d["share_leg1"][f_sel, c_map, t_map])
        share2 = wmean(d["share_leg2"][f_sel, c_map, t_map])
    share_cross = B_bar - share1 - share2
    denom = B_bar if B_bar > 0 else np.nan

    return {
        "block_start": block_start,
        "n_valid_snapshots": n_valid_snapshots,
        "n_freq_used": int(f_sel.sum()),
        "theta_deg": float(theta_deg[t_map]),
        "E": float(E_all[e_map]),
        "P": float(C_all[p_map]),
        "theta_mass": theta_mass,
        "theta_global_map_deg": float(theta_deg[t_global]),
        "global_map_in_window": bool(t_global in bins),
        "B_bar": B_bar,
        "share_leg1": share1,
        "share_leg2": share2,
        "share_cross": share_cross,
        "frac_leg1": share1 / denom,
        "frac_leg2": share2 / denom,
        "frac_cross": share_cross / denom,
        "beam_source": beam_src,
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


def run_phase2(args, ctx) -> None:
    files = sorted(ctx["out_dir"].glob("gxleg_*.npz"))
    if not files:
        print(f"  no share blocks in {ctx['out_dir']}; nothing to summarize")
        return

    theta_tag = stage3.theta_fix_tag(args.theta_fix, args.theta_half_width)
    tag = f"{theta_tag}_{window_tag(args.fmin, args.fmax)}"
    print(f"\n--- summary over {tag}: {len(files)} blocks ---")

    # Blocks are independent and each mostly decompresses one B, so they are
    # spread over --workers processes; results come back in file order.
    workers = max(1, min(args.workers, len(files)))
    print(f"  {workers} worker process(es)")

    if workers == 1:
        results = (summarize_block(path, args) for path in files)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(summarize_block, files, repeat(args))

    rows = []
    for path, row in zip(files, results):
        if row is None:
            print(f"  SKIP {path.name}: no finite likelihood cell")
            continue
        rows.append(row)
        print(f"  [{row['block_start'][:16]}] "
              f"theta={row['theta_deg']:6.1f}  E={row['E']:.3g}  P={row['P']:.3g} "
              f"(global peak {row['theta_global_map_deg']:5.1f}) "
              f"| B=[{row['frac_leg1']:.0%} L1, {row['frac_leg2']:.0%} L2, "
              f"{row['frac_cross']:+.0%} X]", flush=True)

    if workers > 1:
        pool.shutdown()

    if not rows:
        print("  every block was skipped; nothing written")
        return

    df = pd.DataFrame(rows).sort_values("block_start").reset_index(drop=True)

    csv_path = ctx["out_dir"] / f"crossleg_share_{tag}.csv"
    npz_path = ctx["out_dir"] / f"crossleg_share_{tag}.npz"
    stats_path = ctx["out_dir"] / f"crossleg_share_stats_{tag}.csv"

    df.to_csv(csv_path, index=False)
    savez_atomic(
        npz_path,
        block_start=df["block_start"].to_numpy(dtype=str),
        beam_source=df["beam_source"].to_numpy(dtype=str),
        fmin=args.fmin,
        fmax=args.fmax,
        theta_fix_deg=args.theta_fix % 360.0,
        theta_half_width_deg=args.theta_half_width,
        M1=int(args.m1),
        cov_treatment=ctx["norm_tag"],
        steering_postfix=ctx["postfix"],
        share_note=(
            "B_bar and share_* are uniform means over the finite bins of [fmin, fmax] "
            "at the theta-window estimate (step3.fixed_theta_record's cell); "
            "share_cross = B_bar - share_leg1 - share_leg2 and frac_* = share_* / B_bar."
        ),
        **{k: df[k].to_numpy() for k in df.columns
           if k not in ("block_start", "beam_source")},
    )

    stats = across_block_stats(df)
    stats.to_csv(stats_path, index=False)

    print(f"\n  {len(df)} blocks, evaluated at theta = {args.theta_fix % 360:g} "
          f"+- {args.theta_half_width:g} deg; {int(df['global_map_in_window'].sum())} "
          f"have their unconstrained peak inside the window")
    print(f"  {'fraction of B':<16} {'n':>5} {'median':>7} {'[q25, q75]':>16} "
          f"{'mean':>7} {'std':>6}")
    for _, s in stats.iterrows():
        if s["n_blocks"] == 0:
            print(f"  {s['metric']:<16} {0:>5}  (no finite values)")
            continue
        print(f"  {s['metric']:<16} {int(s['n_blocks']):>5} {s['median']:>+7.3f} "
              f"[{s['q25']:+.3f}, {s['q75']:+.3f}] {s['mean']:>+7.3f} {s['std']:>6.3f}")

    print(f"  -> {csv_path}")
    print(f"  -> {npz_path}")
    print(f"  -> {stats_path}")


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
        description="Leg-wise and cross-leg shares of the stage-2 dispersion beam. "
                    "Raw or coherence-normalized covariance only."
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
                             "before the decomposition (default: true)")
    parser.add_argument("--directional-floor", type=float, required=True,
                        help="Names the steering cache; must match stage 1")
    parser.add_argument("--directional-power", type=float, required=True,
                        help="Names the steering cache; 0 selects the omni cache")
    parser.add_argument("--reuse", choices=("auto", "require", "never"), default="auto",
                        help="auto: reuse a matching stage-2 beam, else compute it here. "
                             "require: skip blocks with no matching stage-2 beam. "
                             "never: always beamform here and store B inline.")
    parser.add_argument("--fmin", type=float, default=None,
                        help="Phase-2 integration window (default: the band's fmin)")
    parser.add_argument("--fmax", type=float, default=None,
                        help="Phase-2 integration window (default: the band's fmax)")
    parser.add_argument("--theta-fix", type=float, default=THETA_FIX,
                        help="Phase 2: centre of the theta window the estimate is "
                             "conditioned on [deg, 0 = north, CW], as step 3's --theta-fix "
                             f"(default: {THETA_FIX:g})")
    parser.add_argument("--theta-half-width", type=float, default=THETA_HALF_WIDTH,
                        help="Phase 2: half-width of that window [deg]; 0 = the single "
                             f"nearest theta bin (default: {THETA_HALF_WIDTH:g})")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                        help="Phase 2: processes the blocks are spread over; each holds "
                             "one block's B at a time (default: $SLURM_CPUS_PER_TASK, else 1)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip phase 1 and re-summarize existing share blocks")
    parser.add_argument("--no-summary", action="store_true",
                        help="Run phase 1 only")
    parser.add_argument("--num-shards", type=int, default=num_shards)
    parser.add_argument("--shard-index", type=int, default=shard_index)
    return parser


def main() -> None:
    args = make_parser().parse_args()

    if args.theta_half_width < 0:
        raise SystemExit(f"--theta-half-width must be >= 0, got {args.theta_half_width}")

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
              f"--is-coherence-norm {args.is_coherence_norm} --summary-only "
              f"--theta-fix {args.theta_fix:g} --theta-half-width {args.theta_half_width:g}")
        return

    run_phase2(args, ctx)


if __name__ == "__main__":
    main()
