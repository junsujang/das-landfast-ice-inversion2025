"""Step 3: per-block (E, P, theta) posteriors from the step2_bf_disp_hourly.py blocks.

Each step-2 block holds B(f, curve, theta) at the exact k(f; E, P) of every
scored dispersion curve, so scoring is a sum over a frequency window
[--fmin, --fmax], a softmax and some marginals; the same blocks can be re-scored
over any sub-window without re-running step 2. Phase 1 writes one small summary
per block (resumable, shardable; see slurms/); phase 2 combines them into time
series, a record-wide p(E, P) and figures.

--ll-scale bandwidth multiplies LL by delta_f = 1/window_s. That is a pure
temperature: MAPs and peaks are unchanged, only credible widths differ.

<results> is --results-dir (default outputs/bf_inv_disp via proj_path).
Reads  <results>/<band>/<raw|coh>/bfdisp_*_<postfix>.npz, selected by the
        directional pair and --f-bin-skip (band and skip default to step 1's
        steering_meta_<postfix>.json via das_beamforming.resolve_cache).
Writes  <results>/posterior/<postfix>_<norm>_f<fmin>-<fmax>Hz_ll<scale>/
            config.json
            blocks/post_<block_start>.npz         per-block summary
            posterior_timeseries.{csv,npz}        one row per block (+ p(theta))
            posterior_stacked_EP.npz, fig_stacked_EP.png   one p(E, P) for the record
            fig_map_timeseries.png                joint MAP of E, P, theta vs time
            fig_theta_{peaks,posterior}{,_log,_<overlay>}.png   time vs p(theta)
            examples/fig_example_<t>.png          f vs theta for a few blocks
            theta_fix/<tag>/                      with --theta-fix (fixed_theta_record)

Usage:
    python step3_analyze_bf_disp_posterior.py --directional-floor 1.0 --directional-power 0
    python step3_analyze_bf_disp_posterior.py --directional-floor 1.0 --directional-power 0 \
        --fmin 0.10 --fmax 0.40 --theta-fix 322 --theta-half-width 10
    # redraw only, from existing summaries: add --plots-only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import logsumexp

from das_tools.beamforming import das_beamforming
from das_tools.utils import proj_path
from das_tools.utils.helpers import savez_atomic, str2bool

# --------------------------------------------------------------------- #
# Config -- defaults, overridable on the command line. Paths come from
# das_tools.utils.proj_path (DAS_LFI_* environment variables).
# --------------------------------------------------------------------- #

RESULTS_DIR = proj_path.output_dir("bf_inv_disp")

TOP_N = 1
N_EXAMPLES = 4

EPS = 1e-12


# --------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------- #

def run_tag(postfix: str, norm_tag: str) -> str:
    """Which step-2 run these posteriors came from: steering cache + covariance."""
    return f"{postfix}_{norm_tag}"


def window_tag(fmin: float, fmax: float, ll_scale: str) -> str:
    """Directory tag for a scoring run: window and ll-scale.

    Both are in the name because phase 1 skips existing summaries, so two
    settings sharing a directory would silently reuse each other's numbers.
    """
    return f"f{fmin:.2f}-{fmax:.2f}Hz_ll{ll_scale}"


def out_dir_for(results_dir: Path, postfix: str, norm_tag: str, fmin: float,
                fmax: float, ll_scale: str) -> Path:
    return (Path(results_dir) / "posterior" /
            f"{run_tag(postfix, norm_tag)}_{window_tag(fmin, fmax, ll_scale)}")


def theta_fix_tag(theta_fix: float, half_width: float) -> str:
    """Identity of a fixed-direction run: 'theta322.0_hw0', 'theta045.0_hw7.5'."""
    return f"theta{theta_fix % 360.0:05.1f}_hw{half_width:g}"


def theta_fix_dir(out_dir: Path, theta_fix: float, half_width: float) -> Path:
    return Path(out_dir) / "theta_fix" / theta_fix_tag(theta_fix, half_width)


def find_block_dir(results_dir: Path, band: str, norm_tag: str, postfix: str) -> Path:
    """The step-2 block directory, <results>/<band>/<norm_tag>/.

    For a raw run whose tagged directory is empty, falls back to <results>/<band>/.
    """
    tagged = Path(results_dir) / band.replace(" ", "_") / norm_tag
    if sorted(tagged.glob(f"bfdisp_*_{postfix}.npz")):
        return tagged

    flat = Path(results_dir) / band.replace(" ", "_")
    if norm_tag == "raw" and sorted(flat.glob(f"bfdisp_*_{postfix}.npz")):
        print(f"NOTE: reading the pre-subdirectory layout at {flat}. New stage-2 runs "
              f"write to {tagged}; move them with\n"
              f"      mkdir -p {tagged} && mv {flat}/bfdisp_*.npz {tagged}/")
        return flat

    return tagged


def check_block_settings(path: Path, is_coherence_norm: bool,
                         f_bin_skip: int) -> None:
    """Exit if the block's recorded f_bin_skip differs from this run's.

    Read from the block itself, not inferred from its path or name.
    """
    with np.load(path) as d:
        got_skip = int(d["f_bin_skip"]) if "f_bin_skip" in d.files else None

    if got_skip is not None and got_skip != int(f_bin_skip):
        raise SystemExit(
            f"{path.name} was beamformed with f_bin_skip={got_skip}, but this run "
            f"asked for {f_bin_skip}. Its frequency axis is every {got_skip}th bin, "
            "so the window this run scores would not be the window it names. Pass "
            f"--f-bin-skip {got_skip}, or rerun stage 2 at {f_bin_skip}."
        )


def check_config(out_dir: Path, cfg: dict) -> None:
    """Exit if out_dir/config.json records a different scoring config.

    Catches what the directory name does not (band, cache, top_n, ...), since
    phase 1 would otherwise reuse the existing summaries under the new config.
    """
    path = out_dir / "config.json"
    if not path.exists():
        return

    old = json.loads(path.read_text())
    differs = {k: (old.get(k), cfg[k]) for k in ("band", "postfix", "norm_tag",
                                                 "f_bin_skip", "fmin", "fmax",
                                                 "ll_scale", "top_n")
               if k in old and old[k] != cfg[k]}
    if differs:
        detail = ", ".join(f"{k}: {was!r} -> {now!r}" for k, (was, now) in differs.items())
        raise SystemExit(
            f"{out_dir} already holds summaries scored with a different config ({detail}).\n"
            "Those per-block files would be reused as-is and silently reported under the "
            "new config. Delete the directory to rescore, or pass --out-dir to keep both."
        )


# --------------------------------------------------------------------- #
# Posterior helpers
# --------------------------------------------------------------------- #

def scatter_valid(valid_pairs: np.ndarray, n_E: int, n_P: int) -> np.ndarray:
    """The (nE, nP) boolean mask corresponding to a list of scored (e, p) pairs."""
    mask = np.zeros((n_E, n_P), dtype=bool)
    mask[valid_pairs[:, 0], valid_pairs[:, 1]] = True
    return mask


def scatter_curve_to_grid(values: np.ndarray, valid_pairs: np.ndarray,
                          n_E: int, n_P: int, fill: float = np.nan) -> np.ndarray:
    """A per-curve quantity placed on the (nE, nP) grid, `fill` where unscored."""
    grid = np.full((n_E, n_P), fill, dtype=np.float64)
    grid[valid_pairs[:, 0], valid_pairs[:, 1]] = values
    return grid


def ll_marginal_over_theta(LL: np.ndarray) -> np.ndarray:
    """LL_EP(curve) = log (1/nTheta) sum_theta exp LL(curve, theta).

    theta integrated out under a uniform prior (not conditioned, as in
    peak_records, nor maximized, as in the joint MAP). Kept in the log domain so
    it can be summed across blocks, which the per-block normalized marginals
    cannot (see stack_theta_marginal_LL).
    """
    n_theta = LL.shape[1]
    safe = np.where(np.isfinite(LL), LL, -np.inf)
    return logsumexp(safe, axis=1) - np.log(n_theta)


def posterior_from_LL(LL: np.ndarray) -> np.ndarray | None:
    """softmax(LL) over finite cells only, or None if nothing is finite."""
    LL = np.asarray(LL, dtype=float)
    finite = np.isfinite(LL)
    if not np.any(finite):
        return None

    m = np.max(LL[finite])
    post = np.where(finite, np.exp(LL - m), 0.0)
    total = post.sum()
    if total <= 0:
        return None
    return post / total


def stats_from_1d_dist(x_vals: np.ndarray, dist: np.ndarray) -> dict:
    """Mode, mean and 5/50/95% percentiles of a (not necessarily normalized) 1D distribution."""
    total = dist.sum()
    if total <= 0:
        return {k: np.nan for k in ("mode", "mean", "p05", "p50", "p95")}
    dist = dist / total

    mode = float(x_vals[np.argmax(dist)])
    mean = float(np.sum(x_vals * dist))

    cdf = np.cumsum(dist)
    p05, p50, p95 = (float(x_vals[np.searchsorted(cdf, q)]) for q in (0.05, 0.50, 0.95))

    return {"mode": mode, "mean": mean, "p05": p05, "p50": p50, "p95": p95}


def circular_top_peaks(p_theta: np.ndarray, theta_deg: np.ndarray, n: int):
    """The n highest peaks of a circular p(theta), strongest first.

    Tiled three times so a peak straddling 0/360 is found; ranked by height, not
    prominence. Returns (angles_deg, heights), each length n, NaN-padded when
    there are fewer than n peaks.
    """
    n_theta = p_theta.size
    tiled = np.concatenate([p_theta, p_theta, p_theta])

    idx, props = find_peaks(tiled)
    keep = (idx >= n_theta) & (idx < 2 * n_theta)
    idx = idx[keep] - n_theta

    if idx.size == 0:
        # No interior maximum (flat p(theta)): fall back on the argmax.
        idx = np.array([int(np.argmax(p_theta))])

    order = np.argsort(p_theta[idx])[::-1]
    idx = idx[order][:n]

    angles = np.full(n, np.nan)
    heights = np.full(n, np.nan)
    angles[:idx.size] = theta_deg[idx]
    heights[:idx.size] = p_theta[idx]

    return angles, heights


# --------------------------------------------------------------------- #
# Phase 1 -- one block
# --------------------------------------------------------------------- #

def block_log_likelihood(B: np.ndarray, f_sel: np.ndarray) -> np.ndarray:
    """LL(curve, theta) = sum_{f in window} log(B(f, curve, theta) + eps).

    With a uniform prior the posterior is softmax(LL); curves step 2 never scored
    are absent from B and so carry no mass. Accumulated per frequency to avoid a
    temporary the size of B.
    """
    n_curves, n_theta = B.shape[1], B.shape[2]
    LL = np.zeros((n_curves, n_theta), dtype=np.float64)

    for i in np.flatnonzero(f_sel):
        LL += np.log(B[i] + EPS)

    return LL


def curve_weighted_f_theta(B: np.ndarray, f_sel: np.ndarray, w_curve: np.ndarray) -> np.ndarray:
    """Bbar(f, theta) = sum_curve p(curve) B(f, curve, theta), shape (nFreqUsed, nTheta).

    (E, P) marginalized under this block's posterior, not a plain mean over the grid.
    """
    idx = np.flatnonzero(f_sel)
    out = np.empty((idx.size, B.shape[2]), dtype=np.float64)

    for j, i in enumerate(idx):
        out[j] = w_curve @ B[i]

    return out


def block_power_db(data, B: np.ndarray, f_sel: np.ndarray) -> float:
    """Band-summed total power of the block, 10*log10(sum_f tr(R_f)) [dB].

    Recovered from step 2's E3 = M * (log(1 - B) + log(tr(R_f))), M channels:
    tr(R_f) = exp(E3 / M - log(1 - B)) at any steering cell, taken at the cell
    with the smallest B to stay clear of B -> 1. An absolute level (after any
    covariance normalization step 2 applied; little meaning under
    --is-coherence-norm), unlike the relative peak_bf_db. NaN if E3 is missing or
    no cell is usable.
    """
    try:
        E3 = data["E3"]
        n_chan = int(np.asarray(data["channel"]).size)
    except Exception:
        return float("nan")

    if n_chan <= 0:
        return float("nan")

    idx = np.flatnonzero(f_sel)
    total = 0.0

    for i in idx:
        b_i = B[i]
        e_i = E3[i]
        ok = np.isfinite(b_i) & np.isfinite(e_i) & (b_i > 0.0) & (b_i < 1.0)
        if not ok.any():
            continue

        flat = np.where(ok, b_i, np.inf).ravel()
        j = int(np.argmin(flat))
        b_j = float(flat[j])
        e_j = float(e_i.ravel()[j])

        total += float(np.exp(e_j / n_chan - np.log1p(-b_j)))

    del E3
    return 10.0 * np.log10(total) if total > 0 else float("nan")


def peak_records(post, LL, B, f_sel, freq, theta_deg, peak_deg,
                 valid_pairs, E_all, C_all) -> dict:
    """Per-peak fields, each conditioned on that peak's single theta bin.

    (E, P) MAP and marginals, LL over curves (used by
    step4_stitch_bf_disp_series.py), peak_bf_db = 10*log10(sum_f B) at the MAP
    curve (relative to a perfectly matched beam, so <= 0 dB) and peak_freq_hz
    where B is largest. Rows stay NaN for peaks circular_top_peaks did not find.
    """
    n_peaks = peak_deg.size
    n_curves = post.shape[0]
    f_idx = np.flatnonzero(f_sel)

    out = {
        "peak_E_map": np.full(n_peaks, np.nan),
        "peak_P_map": np.full(n_peaks, np.nan),
        "peak_bf_db": np.full(n_peaks, np.nan),
        "peak_freq_hz": np.full(n_peaks, np.nan),
        "peak_p_E": np.full((n_peaks, E_all.size), np.nan, dtype=np.float32),
        "peak_p_P": np.full((n_peaks, C_all.size), np.nan, dtype=np.float32),
        "peak_LL_curve": np.full((n_peaks, n_curves), np.nan, dtype=np.float32),
    }

    for i, deg in enumerate(peak_deg):
        if not np.isfinite(deg):
            continue

        t_i = int(np.argmin(np.abs(theta_deg - deg)))

        # Posterior over curves GIVEN this direction.
        col = post[:, t_i]
        total = col.sum()
        if total <= 0:
            continue
        p_curve_i = col / total

        p_E_i = np.zeros(E_all.size)
        p_P_i = np.zeros(C_all.size)
        np.add.at(p_E_i, valid_pairs[:, 0], p_curve_i)
        np.add.at(p_P_i, valid_pairs[:, 1], p_curve_i)

        c_i = int(np.argmax(col))
        e_i, p_i = valid_pairs[c_i]

        b_f = B[f_idx, c_i, t_i]
        out["peak_bf_db"][i] = 10.0 * np.log10(max(float(b_f.sum()), EPS))
        out["peak_freq_hz"][i] = float(freq[f_idx][int(np.argmax(b_f))])

        out["peak_E_map"][i] = float(E_all[e_i])
        out["peak_P_map"][i] = float(C_all[p_i])
        out["peak_p_E"][i] = p_E_i
        out["peak_p_P"][i] = p_P_i

        # Shifted to a zero maximum: only differences survive the softmax, and
        # it keeps float32 in range.
        ll_i = LL[:, t_i]
        out["peak_LL_curve"][i] = ll_i - np.nanmax(ll_i)

    return out


def theta_window_bins(theta_deg: np.ndarray, theta_fix: float, half_width: float) -> np.ndarray:
    """Indices of the theta bins within half_width of theta_fix (circular).

    The single nearest bin when half_width is 0 or the window catches no bin.
    """
    dist = np.abs((np.asarray(theta_deg, dtype=float) - theta_fix + 180.0) % 360.0 - 180.0)
    idx = np.flatnonzero(dist <= half_width + 1e-9) if half_width > 0 else np.array([], dtype=int)
    return idx if idx.size else np.array([int(np.argmin(dist))])


def fixed_theta_record(LL, B, f_sel, freq, theta_deg, valid_pairs, E_all, C_all,
                       theta_fix: float, half_width: float) -> dict:
    """(E, P) conditioned on theta in the window W around theta_fix, for one block.

        LL_W(curve) = log (1/|W|) sum_{theta in W} exp LL(curve, theta)
        p(E, P | W) = softmax(LL_W)

    Built from LL because the posterior underflows to zero at W when p(theta)
    peaks elsewhere. theta_mass is the share of p(theta) inside W: near 0 means
    the result says "if the wave came from W", not that it did.
    """
    bins = theta_window_bins(theta_deg, theta_fix, half_width)
    safe = np.where(np.isfinite(LL), LL, -np.inf)

    ll_w = logsumexp(safe[:, bins], axis=1) - np.log(bins.size)       # (nCurves,)
    log_total = logsumexp(safe)
    theta_mass = (float(np.exp(logsumexp(safe[:, bins]) - log_total))
                  if np.isfinite(log_total) else np.nan)

    out = dict(
        theta_fix_deg=float(theta_fix) % 360.0,
        theta_half_width_deg=float(half_width),
        theta_bins_deg=theta_deg[bins],
        theta_mass=theta_mass,
        E_all=E_all,
        C_all=C_all,
        valid_disp=scatter_valid(valid_pairs, E_all.size, C_all.size),
    )

    p_curve = posterior_from_LL(ll_w)
    if p_curve is None:
        nan = np.nan
        out.update(theta_best_deg=nan, E_map=nan, P_map=nan, bf_db=nan, freq_hz=nan,
                   p_E=np.full(E_all.size, np.nan), p_P=np.full(C_all.size, np.nan),
                   LL_curve=np.full(ll_w.size, np.nan, dtype=np.float32),
                   **{f"{q}_{s}": nan for q in ("E", "P")
                      for s in ("mode", "mean", "p05", "p50", "p95")})
        return out

    p_E = np.zeros(E_all.size)
    p_P = np.zeros(C_all.size)
    np.add.at(p_E, valid_pairs[:, 0], p_curve)
    np.add.at(p_P, valid_pairs[:, 1], p_curve)

    c_map = int(np.argmax(p_curve))
    e_map, p_map = valid_pairs[c_map]

    # Beam descriptors at the MAP curve, in the bin of W that curve likes best.
    t_best = int(bins[np.argmax(safe[c_map, bins])])
    f_idx = np.flatnonzero(f_sel)
    b_f = B[f_idx, c_map, t_best]

    stats_E = stats_from_1d_dist(E_all, p_E)
    stats_P = stats_from_1d_dist(C_all, p_P)
    out.update(
        theta_best_deg=float(theta_deg[t_best]),
        E_map=float(E_all[e_map]),
        P_map=float(C_all[p_map]),
        bf_db=10.0 * np.log10(max(float(b_f.sum()), EPS)),
        freq_hz=float(freq[f_idx][int(np.argmax(b_f))]),
        p_E=p_E,
        p_P=p_P,
        # Shifted to a zero maximum, as peak_LL_curve.
        LL_curve=(ll_w - np.max(ll_w[np.isfinite(ll_w)])).astype(np.float32),
        **{f"E_{k}": v for k, v in stats_E.items()},
        **{f"P_{k}": v for k, v in stats_P.items()},
    )
    return out


def process_block(path: Path, out_path: Path, args, keep_f_theta: bool,
                  fix_path: Path | None = None, write_base: bool = True) -> str:
    """Score one step-2 block and write its posterior summary to out_path.

    With fix_path, also write the --theta-fix summary there; write_base False
    writes that alone. Returns "saved", "corrupt" or "degenerate".
    """
    try:
        data = np.load(path)
        B = data["B"]
        freq = data["freq"]
        valid_pairs = data["valid_pairs"]
        E_all = data["E_all"]
        C_all = data["C_all"]
        theta_deg = data["theta_deg"]
        window_s = float(data["window_s"])
        block_start = str(data["block_start"])
        block_end = str(data["block_end"])
        n_valid_snapshots = int(data["n_valid_snapshots"])
    except Exception as exc:
        # Most likely an interrupted step-2 write; rerun step 2 for this block.
        print(f"  SKIP unreadable {path.name}: {exc}")
        return "corrupt"

    f_sel = (freq >= args.fmin) & (freq <= args.fmax)
    if not f_sel.any():
        raise ValueError(
            f"No saved frequency bin lies in [{args.fmin}, {args.fmax}] Hz. "
            f"{path.name} holds {freq.size} bins spanning "
            f"{freq.min():.3f}-{freq.max():.3f} Hz."
        )

    LL = block_log_likelihood(B, f_sel)
    if args.ll_scale == "bandwidth":
        LL *= 1.0 / window_s

    post = posterior_from_LL(LL)                       # (nCurves, nTheta)
    if post is None:
        print(f"  SKIP {path.name}: posterior has no finite mass")
        return "degenerate"

    if fix_path is not None:
        savez_atomic(
            fix_path,
            block_start=block_start,
            block_end=block_end,
            n_valid_snapshots=n_valid_snapshots,
            source=path.name,
            **fixed_theta_record(LL, B, f_sel, freq, theta_deg, valid_pairs, E_all, C_all,
                                 args.theta_fix, args.theta_half_width),
        )
    if not write_base:
        return "saved"

    # E and P marginals: curve mass scattered onto the grid via valid_pairs.
    p_curve = post.sum(axis=1)                          # (nCurves,)
    p_theta = post.sum(axis=0)                          # (nTheta,)

    p_E = np.zeros(E_all.size)
    p_P = np.zeros(C_all.size)
    np.add.at(p_E, valid_pairs[:, 0], p_curve)
    np.add.at(p_P, valid_pairs[:, 1], p_curve)

    LL_EP = scatter_curve_to_grid(
        ll_marginal_over_theta(LL), valid_pairs, E_all.size, C_all.size
    )

    # Joint MAP from the joint posterior: the marginal modes need not share a cell.
    c_map, t_map = np.unravel_index(int(np.argmax(post)), post.shape)
    e_map, p_map = valid_pairs[c_map]

    peak_deg, peak_height = circular_top_peaks(p_theta, theta_deg, args.top_n)

    stats_E = stats_from_1d_dist(E_all, p_E)
    stats_P = stats_from_1d_dist(C_all, p_P)

    # The block marginals mix all directions; with two arrivals, each one's
    # (E, P) is only visible conditioned on its own theta.
    peak = peak_records(
        post=post, LL=LL, B=B, f_sel=f_sel, freq=freq, theta_deg=theta_deg,
        peak_deg=peak_deg, valid_pairs=valid_pairs, E_all=E_all, C_all=C_all,
    )

    # Off by default: E3 is as large as B, so this roughly doubles peak memory.
    power_db = block_power_db(data, B, f_sel) if args.block_power else float("nan")

    out = dict(
        block_start=block_start,
        block_end=block_end,
        n_valid_snapshots=n_valid_snapshots,
        source=path.name,
        fmin=args.fmin,
        fmax=args.fmax,
        ll_scale=args.ll_scale,
        n_freq_used=int(f_sel.sum()),
        freq_used=freq[f_sel],
        theta_deg=theta_deg,
        p_theta=p_theta,
        p_E=p_E,
        p_P=p_P,
        E_all=E_all,
        C_all=C_all,
        E_map=float(E_all[e_map]),
        P_map=float(C_all[p_map]),
        theta_map_deg=float(theta_deg[t_map]),
        post_map=float(post[c_map, t_map]),
        curve_map=int(c_map),
        peak_theta_deg=peak_deg,
        peak_height=peak_height,
        block_power_db=power_db,
        # Step 4 needs this mask to scatter peak_LL_curve back onto the grid.
        valid_disp=scatter_valid(valid_pairs, E_all.size, C_all.size),
        # NaN on unscored cells, deliberately un-shifted: phase 2 sums these.
        LL_EP_theta_marg=LL_EP,
        **peak,
        E_mode=stats_E["mode"], E_mean=stats_E["mean"],
        E_p05=stats_E["p05"], E_p50=stats_E["p50"], E_p95=stats_E["p95"],
        P_mode=stats_P["mode"], P_mean=stats_P["mean"],
        P_p05=stats_P["p05"], P_p50=stats_P["p50"], P_p95=stats_P["p95"],
    )

    if keep_f_theta:
        # Example blocks only: two small (nFreqUsed, nTheta) slices instead of B.
        out["ft_marginal"] = curve_weighted_f_theta(B, f_sel, p_curve)
        out["ft_map"] = B[f_sel, c_map, :]
        out["is_example"] = True
    else:
        out["is_example"] = False

    savez_atomic(out_path, **out)
    return "saved"


# --------------------------------------------------------------------- #
# Phase 1 -- sweep
# --------------------------------------------------------------------- #

def assign_shard(items: list, shard_index: int, num_shards: int) -> list:
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards}), got {shard_index}")

    return items[shard_index::num_shards]


def slurm_shard_defaults() -> tuple[int, int]:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return 0, 1

    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    count = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if count is None:
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", task_min))
        count = task_max - task_min + 1

    return int(task_id) - task_min, int(count)


def choose_examples(paths: list[Path], n: int) -> set[Path]:
    """n blocks evenly spaced in time, first and last included (unbiased spot checks)."""
    if n <= 0 or not paths:
        return set()
    if n >= len(paths):
        return set(paths)

    idx = np.unique(np.linspace(0, len(paths) - 1, n).round().astype(int))
    return {paths[i] for i in idx}


def run_phase1(block_paths: list[Path], blocks_dir: Path, args,
               fix_blocks_dir: Path | None = None) -> None:
    examples = choose_examples(block_paths, args.n_examples)
    mine = assign_shard(block_paths, args.shard_index, args.num_shards)

    print(f"phase 1: {len(block_paths)} blocks, {len(mine)} on shard "
          f"{args.shard_index}/{args.num_shards}, {len(examples)} examples")

    n_saved = n_fix_only = n_skip = 0
    for path in mine:
        stem = path.stem[len('bfdisp_'):]
        out_path = blocks_dir / f"post_{stem}.npz"
        fix_path = fix_blocks_dir / f"fix_{stem}.npz" if fix_blocks_dir is not None else None

        need_base = not out_path.exists()
        need_fix = fix_path is not None and not fix_path.exists()
        if not (need_base or need_fix):
            n_skip += 1
            continue

        t0 = time.time()
        status = process_block(path, out_path, args, keep_f_theta=path in examples,
                               fix_path=fix_path if need_fix else None, write_base=need_base)
        if status == "saved":
            if need_base:
                n_saved += 1
            else:
                n_fix_only += 1
            what = {(True, False): "", (True, True): " (+theta-fix)",
                    (False, True): " (theta-fix only)"}[(need_base, need_fix)]
            print(f"  {path.name}{what} [{time.time() - t0:.1f}s]", flush=True)

    print(f"phase 1 done: {n_saved} scored, "
          + (f"{n_fix_only} scored for the fixed direction only, " if fix_blocks_dir else "")
          + f"{n_skip} already present")

    if args.block_power and n_skip:
        # Skipping is by file existence, so summaries scored without
        # --block-power keep a NaN power.
        print(f"  NOTE: {n_skip} summaries were left as they were and may predate "
              "--block-power. Delete them to have their total power filled in.")


# --------------------------------------------------------------------- #
# Phase 2 -- combine
# --------------------------------------------------------------------- #

def load_summaries(blocks_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Every per-block summary, as a time-ordered frame plus the array payloads."""
    rows = []
    p_theta, examples = [], []
    theta_deg = None

    for path in sorted(blocks_dir.glob("post_*.npz")):
        try:
            d = np.load(path, allow_pickle=False)
        except Exception as exc:
            print(f"  skipping unreadable {path.name}: {exc}")
            continue

        if theta_deg is None:
            theta_deg = d["theta_deg"]

        row = {
            "block_start": pd.Timestamp(str(d["block_start"])),
            "block_end": pd.Timestamp(str(d["block_end"])),
            "n_valid_snapshots": int(d["n_valid_snapshots"]),
            "n_freq_used": int(d["n_freq_used"]),
            "E_map": float(d["E_map"]),
            "P_map": float(d["P_map"]),
            "theta_map_deg": float(d["theta_map_deg"]),
            "post_map": float(d["post_map"]),
        }
        for key in ("E_mode", "E_mean", "E_p05", "E_p50", "E_p95",
                    "P_mode", "P_mean", "P_p05", "P_p50", "P_p95"):
            row[key] = float(d[key])

        row["block_power_db"] = (
            float(d["block_power_db"]) if "block_power_db" in d.files else np.nan
        )

        for i, (a, h) in enumerate(zip(d["peak_theta_deg"], d["peak_height"])):
            row[f"peak{i + 1}_theta_deg"] = float(a)
            row[f"peak{i + 1}_height"] = float(h)
            row[f"peak{i + 1}_bf_db"] = float(d["peak_bf_db"][i])
            row[f"peak{i + 1}_freq_hz"] = float(d["peak_freq_hz"][i])

        rows.append(row)
        p_theta.append(d["p_theta"])

        if bool(d["is_example"]):
            examples.append({
                "block_start": row["block_start"],
                "freq_used": d["freq_used"],
                "theta_deg": d["theta_deg"],
                "ft_marginal": d["ft_marginal"],
                "ft_map": d["ft_map"],
                "p_theta": d["p_theta"],
                "E_map": row["E_map"], "P_map": row["P_map"],
                "theta_map_deg": row["theta_map_deg"],
            })

    if not rows:
        raise FileNotFoundError(f"No per-block summaries under {blocks_dir}")

    df = pd.DataFrame(rows).sort_values("block_start").reset_index(drop=True)
    order = np.argsort([r["block_start"] for r in rows])

    payload = {
        "theta_deg": theta_deg,
        "p_theta": np.asarray(p_theta)[order],
        "examples": sorted(examples, key=lambda e: e["block_start"]),
    }
    return df, payload


# --------------------------------------------------------------------- #
# Phase 2 -- the record-wide (E, P) posterior
# --------------------------------------------------------------------- #

STACK_NPZ = "posterior_stacked_EP.npz"


def stack_theta_marginal_LL(blocks_dir: Path) -> dict | None:
    """One p(E, P) for the whole record, with frequency, theta and time integrated out.

        LL_sum(E, P)    = sum_blocks LL_EP_theta_marg(E, P)
        post_EP         = softmax(LL_sum)
        post_EP_mean_ll = softmax(LL_sum / n_blocks)

    Summing assumes independent blocks, which neighbouring blocks (same ice,
    correlated noise) are not, so post_EP is over-confident (a cell or two wide).
    post_EP_mean_ll has the same MAP with one block's worth of information, and
    LL_sum is kept for any other tempering. Only cells valid in every block are
    used. Returns the dict write_stack saves, or None if nothing can be stacked.
    """
    LL_sum = None
    n_cell = None
    E_all = C_all = None
    starts, ends, sources = [], [], []
    n_missing = 0

    for path in sorted(blocks_dir.glob("post_*.npz")):
        try:
            d = np.load(path, allow_pickle=False)
        except Exception as exc:
            print(f"  skipping unreadable {path.name}: {exc}")
            continue

        if "LL_EP_theta_marg" not in d.files:
            n_missing += 1
            continue

        ll = np.asarray(d["LL_EP_theta_marg"], dtype=np.float64)

        if LL_sum is None:
            LL_sum = np.zeros(ll.shape, dtype=np.float64)
            n_cell = np.zeros(ll.shape, dtype=np.int64)
            E_all = np.asarray(d["E_all"], dtype=float)
            C_all = np.asarray(d["C_all"], dtype=float)
        elif ll.shape != LL_sum.shape:
            raise SystemExit(
                f"{path.name} holds a {ll.shape} (E, P) grid but the blocks before "
                f"it hold {LL_sum.shape}. Stage 1 was re-run with a different grid; "
                "score the two sets into separate --out-dir."
            )

        finite = np.isfinite(ll)
        LL_sum += np.where(finite, ll, 0.0)
        n_cell += finite
        starts.append(pd.Timestamp(str(d["block_start"])))
        ends.append(pd.Timestamp(str(d["block_end"])))
        sources.append(path.name)

    if LL_sum is None:
        print("  no block carries LL_EP_theta_marg: these summaries predate the "
              "stacked posterior. Delete out_dir/blocks and rerun phase 1.")
        return None

    n_blocks = len(starts)
    if n_missing:
        print(f"  WARNING: {n_missing} of {n_blocks + n_missing} summaries predate "
              "LL_EP_theta_marg and are NOT in the stack. Delete out_dir/blocks and "
              "rerun phase 1 to include them.")

    valid = n_cell == n_blocks
    if not valid.any():
        print("  no (E, P) cell was scored in every block; nothing to stack")
        return None
    if valid.sum() < n_cell.max():
        print(f"  {int((n_cell > 0).sum() - valid.sum())} (E, P) cells were scored in "
              f"some but not all blocks and are excluded")

    def _posterior(ll_grid: np.ndarray) -> np.ndarray:
        masked = np.where(valid, ll_grid, -np.inf)
        post = np.where(valid, np.exp(masked - masked[valid].max()), 0.0)
        return post / post.sum()

    post = _posterior(LL_sum)
    post_mean_ll = _posterior(LL_sum / max(n_blocks, 1))

    e_map, p_map = np.unravel_index(
        int(np.argmax(np.where(valid, LL_sum, -np.inf))), LL_sum.shape
    )

    p_E = post.sum(axis=1)
    p_P = post.sum(axis=0)
    stats_E = stats_from_1d_dist(E_all, p_E)
    stats_P = stats_from_1d_dist(C_all, p_P)

    # Tempered summaries too: the joint sum's 5-95% is usually just the grid step.
    p_E_mean = post_mean_ll.sum(axis=1)
    p_P_mean = post_mean_ll.sum(axis=0)
    stats_E_mean = stats_from_1d_dist(E_all, p_E_mean)
    stats_P_mean = stats_from_1d_dist(C_all, p_P_mean)

    # E_mmse / P_mmse: the marginal means, equal to the joint posterior mean.
    st = dict(
        E_all=E_all,
        C_all=C_all,
        LL_sum=LL_sum,
        n_blocks_per_cell=n_cell,
        valid=valid,
        post_EP=post,
        post_EP_mean_ll=post_mean_ll,
        p_E=p_E,
        p_P=p_P,
        p_E_mean_ll=p_E_mean,
        p_P_mean_ll=p_P_mean,
        n_blocks=n_blocks,
        n_blocks_missing=n_missing,
        block_start_first=str(min(starts)),
        block_start_last=str(max(starts)),
        block_end_last=str(max(ends)),
        block_starts=np.array([str(t) for t in sorted(starts)]),
        E_map=float(E_all[e_map]),
        P_map=float(C_all[p_map]),
        E_mmse=stats_E["mean"],
        P_mmse=stats_P["mean"],
        E_p05=stats_E["p05"], E_p50=stats_E["p50"], E_p95=stats_E["p95"],
        P_p05=stats_P["p05"], P_p50=stats_P["p50"], P_p95=stats_P["p95"],
        E_mmse_mean_ll=stats_E_mean["mean"],
        P_mmse_mean_ll=stats_P_mean["mean"],
        E_p05_mean_ll=stats_E_mean["p05"], E_p50_mean_ll=stats_E_mean["p50"],
        E_p95_mean_ll=stats_E_mean["p95"],
        P_p05_mean_ll=stats_P_mean["p05"], P_p50_mean_ll=stats_P_mean["p50"],
        P_p95_mean_ll=stats_P_mean["p95"],
    )
    return st


def write_stack(out_dir: Path, st: dict, args) -> Path:
    out_path = out_dir / STACK_NPZ
    np.savez_compressed(
        out_path,
        fmin=args.fmin,
        fmax=args.fmax,
        ll_scale=args.ll_scale,
        **st,
    )

    print(f"  wrote {out_path}")
    print(f"    stacked over {st['n_blocks']} blocks, "
          f"{st['block_start_first']} .. {st['block_start_last']}")
    print(f"    MAP  : E = {st['E_map'] / 1e9:.2f} GPa, P = {st['P_map'] / 1e6:.3f} MPa")
    print(f"    MMSE : E = {st['E_mmse'] / 1e9:.2f} GPa, P = {st['P_mmse'] / 1e6:.3f} MPa")
    print(f"    5-95%: E = [{st['E_p05'] / 1e9:.2f}, {st['E_p95'] / 1e9:.2f}] GPa, "
          f"P = [{st['P_p05'] / 1e6:.3f}, {st['P_p95'] / 1e6:.3f}] MPa")
    print(f"    5-95% at mean log-likelihood (one block's worth): "
          f"E = [{st['E_p05_mean_ll'] / 1e9:.2f}, {st['E_p95_mean_ll'] / 1e9:.2f}] GPa, "
          f"P = [{st['P_p05_mean_ll'] / 1e6:.3f}, {st['P_p95_mean_ll'] / 1e6:.3f}] MPa")
    return out_path


def fig_stacked_EP(st: dict, args, out_path: Path) -> None:
    """Stacked p(E, P): post_EP (left) and post_EP_mean_ll (right), with MAP and MMSE."""
    E_gpa = st["E_all"] / 1e9
    P_mpa = st["C_all"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    panels = (
        (st["post_EP"], f"sum over {st['n_blocks']} blocks"),
        (st["post_EP_mean_ll"], "mean log-likelihood (one block's worth)"),
    )

    for ax, (field, title) in zip(axes, panels):
        shown = np.where(st["valid"], field, np.nan)
        mesh = ax.pcolormesh(E_gpa, P_mpa, shown.T, shading="nearest",
                             cmap="magma")
        fig.colorbar(mesh, ax=ax, label="p(E, P)")

        ax.plot(st["E_map"] / 1e9, st["P_map"] / 1e6, "x", color="tab:cyan",
                ms=12, mew=2.5, label="MAP")
        ax.plot(st["E_mmse"] / 1e9, st["P_mmse"] / 1e6, "+", color="tab:green",
                ms=12, mew=2.5, label="MMSE")

        ax.set_xlabel("E [GPa]")
        ax.set_ylabel("P [MPa]")
        ax.set_title(title)
        ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(f"p(E, P) -- theta and time marginalized, "
                 f"{args.fmin:.2f}-{args.fmax:.2f} Hz, ll_scale={args.ll_scale}")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------- #
# Phase 2 -- figures
# --------------------------------------------------------------------- #

def theta_density(p_theta: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    """p(theta) as a density in 1/deg (mass per bin / bin width).

    Independent of the theta grid; a uniform posterior sits at 1/360 deg^-1.
    """
    dtheta = float(np.median(np.diff(theta_deg))) if theta_deg.size > 1 else 1.0
    if not np.isfinite(dtheta) or dtheta <= 0:
        dtheta = 1.0

    mass = p_theta / np.maximum(p_theta.sum(axis=-1, keepdims=True), EPS)
    return mass / dtheta


def regular_time_grid(t: pd.Series, p: np.ndarray, cadence_min: float | None):
    """Put the blocks on a uniform time axis, NaN where no block landed.

    Otherwise pcolormesh stretches the block before a gap across it. Cadence is
    cadence_min, else the median block spacing. Returns (grid_times, values),
    values (nGrid, nTheta).
    """
    t = pd.DatetimeIndex(pd.to_datetime(t))
    if t.size == 0:
        return t, np.empty((0, p.shape[1]))

    # Timedelta arithmetic, not .asi8, whose unit (ns or us) varies across
    # pandas versions.
    if cadence_min is not None:
        step = pd.Timedelta(minutes=float(cadence_min))
    else:
        diffs = np.diff(t)
        diffs = diffs[diffs > np.timedelta64(0, "s")]
        step = pd.Timedelta(np.median(diffs)) if diffs.size else pd.Timedelta(minutes=30)

    if t.size == 1 or step <= pd.Timedelta(0):
        return t, p

    n_slot = int(round((t[-1] - t[0]) / step)) + 1
    grid = pd.DatetimeIndex(t[0] + step * np.arange(n_slot))

    out = np.full((n_slot, p.shape[1]), np.nan)
    idx = np.rint((t - t[0]) / step).astype(int)
    ok = (idx >= 0) & (idx < n_slot)
    out[idx[ok]] = p[ok]

    n_gap = int(np.isnan(out[:, 0]).sum())
    print(f"  time grid: {n_slot} slots of {step / pd.Timedelta(minutes=1):.4g} min, "
          f"{t.size} with data, {n_gap} empty")

    return grid, out


def theta_norm(dens: np.ndarray, log_scale: bool):
    """Colour norm shared by all blocks (not per column), plus its label.

    vmax is the 99.5th percentile; the log floor sits 4-6 decades below it.
    """
    finite = dens[np.isfinite(dens)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    if not log_scale:
        return matplotlib.colors.Normalize(vmin=0.0, vmax=vmax), "p(theta) [1/deg]"

    pos = finite[finite > 0]
    vmin = float(np.percentile(pos, 1.0)) if pos.size else vmax * 1e-4
    vmin = min(max(vmin, vmax * 1e-6), vmax * 1e-4)
    return matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), "p(theta) [1/deg, log]"


def theta_cmap():
    """magma with an explicit BLACK for missing blocks (NaN)."""
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("black")
    return cmap


OVERLAY_SPECS = {
    "bf_db": {
        "column": "peak{p}_bf_db",
        "label": "beam response at peak theta [dB]",
        "color": "tab:cyan",
    },
    "power_db": {
        "column": "block_power_db",
        "label": "block total power [dB]",
        "color": "tab:orange",
    },
}


def overlay_series(df: pd.DataFrame, kind: str, cadence_min: float | None):
    """The right-axis series on the mesh's uniform time grid (NaN gaps break the line).

    Returns (times, values, label, color), or None if the column is absent or all NaN.
    """
    spec = OVERLAY_SPECS[kind]
    col = spec["column"].format(p=1)

    if col not in df.columns:
        return None

    vals = df[col].to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return None

    t, gridded = regular_time_grid(df["block_start"], vals[:, None], cadence_min)
    return t, gridded[:, 0], spec["label"], spec["color"]


def draw_overlay(ax, df: pd.DataFrame, kind: str, cadence_min: float | None):
    """Add one dB series on a twinned right axis. Returns its handle, or None."""
    series = overlay_series(df, kind, cadence_min)
    if series is None:
        return None

    t, vals, label, color = series

    ax2 = ax.twinx()
    line, = ax2.plot(t, vals, lw=1.1, color=color, alpha=0.9, label=label, zorder=4)
    ax2.set_ylabel(label, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    # Keep the line in the lower third so it does not hide the mesh.
    finite = vals[np.isfinite(vals)]
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    ax2.set_ylim(lo - pad, hi + pad + 2.0 * (hi - lo))

    return line


def fig_theta_marginal(df: pd.DataFrame, payload: dict, args, out_path: Path,
                       show_peaks: bool, log_scale: bool,
                       overlay: str | None = None) -> None:
    """Time vs p(theta) density, black where no block exists.

    show_peaks draws the top-n peak tracks; log_scale shows the low-probability
    floor. overlay adds a dB series on a right axis, since p(theta) is normalized
    and says nothing about strength: "bf_db" (top peak's beam response, see
    peak_records) or "power_db" (block total power, see block_power_db).
    """
    t, dens = regular_time_grid(
        df["block_start"], theta_density(payload["p_theta"], payload["theta_deg"]),
        args.cadence_min,
    )

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)

    norm, cbar_label = theta_norm(dens, log_scale)
    mesh = ax.pcolormesh(t, payload["theta_deg"], np.ma.masked_invalid(dens).T,
                         shading="nearest", cmap=theta_cmap(), norm=norm)

    over_line = draw_overlay(ax, df, overlay, args.cadence_min) if overlay else None

    # The colorbar has to steal space from BOTH axes once there is a twin, or
    # constrained_layout shrinks only the left one and the two stop lining up.
    cbar_axes = [ax, over_line.axes] if over_line is not None else ax
    fig.colorbar(mesh, ax=cbar_axes, label=cbar_label,
                 extend="both" if log_scale else "max")

    n_peaks = sum(c.endswith("_theta_deg") and c.startswith("peak") for c in df.columns)
    handles = []
    if show_peaks:
        # Follow the summaries' peak columns, not the current --top-n.
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_peaks, 1)))
        for i in range(n_peaks):
            handles.append(ax.scatter(
                df["block_start"], df[f"peak{i + 1}_theta_deg"], s=14,
                color=colors[i], edgecolor="white", linewidth=0.3,
                label=f"peak {i + 1}", zorder=3))

    if over_line is not None:
        handles.append(over_line)
    if handles:
        ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 45))
    ax.set_ylabel("arrival direction theta [deg, 0 = north, CW]")
    ax.set_xlabel("time [UTC]  (black = no block)")

    what = f"Top-{n_peaks} peaks of p(theta) over" if show_peaks else "p(theta),"
    extra = f", {OVERLAY_SPECS[overlay]['label']} right" if over_line is not None else ""
    ax.set_title(f"{what} the theta marginal, marginalized over E and P "
                 f"-- {args.fmin:.2f}-{args.fmax:.2f} Hz"
                 f"{' [log colour]' if log_scale else ''}{extra}")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_map_timeseries(df: pd.DataFrame, args, out_path: Path) -> None:
    """Time vs the joint MAP of (E, P, theta); E and P with their 5-95% band.

    No band on theta: an interval on a circular, possibly bimodal p(theta) misleads.
    """
    t = df["block_start"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.fill_between(t, df["E_p05"] / 1e9, df["E_p95"] / 1e9,
                    color="tab:blue", alpha=0.2, label="5-95% marginal")
    ax.plot(t, df["E_map"] / 1e9, ".", color="tab:blue", ms=4, label="joint MAP")
    ax.set_ylabel("E [GPa]")
    ax.legend(loc="upper right", framealpha=0.9)

    ax = axes[1]
    ax.fill_between(t, df["P_p05"] / 1e6, df["P_p95"] / 1e6,
                    color="tab:orange", alpha=0.2, label="5-95% marginal")
    ax.plot(t, df["P_map"] / 1e6, ".", color="tab:orange", ms=4, label="joint MAP")
    ax.set_ylabel("P [MPa]")
    ax.legend(loc="upper right", framealpha=0.9)

    ax = axes[2]
    ax.plot(t, df["theta_map_deg"], ".", color="tab:green", ms=4, label="joint MAP")
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 90))
    ax.set_ylabel("theta [deg]")
    ax.set_xlabel("time [UTC]")
    ax.legend(loc="upper right", framealpha=0.9)

    axes[0].set_title(f"Joint MAP over (E, P, theta) -- {args.fmin:.2f}-{args.fmax:.2f} Hz, "
                      f"{len(df)} blocks")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_example(ex: dict, args, out_path: Path) -> None:
    """f vs theta for one block: (E, P)-marginalized and at the MAP curve, on a
    shared colour scale, plus its p(theta)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 0.6]})

    freq = ex["freq_used"]
    theta = ex["theta_deg"]
    vmin = min(ex["ft_marginal"].min(), ex["ft_map"].min())
    vmax = max(ex["ft_marginal"].max(), ex["ft_map"].max())

    for ax, arr, title in (
        (axes[0], ex["ft_marginal"], "marginalized over E, P"),
        (axes[1], ex["ft_map"], f"at MAP curve (E={ex['E_map'] / 1e9:.1f} GPa, "
                               f"P={ex['P_map'] / 1e6:.2f} MPa)"),
    ):
        mesh = ax.pcolormesh(theta, freq, arr, shading="nearest", cmap="magma",
                             vmin=vmin, vmax=vmax)
        ax.axvline(ex["theta_map_deg"], color="cyan", lw=1.0, ls="--")
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 90))
        ax.set_xlabel("theta [deg]")
        ax.set_title(title, fontsize=10)
        fig.colorbar(mesh, ax=ax, label="B")

    axes[0].set_ylabel("frequency [Hz]")

    ax = axes[2]
    ax.plot(ex["p_theta"], theta, color="tab:blue")
    ax.axhline(ex["theta_map_deg"], color="cyan", lw=1.0, ls="--", label="joint MAP")
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 90))
    ax.set_ylabel("theta [deg]")
    ax.set_xlabel("p(theta)")
    ax.set_title("theta marginal", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(f"{ex['block_start']:%Y-%m-%d %H:%M} UTC  --  "
                 f"{args.fmin:.2f}-{args.fmax:.2f} Hz", fontsize=11)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


FIXED_KEYS = ("n_valid_snapshots", "theta_mass", "theta_best_deg", "bf_db", "freq_hz",
              "E_map", "P_map", "E_mode", "E_mean", "E_p05", "E_p50", "E_p95",
              "P_mode", "P_mean", "P_p05", "P_p50", "P_p95")


def run_phase2_fixed(fix_dir: Path, args) -> None:
    """The --theta-fix summaries as one table, plus a time-series figure."""
    rows = []
    for path in sorted((fix_dir / "blocks").glob("fix_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as d:
                rows.append({"block_start": pd.Timestamp(str(d["block_start"])),
                             **{k: d[k].item() for k in FIXED_KEYS}})
        except Exception as exc:
            print(f"  skipping unreadable {path.name}: {exc}")
    if not rows:
        print(f"  no fixed-direction summaries under {fix_dir / 'blocks'}")
        return

    df = pd.DataFrame(rows).sort_values("block_start").reset_index(drop=True)
    csv_path = fix_dir / "theta_fixed_timeseries.csv"
    df.to_csv(csv_path, index=False)
    print(f"phase 2, {fix_dir.name}: {len(df)} blocks")
    print(f"  wrote {csv_path}")

    t = df["block_start"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    for ax, q, scale, unit, color in ((axes[0], "E", 1e9, "GPa", "tab:blue"),
                                      (axes[1], "P", 1e6, "MPa", "tab:orange")):
        ax.fill_between(t, df[f"{q}_p05"] / scale, df[f"{q}_p95"] / scale,
                        color=color, alpha=0.2, label="5-95% marginal")
        ax.plot(t, df[f"{q}_p50"] / scale, ".", color=color, ms=4, label="median")
        ax.set_ylabel(f"{q} [{unit}]")
        ax.legend(loc="upper right", framealpha=0.9)

    axes[2].plot(t, df["theta_mass"], ".", color="0.3", ms=3)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("p(theta) mass\nin window")
    axes[2].set_xlabel("time [UTC]")
    axes[0].set_title(f"(E, P) given theta = {args.theta_fix % 360:g} "
                      f"+- {args.theta_half_width:g} deg -- {args.fmin:.2f}-{args.fmax:.2f} Hz, "
                      f"{len(df)} blocks")

    fig_path = fix_dir / "fig_theta_fixed_timeseries.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {fig_path}")


def run_phase2(blocks_dir: Path, out_dir: Path, args) -> None:
    df, payload = load_summaries(blocks_dir)
    print(f"phase 2: {len(df)} blocks, {df['block_start'].min()} .. {df['block_start'].max()}")

    csv_path = out_dir / "posterior_timeseries.csv"
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    np.savez_compressed(
        out_dir / "posterior_timeseries.npz",
        block_start=df["block_start"].astype("int64").to_numpy(),
        theta_deg=payload["theta_deg"],
        p_theta=payload["p_theta"],
        **{c: df[c].to_numpy() for c in df.columns if c != "block_start" and c != "block_end"},
    )
    print(f"  wrote {out_dir / 'posterior_timeseries.npz'}")

    # Stack first, so it exists even if a figure below raises.
    if not args.no_stack:
        st = stack_theta_marginal_LL(blocks_dir)
        if st is not None:
            write_stack(out_dir, st, args)
            fig_stacked_EP(st, args, out_dir / "fig_stacked_EP.png")

    for show_peaks, stem in ((True, "fig_theta_peaks"), (False, "fig_theta_posterior")):
        for log_scale, suffix in ((False, ""), (True, "_log")):
            fig_theta_marginal(df, payload, args, out_dir / f"{stem}{suffix}.png",
                               show_peaks=show_peaks, log_scale=log_scale)

    # Overlaid figures are written alongside the plain ones, not instead of them.
    for kind in args.overlay:
        # Otherwise an unavailable overlay would write a plain copy under its name.
        if overlay_series(df, kind, args.cadence_min) is None:
            print(f"  overlay '{kind}': not in these summaries, no figure written"
                  + (" (rerun phase 1 with --block-power)" if kind == "power_db" else ""))
            continue

        for show_peaks, stem in ((True, "fig_theta_peaks"), (False, "fig_theta_posterior")):
            fig_theta_marginal(df, payload, args, out_dir / f"{stem}_{kind}.png",
                               show_peaks=show_peaks, log_scale=False, overlay=kind)

    fig_map_timeseries(df, args, out_dir / "fig_map_timeseries.png")

    if payload["examples"]:
        ex_dir = out_dir / "examples"
        ex_dir.mkdir(parents=True, exist_ok=True)
        for ex in payload["examples"]:
            fig_example(ex, args,
                        ex_dir / f"fig_example_{ex['block_start']:%Y%m%d_%H%M%S}.png")
    else:
        print("  no example blocks stored (phase 1 ran with --n-examples 0?)")


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def make_parser() -> argparse.ArgumentParser:
    shard_index, num_shards = slurm_shard_defaults()

    parser = argparse.ArgumentParser(
        description="Posteriors and diagnostics from step2_bf_disp_hourly.py blocks (stage 3)."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Same --results-dir stages 1 and 2 were given")
    parser.add_argument("--band", type=str, default=None,
                        help="Band subdirectory holding the bfdisp_*.npz blocks, e.g. "
                             "0.10-1.00Hz (default: the band stage 1 built for this cache)")
    parser.add_argument("--directional-floor", type=float, required=True,
                        help="Selects which stage-2 outputs to read; must match stage 2")
    parser.add_argument("--directional-power", type=float, required=True,
                        help="Selects which stage-2 outputs to read; 0 means the omni run")
    parser.add_argument("--f-bin-skip", type=int, default=None,
                        help="The stage-2 frequency decimation to score. Part of the "
                             "cache postfix, so it selects blocks the same way the "
                             "directional pair does; nothing is re-decimated here "
                             "(default: the skip stage 1 built the cache with; only needed when it built several)")
    # Selects WHICH step-2 run to score; no normalization is applied here.
    parser.add_argument("--is-coherence-norm", type=str2bool, default=False,
                        help="Score the stage-2 run made with coherence normalization")
    parser.add_argument("--fmin", type=float, default=None,
                        help="Low edge of the scoring window (default: the lowest saved bin)")
    parser.add_argument("--fmax", type=float, default=None,
                        help="High edge of the scoring window (default: the highest saved bin)")
    parser.add_argument("--ll-scale", choices=("none", "bandwidth"), default="none",
                        help="'none' = sum_f log B (default); 'bandwidth' = delta_f * sum_f log B, "
                             "i.e. per unit bandwidth. A pure posterior "
                             "temperature -- MAP and peaks are identical either way.")
    parser.add_argument("--top-n", type=int, default=TOP_N,
                        help="How many peaks of p(theta) to track per block")
    parser.add_argument("--cadence-min", type=float, default=None,
                        help="Block spacing in minutes for the theta figures' time axis; "
                             "slots with no block are drawn black (default: the median "
                             "spacing of the blocks present)")
    parser.add_argument("--overlay", nargs="*", choices=tuple(OVERLAY_SPECS) + ("none",),
                        default=["bf_db"],
                        help="Extra linear-scale theta figures carrying a dB series on a "
                             "right-hand axis: 'bf_db' = band-summed beam response at the "
                             "top peak's direction, 'power_db' = the block's total power "
                             "(needs --block-power in phase 1). 'none' draws none.")
    parser.add_argument("--block-power", action="store_true",
                        help="Recover each block's total power, 10*log10(sum_f tr(R)), from "
                             "the saved E3. Roughly doubles phase-1 peak memory, since E3 "
                             "is the same size as B; required for --overlay power_db.")
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES,
                        help="How many blocks, evenly spaced in time, keep their f-theta slices")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <results-dir>/posterior/"
                             "<postfix>_<norm>_f<fmin>-<fmax>Hz_ll<scale>, where "
                             "<postfix> already carries the directional pair and "
                             "the f-bin skip")
    parser.add_argument("--no-stack", action="store_true",
                        help="Skip the record-wide p(E, P): theta and time marginalized "
                             "out of every block and the log-likelihoods summed. Written "
                             "to posterior_stacked_EP.npz in phase 2; needs phase 1 to "
                             "have stored LL_EP_theta_marg.")
    parser.add_argument("--theta-fix", type=float, default=None,
                        help="Also score every block with theta fixed to this direction "
                             "[deg, 0 = north, CW]: (E, P) conditioned on the window "
                             "below, written under <out-dir>/theta_fix/<tag>/ (default: off)")
    parser.add_argument("--theta-half-width", type=float, default=0.0,
                        help="Half-width of the --theta-fix window [deg]; theta is "
                             "integrated over the bins within it. 0 = the single theta bin "
                             "nearest --theta-fix (default)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip phase 1 and redraw from the per-block summaries")
    parser.add_argument("--no-plots", action="store_true",
                        help="Run phase 1 only (for array tasks; draw later with --plots-only)")
    parser.add_argument("--num-shards", type=int, default=num_shards)
    parser.add_argument("--shard-index", type=int, default=shard_index)
    return parser


def resolve_window(block_paths: list[Path], args) -> None:
    """Default --fmin/--fmax to the first block's saved frequency range; check the window."""
    with np.load(block_paths[0]) as d:
        freq = d["freq"]

    if args.fmin is None:
        args.fmin = float(freq.min())
    if args.fmax is None:
        args.fmax = float(freq.max())

    if args.fmin > args.fmax:
        raise SystemExit(f"--fmin {args.fmin} is above --fmax {args.fmax}")

    n_in = int(((freq >= args.fmin) & (freq <= args.fmax)).sum())
    if n_in == 0:
        raise SystemExit(
            f"No saved bin in [{args.fmin}, {args.fmax}] Hz; the blocks hold "
            f"{freq.size} bins spanning {freq.min():.3f}-{freq.max():.3f} Hz."
        )

    print(f"scoring window     : {args.fmin:.3f}-{args.fmax:.3f} Hz "
          f"({n_in} of {freq.size} saved bins), ll_scale={args.ll_scale}")


def main() -> None:
    args = make_parser().parse_args()
    args.overlay = [k for k in args.overlay if k != "none"]

    # Band and f_bin_skip default to what step 1 built; given both, no steering
    # metadata is needed.
    if args.band is None or args.f_bin_skip is None:
        band, meta, _ = das_beamforming.resolve_cache(
            args.results_dir, args.directional_floor, args.directional_power,
            args.f_bin_skip, args.band,
        )
        args.band, args.f_bin_skip = band["name"], int(meta["f_bin_skip"])
    postfix = das_beamforming.steering_postfix(
        args.directional_floor, args.directional_power, args.f_bin_skip
    )
    norm_tag = das_beamforming.normalization_tag(args.is_coherence_norm)

    band_dir = find_block_dir(args.results_dir, args.band, norm_tag, postfix)
    block_paths = sorted(band_dir.glob(f"bfdisp_*_{postfix}.npz"))

    if not block_paths:
        raise SystemExit(
            f"No stage-2 blocks matching bfdisp_*_{postfix}.npz under {band_dir}. "
            "Check --band, the directional pair, --f-bin-skip and --is-coherence-norm, "
            "and that stage 2 has run for that combination."
        )

    check_block_settings(block_paths[0], args.is_coherence_norm,
                         args.f_bin_skip)

    print(f"{len(block_paths)} blocks, cache [{postfix}], covariance [{norm_tag}], "
          f"band {args.band}, f_bin_skip {args.f_bin_skip}")
    resolve_window(block_paths, args)

    out_dir = args.out_dir or out_dir_for(
        args.results_dir, postfix, norm_tag, args.fmin, args.fmax, args.ll_scale
    )
    cfg = {
        "band": args.band,
        "postfix": postfix,
        "norm_tag": norm_tag,
        "f_bin_skip": int(args.f_bin_skip),
        "is_coherence_norm": bool(args.is_coherence_norm),
        "fmin": args.fmin,
        "fmax": args.fmax,
        "ll_scale": args.ll_scale,
        "top_n": args.top_n,
        "n_examples": args.n_examples,
        "block_power": bool(args.block_power),
        "source_dir": str(band_dir),
    }
    check_config(out_dir, cfg)

    blocks_dir = out_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    print(f"out                : {out_dir}")

    # check_config compares against this on the next run.
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Not in config.json: each fixed direction has its own directory and theta_fix.json.
    fix_dir = fix_blocks_dir = None
    if args.theta_fix is not None:
        if args.theta_half_width < 0:
            raise SystemExit(f"--theta-half-width must be >= 0, got {args.theta_half_width}")
        fix_dir = theta_fix_dir(out_dir, args.theta_fix, args.theta_half_width)
        fix_blocks_dir = fix_dir / "blocks"
        fix_blocks_dir.mkdir(parents=True, exist_ok=True)

        with np.load(block_paths[0]) as d:
            theta_deg = d["theta_deg"]
        bins = theta_window_bins(theta_deg, args.theta_fix, args.theta_half_width)
        (fix_dir / "theta_fix.json").write_text(json.dumps({
            "theta_fix_deg": args.theta_fix % 360.0,
            "theta_half_width_deg": args.theta_half_width,
            "theta_bins_deg": theta_deg[bins].tolist(),
        }, indent=2))
        print(f"theta fix          : {args.theta_fix % 360:g} +- {args.theta_half_width:g} deg "
              f"-> {bins.size} bin(s) at {', '.join(f'{x:g}' for x in theta_deg[bins])} deg")
        print(f"theta fix out      : {fix_dir}")

    if not args.plots_only:
        run_phase1(block_paths, blocks_dir, args, fix_blocks_dir)

    if not args.no_plots:
        run_phase2(blocks_dir, out_dir, args)
        if fix_dir is not None:
            run_phase2_fixed(fix_dir, args)


if __name__ == "__main__":
    main()
