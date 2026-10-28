"""Diagnostic: sensitivity of one block's (E, P) estimate to the fixed ice/water
parameters nu, h_w, rho_i and rho_w.

Each parameter is swept one at a time around the baseline (the ice config
recorded in step 1's steering_meta, else step1_steer_vec_disp.ICE_CFG), and
steps 2 + 3 are re-run on a single covariance block for every value. Changing
any of these moves every k(f; E, P), so the step-1 steering cache cannot be
reused: each value gets its own dispersion grid and its steering vectors are
built on the fly, keeping only LL(curve, theta). Covariance, geometry and
posterior code are imported from steps 1-3, so the baseline run should
reproduce step 3's summary of the same block -- check that first.

The (E, P) grid is shared across values, but the set of valid (subcritical)
curves moves with nu and rho_w, so valid_disp is saved per value.

Writes <results-dir>/sensitivity/<block>_<postfix>_<norm>_f<fmin>-<fmax>Hz_ll<scale>/:

    runs/sens_<task>.npz          one per (parameter, value) or baseline; resumable
    disp/disp_<task>.npz          dispersion grid for that ice config
    sensitivity_<param>.npz/.csv  that parameter's sweep stacked with the baseline,
                                  sorted by value (written when its sweep ends)

One SLURM array element per parameter (slurms/diag_sensitivity.slurm).

    python diag_sensitivity_ice_params.py --list-tasks
    python diag_sensitivity_ice_params.py --params rho_w
    python diag_sensitivity_ice_params.py --params nu --nu-values 0.25:0.40:0.05
    python diag_sensitivity_ice_params.py --combine      # re-stack finished runs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Import the pipeline scripts so this test runs the production code, not a copy.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_bf_disp_hourly as stage2          # noqa: E402
import step1_steer_vec_disp as stage1          # noqa: E402
import step3_analyze_bf_disp_posterior as stage3    # noqa: E402

from das_tools.readers import decimated                  # noqa: E402
from das_tools.processing import pisces25                # noqa: E402
from das_tools.beamforming import das_beamforming, steering  # noqa: E402
from das_tools.utils.helpers import savez_atomic, str2bool  # noqa: E402

# --------------------------------------------------------------------- #
# Config -- paths default via das_tools.utils.proj_path (DAS_LFI_* env vars).
# --------------------------------------------------------------------- #

DAS_DATA_PATH = stage2.DAS_DATA_PATH
DATA_FILE = DAS_DATA_PATH / "das_20250506_000000_5Hz.h5"
RESULTS_DIR = stage2.RESULTS_DIR

# Default sweeps, stop inclusive. The baseline is always run as its own task.
SWEEP_DEFAULTS = {
    "nu":    "0.10:0.40:0.02",     # Poisson's ratio [-]
    "h_w":   "5:15:.5",             # water depth [m]
    "rho_i": "860:960:10",         # ice density [kg/m^3]
    "rho_w": "990:1030:5",         # water density [kg/m^3]
}
PARAM_LABELS = {
    "nu": "Poisson's ratio",
    "h_w": "water depth [m]",
    "rho_i": "ice density [kg/m^3]",
    "rho_w": "water density [kg/m^3]",
}

# Curves per steering chunk: (M, chunk * nTheta) complex128, ~66 MB at
# 256 curves x 180 theta x 90 channels (R @ A doubles it).
CHUNK_CURVES = 256

EPS = stage2.EPS


# --------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------- #

def parse_values(spec: str) -> np.ndarray:
    """'start:stop:step' (stop inclusive) or 'a,b,c' -> sorted unique floats."""
    spec = spec.strip()
    if ":" in spec:
        start, stop, step = (float(s) for s in spec.split(":"))
        if step <= 0:
            raise ValueError(f"step must be > 0 in {spec!r}")
        vals = np.arange(start, stop + 0.5 * step, step)
    else:
        vals = np.array([float(s) for s in spec.split(",") if s.strip()])
    return np.unique(np.round(vals, 10))


def task_name(param: str | None, value: float | None) -> str:
    return "baseline" if param is None else f"{param}_{value:.6g}"


def build_tasks(params: list[str], sweeps: dict[str, np.ndarray],
                baseline: steering.IceConfig) -> list[dict]:
    """Baseline first, then every (param, value) that differs from it.

    Deterministic, so every SLURM array element derives the same list.
    """
    tasks = [{"name": "baseline", "param": None, "value": None}]
    for p in params:
        base_v = float(getattr(baseline, p))
        for v in sweeps[p]:
            if np.isclose(v, base_v):
                continue
            tasks.append({"name": task_name(p, float(v)), "param": p, "value": float(v)})
    return tasks


def task_ice_cfg(task: dict, baseline: steering.IceConfig) -> steering.IceConfig:
    if task["param"] is None:
        return baseline
    return dataclasses.replace(baseline, **{task["param"]: task["value"]})


def load_baseline_cfg(results_dir: Path, band_name: str, postfix: str
                      ) -> tuple[steering.IceConfig, str]:
    """(baseline IceConfig, its source): steering_meta_<postfix>.json's ice_cfg
    if recorded, else step1_steer_vec_disp.ICE_CFG.
    """
    meta_path = das_beamforming.steering_meta_path(results_dir, band_name, postfix)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "ice_cfg" in meta:
            cfg = steering.IceConfig(**meta["ice_cfg"])
            if cfg != stage1.ICE_CFG:
                print(f"NOTE: {meta_path.name} ice_cfg differs from "
                      "step1_steer_vec_disp.ICE_CFG; using the cache's (what was run).")
            return cfg, str(meta_path)
    return stage1.ICE_CFG, "step1_steer_vec_disp.ICE_CFG"


# --------------------------------------------------------------------- #
# Data: one covariance block
# --------------------------------------------------------------------- #

def load_block_covariance(data_file: Path, block_start: pd.Timestamp | None,
                          band: dict, args):
    """Averaged covariance of one block, as step 2 forms it (plus timing/channel info)."""
    with h5py.File(data_file, "r") as f:
        file_start = pd.Timestamp(f.attrs["start_time"])
    hour = file_start.floor("h")
    hourly_map = {hour: data_file}

    if block_start is None:
        block_start = hour
    if block_start.tzinfo is None and hour.tzinfo is not None:
        block_start = block_start.tz_localize(hour.tzinfo)
    block_end = block_start + pd.Timedelta(hours=band["cov_duration_h"])

    ds = decimated.open_decimated(data_file, time_slice=slice(0, 1))
    if not bool(ds.attrs.get("channel_is_bend_relative", False)):
        raise ValueError(f"{data_file} has an absolute channel axis; this test needs "
                         "the bend-relative axis written by decimate_merge.")

    loaded = decimated.load_block(hourly_map, block_start, block_end)
    if loaded is None:
        raise SystemExit(
            f"[{block_start} .. {block_end}) is not covered by {data_file.name} "
            f"(hour {hour}). --block-start must leave a full "
            f"{band['cov_duration_h']} h inside that file."
        )
    block, channel, fs = loaded

    cov = stage2.block_covariance(block, fs, band)
    if cov is None:
        raise SystemExit(f"Block {block_start} has fewer than "
                         f"{stage2.MIN_VALID_SNAPSHOTS} clean snapshots.")
    Rk, freq, n_valid, n_expected = cov

    if args.is_coherence_norm:
        Rk = stage2.normalize_to_coherence(Rk)

    expected = stage1.band_frequencies(fs, band)
    if freq.shape != expected.shape or not np.allclose(freq, expected):
        raise ValueError("Covariance frequency grid differs from the band's rFFT bins.")

    return {
        "Rk": Rk, "freq": freq, "fs": fs, "channel": channel,
        "block_start": block_start, "block_end": block_end,
        "n_valid": n_valid, "n_expected": n_expected,
    }


# --------------------------------------------------------------------- #
# Dispersion grid, per ice config
# --------------------------------------------------------------------- #

def load_or_build_disp(path: Path, freq: np.ndarray, theta_rad: np.ndarray,
                       cfg: steering.IceConfig) -> dict:
    """steering.load_or_build_disp_curves with an atomic write.

    Array tasks build concurrently; a job killed mid-save must not leave a
    valid-looking archive.
    """
    if path.exists():
        try:
            with np.load(path) as npz:
                cached = {k: npz[k] for k in npz.files}
            if steering._disp_cache_matches(cached, freq, cfg):
                return cached
            print(f"  [disp] {path.name} does not match this config -- rebuilding")
        except Exception as exc:
            print(f"  [disp] unreadable {path.name} ({exc}) -- rebuilding")

    disp = steering.compute_disp_curves(None, freq, theta_rad, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(path, **disp)
    return disp


# --------------------------------------------------------------------- #
# Beam log-likelihood, steering built on the fly
# --------------------------------------------------------------------- #

def beam_log_likelihood(Rk: np.ndarray, k_disp_valid: np.ndarray, f_idx: np.ndarray,
                        theta_rad: np.ndarray, ch_xy: np.ndarray, W_theta: np.ndarray,
                        chunk_curves: int) -> np.ndarray:
    """LL(curve, theta) = sum_{f in f_idx} log(B(f, curve, theta) + eps), (nCurves, nTheta).

    Same beam and steering vectors as step 2 (see das_beamforming), built per
    frequency and curve chunk instead of read from the step-1 cache.
    """
    n_curves = k_disp_valid.shape[0]
    n_theta = theta_rad.size

    LL = np.zeros((n_curves, n_theta), dtype=np.float64)

    for i in f_idx:
        Rf = Rk[i]
        tr_R = np.real(np.trace(Rf))
        k_f = np.round(k_disp_valid[:, i], 12)

        for c0 in range(0, n_curves, chunk_curves):
            k = k_f[c0:c0 + chunk_curves]
            n = k.size

            # Step 2 beamforms with conj(A_cache).
            A = das_beamforming.steering_matrix(k, theta_rad, ch_xy, W_theta).conj()
            num, a_norm2 = das_beamforming.quadratic_form(Rf, A)
            B = num / np.maximum(tr_R * a_norm2, EPS)

            # Clip round-off below zero so log() cannot produce NaN.
            LL[c0:c0 + n] += np.log(np.maximum(B, 0.0).reshape(n, n_theta) + EPS)

    return LL


# --------------------------------------------------------------------- #
# One task
# --------------------------------------------------------------------- #

def summarize_posterior(LL: np.ndarray, valid_pairs: np.ndarray, E_all: np.ndarray,
                        C_all: np.ndarray, theta_deg: np.ndarray) -> dict:
    """The step-3 per-block summary, reduced to what this test reads."""
    post = stage3.posterior_from_LL(LL)
    if post is None:
        raise RuntimeError("posterior has no finite mass")

    nE, nP = E_all.size, C_all.size
    p_curve = post.sum(axis=1)
    p_theta = post.sum(axis=0)

    post_EP = stage3.scatter_curve_to_grid(p_curve, valid_pairs, nE, nP, fill=0.0)
    p_E = post_EP.sum(axis=1)
    p_P = post_EP.sum(axis=0)

    LL_EP = stage3.scatter_curve_to_grid(
        stage3.ll_marginal_over_theta(LL), valid_pairs, nE, nP
    )

    # Joint MAP over (E, P, theta), as step 3 reports it ...
    c_map, t_map = np.unravel_index(int(np.argmax(post)), post.shape)
    e_map, p_map = valid_pairs[c_map]
    # ... and the mode of p(E, P); they differ when p is spread over theta.
    e_ep, p_ep = np.unravel_index(int(np.argmax(post_EP)), post_EP.shape)

    peak_deg, peak_height = stage3.circular_top_peaks(p_theta, theta_deg, 1)
    stats_E = stage3.stats_from_1d_dist(E_all, p_E)
    stats_P = stage3.stats_from_1d_dist(C_all, p_P)

    return dict(
        post_EP=post_EP,
        LL_EP_theta_marg=LL_EP,
        LL_curve_theta=(LL - np.nanmax(LL)).astype(np.float32),
        p_E=p_E, p_P=p_P, p_theta=p_theta,
        E_map=float(E_all[e_map]), P_map=float(C_all[p_map]),
        theta_map_deg=float(theta_deg[t_map]), post_map=float(post[c_map, t_map]),
        E_map_EP=float(E_all[e_ep]), P_map_EP=float(C_all[p_ep]),
        post_EP_map=float(post_EP[e_ep, p_ep]),
        peak_theta_deg=float(peak_deg[0]), peak_height=float(peak_height[0]),
        E_mean=stats_E["mean"], E_p05=stats_E["p05"], E_p50=stats_E["p50"], E_p95=stats_E["p95"],
        P_mean=stats_P["mean"], P_p05=stats_P["p05"], P_p50=stats_P["p50"], P_p95=stats_P["p95"],
    )


def run_task(task: dict, ctx: dict, args) -> str:
    out_path = ctx["out_dir"] / "runs" / f"sens_{task['name']}.npz"
    if out_path.exists() and not args.overwrite:
        return "exists"

    cfg = task_ice_cfg(task, ctx["baseline"])
    t0 = time.time()

    disp = load_or_build_disp(ctx["out_dir"] / "disp" / f"disp_{task['name']}.npz",
                              ctx["freq"], ctx["theta_rad"], cfg)
    t_disp = time.time() - t0

    valid_pairs = np.asarray(disp["valid_pairs"])
    k_disp_valid = np.asarray(disp["k_disp_valid"])
    E_all = np.asarray(disp["E_all"], dtype=float)
    C_all = np.asarray(disp["P_all"], dtype=float)

    if valid_pairs.shape[0] == 0:
        print(f"  [{task['name']}] no valid dispersion curve -- nothing to score")
        return "no-curves"

    t1 = time.time()
    LL = beam_log_likelihood(ctx["Rk"], k_disp_valid, ctx["f_idx"], ctx["theta_rad"],
                             ctx["ch_xy"], ctx["W_theta"], args.chunk_curves)
    if args.ll_scale == "bandwidth":
        LL *= 1.0 / float(ctx["band"]["window_s"])
    t_bf = time.time() - t1

    summ = summarize_posterior(LL, valid_pairs, E_all, C_all, ctx["theta_deg"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    savez_atomic(
        out_path,
        task_name=task["name"],
        param_name=task["param"] or "baseline",
        param_value=np.nan if task["value"] is None else task["value"],
        **{f"ice_{k}": v for k, v in dataclasses.asdict(cfg).items()},
        data_file=str(ctx["data_file"]),
        block_start=ctx["block_start"].isoformat(),
        block_end=ctx["block_end"].isoformat(),
        n_valid_snapshots=ctx["n_valid"],
        n_expected_snapshots=ctx["n_expected"],
        band_name=ctx["band"]["name"],
        fmin=args.fmin, fmax=args.fmax, ll_scale=args.ll_scale,
        f_bin_skip=args.f_bin_skip,
        freq_used=ctx["freq"][ctx["f_idx"]],
        n_freq_used=int(ctx["f_idx"].size),
        is_coherence_norm=args.is_coherence_norm,
        directional_floor=args.directional_floor,
        directional_power=args.directional_power,
        theta_deg=ctx["theta_deg"],
        E_all=E_all, C_all=C_all,
        valid_pairs=valid_pairs,
        valid_disp=stage3.scatter_valid(valid_pairs, E_all.size, C_all.size),
        n_curves_valid=int(valid_pairs.shape[0]),
        t_disp_s=t_disp, t_beam_s=t_bf,
        **summ,
    )

    return (f"E_map={summ['E_map'] / 1e9:.2f} GPa  P_map={summ['P_map'] / 1e6:.2f} MPa  "
            f"theta={summ['theta_map_deg']:.0f}  | p(E,P) mode "
            f"{summ['E_map_EP'] / 1e9:.2f} GPa, {summ['P_map_EP'] / 1e6:.2f} MPa  "
            f"[{valid_pairs.shape[0]} curves, disp {t_disp:.0f}s, beam {t_bf:.0f}s]")


# --------------------------------------------------------------------- #
# Combine
# --------------------------------------------------------------------- #

SCALAR_KEYS = ("E_map", "P_map", "theta_map_deg", "post_map", "E_map_EP", "P_map_EP",
               "post_EP_map", "peak_theta_deg", "E_mean", "E_p05", "E_p50", "E_p95",
               "P_mean", "P_p05", "P_p50", "P_p95", "n_curves_valid")
ARRAY_KEYS = ("post_EP", "LL_EP_theta_marg", "valid_disp", "p_E", "p_P", "p_theta")


def combine(tasks: list[dict], params: list[str], ctx: dict) -> None:
    runs_dir = ctx["out_dir"] / "runs"
    loaded = {}
    for t in (t for t in tasks if t["param"] is None or t["param"] in params):
        path = runs_dir / f"sens_{t['name']}.npz"
        if not path.exists():
            print(f"  missing {path.name}")
            continue
        with np.load(path) as d:
            loaded[t["name"]] = {k: d[k] for k in d.files}

    if "baseline" not in loaded:
        raise SystemExit("The baseline task has not been run; nothing to compare against.")

    base = loaded["baseline"]
    for p in params:
        base_v = float(getattr(ctx["baseline"], p))
        series = [(base_v, "baseline", True)] + [
            (t["value"], t["name"], False) for t in tasks
            if t["param"] == p and t["name"] in loaded
        ]
        series.sort(key=lambda s: s[0])

        out = {
            "param_name": p,
            "param_label": PARAM_LABELS[p],
            "baseline_value": base_v,
            "values": np.array([s[0] for s in series]),
            "task_names": np.array([s[1] for s in series]),
            "is_baseline": np.array([s[2] for s in series]),
            "E_all": base["E_all"], "C_all": base["C_all"],
            "theta_deg": base["theta_deg"], "freq_used": base["freq_used"],
            "block_start": base["block_start"], "data_file": base["data_file"],
            "fmin": base["fmin"], "fmax": base["fmax"], "ll_scale": base["ll_scale"],
        }
        for k in SCALAR_KEYS:
            out[k] = np.array([float(loaded[s[1]][k]) for s in series])
        for k in ARRAY_KEYS:
            out[k] = np.stack([loaded[s[1]][k] for s in series])
        for q in SWEEP_DEFAULTS:
            out[f"ice_{q}"] = np.array([float(loaded[s[1]][f"ice_{q}"]) for s in series])

        path = ctx["out_dir"] / f"sensitivity_{p}.npz"
        savez_atomic(path, **out)

        # One CSV per parameter: array tasks combine concurrently.
        rows = pd.DataFrame({"value": out["values"], "task": out["task_names"],
                             "is_baseline": out["is_baseline"],
                             **{k: out[k] for k in SCALAR_KEYS}})
        rows.to_csv(ctx["out_dir"] / f"sensitivity_{p}.csv", index=False)

        print(f"\n  {PARAM_LABELS[p]}  ->  {path.name}, {path.stem}.csv")
        print(f"    {'value':>10}  {'E_map GPa':>9}  {'P_map MPa':>9}  "
              f"{'E_EP GPa':>8}  {'P_EP MPa':>8}  {'theta':>5}")
        for i, v in enumerate(out["values"]):
            mark = "  <- baseline" if out["is_baseline"][i] else ""
            print(f"    {v:>10.4g}  {out['E_map'][i] / 1e9:>9.2f}  {out['P_map'][i] / 1e6:>9.2f}  "
                  f"{out['E_map_EP'][i] / 1e9:>8.2f}  {out['P_map_EP'][i] / 1e6:>8.2f}  "
                  f"{out['theta_map_deg'][i]:>5.0f}{mark}")


# --------------------------------------------------------------------- #
# Setup + CLI
# --------------------------------------------------------------------- #

def check_config(out_dir: Path, cfg: dict) -> None:
    """Refuse to mix runs of a different baseline/data/grid into one directory.

    Finished tasks are skipped by existence, so a mismatch would reuse stale files.
    """
    path = out_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        differs = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if differs:
            detail = ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in differs.items())
            raise SystemExit(f"{out_dir} holds runs with a different config ({detail}). "
                             "Delete it or pass --out-dir.")
    else:
        # Atomic: concurrent array tasks must not read a half-written file.
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(cfg, indent=2))
        os.replace(tmp, path)


def make_parser() -> argparse.ArgumentParser:
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    parser = argparse.ArgumentParser(
        description="One-at-a-time sensitivity of the (E, P) inversion to nu, h_w, "
                    "rho_i and rho_w, on a single covariance block."
    )
    parser.add_argument("--data-file", type=Path, default=DATA_FILE,
                        help="Hourly 5 Hz decimated h5 file holding the block")
    parser.add_argument("--block-start", type=str, default=None,
                        help="ISO start of the block (default: the file's hour, i.e. its first 30 min)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <results-dir>/sensitivity/<block>_<postfix>_<norm>_<window>")
    parser.add_argument("--band", type=str, default=None,
                        help="Band stage 1 built, e.g. 0.10-1.00Hz; only needed when it built several")
    parser.add_argument("--ch-geo-path", type=Path, default=stage1.CH_GEO_PATH)
    parser.add_argument("--theta-step-deg", type=float, default=stage1.THETA_STEP_DEG)
    parser.add_argument("--f-bin-skip", type=int, default=None,
                        help="Beamform every Nth bin only -- trades resolution in f for speed "
                             "(default: the skip of stage 1's cache, else 1)")
    parser.add_argument("--fmin", type=float, default=0.10, help="Scoring window, Hz")
    parser.add_argument("--fmax", type=float, default=1.00, help="Scoring window, Hz")
    parser.add_argument("--ll-scale", choices=("none", "bandwidth"), default="none")
    parser.add_argument("--is-coherence-norm", type=str2bool, default=stage2.IS_COHERENCE_NORM)
    # Applied directly (no cache is used); they also select step 1's band/skip metadata.
    parser.add_argument("--directional-floor", type=float, default=1.0)
    parser.add_argument("--directional-power", type=float, default=0.0)
    parser.add_argument("--params", nargs="+", choices=list(SWEEP_DEFAULTS),
                        default=list(SWEEP_DEFAULTS))
    for p, spec in SWEEP_DEFAULTS.items():
        parser.add_argument(f"--{p.replace('_', '-')}-values", dest=f"{p}_values",
                            type=str, default=spec,
                            help=f"{PARAM_LABELS[p]}: start:stop:step or a,b,c (default {spec})")
    parser.add_argument("--chunk-curves", type=int, default=CHUNK_CURVES)
    parser.add_argument("--param-index", type=int,
                        default=None if task_id is None else int(task_id),
                        help="Sweep only --params[i] (default: SLURM_ARRAY_TASK_ID; "
                             "unset sweeps every parameter in turn)")
    parser.add_argument("--list-tasks", action="store_true", help="Print the task list and exit")
    parser.add_argument("--combine", action="store_true",
                        help="Only re-stack finished runs into sensitivity_<param>.npz")
    parser.add_argument("--overwrite", action="store_true", help="Redo tasks already saved")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    # Band and f_bin_skip as step 1 recorded them (das_beamforming.resolve_cache);
    # without step-1 metadata, the default band and no skip.
    try:
        band, meta, _ = das_beamforming.resolve_cache(
            args.results_dir, args.directional_floor, args.directional_power,
            args.f_bin_skip, args.band,
        )
        if args.f_bin_skip is None:
            args.f_bin_skip = int(meta["f_bin_skip"])
    except FileNotFoundError:
        if args.band is not None:
            raise
        band = das_beamforming.make_band(**das_beamforming.DEFAULT_BAND)
        args.f_bin_skip = args.f_bin_skip or 1
        print(f"NOTE: no matching stage-1 cache under {args.results_dir}; using the "
              f"default band {band['name']} and f_bin_skip {args.f_bin_skip}.")
    postfix = das_beamforming.steering_postfix(
        args.directional_floor, args.directional_power, args.f_bin_skip
    )
    baseline, baseline_src = load_baseline_cfg(args.results_dir, band["name"], postfix)

    sweeps = {p: parse_values(getattr(args, f"{p}_values")) for p in args.params}
    tasks = build_tasks(args.params, sweeps, baseline)

    print(f"baseline ({baseline_src}): nu={baseline.nu} h_w={baseline.h_w} "
          f"rho_i={baseline.rho_i} rho_w={baseline.rho_w} h_i={baseline.h_i}")
    if args.list_tasks:
        for i, p in enumerate(args.params):
            names = [t["name"] for t in tasks if t["param"] == p]
            print(f"  array {i}: {p:<5}  {len(names)} values + baseline  "
                  f"({', '.join(n[len(p) + 1:] for n in names)})")
        print(f"{len(args.params)} parameters -> sbatch --array=0-{len(args.params) - 1}")
        return

    if args.param_index is None:
        my_params = list(args.params)
    elif 0 <= args.param_index < len(args.params):
        my_params = [args.params[args.param_index]]
    else:
        print(f"--param-index {args.param_index} is outside 0..{len(args.params) - 1} "
              f"({args.params}); nothing to do")
        return

    block_start = None if args.block_start is None else pd.Timestamp(args.block_start)
    blk = load_block_covariance(args.data_file, block_start, band, args)

    tag = das_beamforming.normalization_tag(args.is_coherence_norm)
    out_dir = args.out_dir or (
        Path(args.results_dir) / "sensitivity" /
        f"{blk['block_start']:%Y%m%d_%H%M%S}_{postfix}_{tag}_"
        f"{stage3.window_tag(args.fmin, args.fmax, args.ll_scale)}"
    )

    check_config(out_dir, {
        "data_file": str(args.data_file),
        "block_start": blk["block_start"].isoformat(),
        "band": band["name"],
        "baseline": dataclasses.asdict(baseline),
        "theta_step_deg": args.theta_step_deg,
        "f_bin_skip": args.f_bin_skip,
        "directional_floor": args.directional_floor,
        "directional_power": args.directional_power,
        "is_coherence_norm": args.is_coherence_norm,
    })

    theta_deg = np.arange(0.0, 360.0, args.theta_step_deg)
    theta_rad = np.deg2rad(theta_deg)

    ch_xy, _m1 = pisces25.load_leg_geometry(args.ch_geo_path, blk["channel"])
    _phi, W_theta = das_beamforming.directional_weight_table(
        ch_xy, theta_rad, args.directional_floor, args.directional_power,
        stage1.ORIENTATION_HALF_WINDOW,
    )

    freq = blk["freq"]
    kept = np.arange(0, freq.size, args.f_bin_skip)
    f_idx = kept[(freq[kept] >= args.fmin) & (freq[kept] <= args.fmax)]
    if f_idx.size == 0:
        raise SystemExit(f"No bin in [{args.fmin}, {args.fmax}] Hz "
                         f"(band spans {freq.min():.3f}-{freq.max():.3f} Hz).")

    ctx = dict(blk, out_dir=out_dir, band=band, baseline=baseline, data_file=args.data_file,
               theta_deg=theta_deg, theta_rad=theta_rad, ch_xy=ch_xy, W_theta=W_theta,
               f_idx=f_idx)

    print(f"block    : {blk['block_start']} .. {blk['block_end']} "
          f"({blk['n_valid']}/{blk['n_expected']} snapshots) from {args.data_file.name}")
    print(f"scoring  : {f_idx.size} bins in {args.fmin}-{args.fmax} Hz, "
          f"{theta_deg.size} theta, [{postfix}, {tag}, ll {args.ll_scale}]")
    print(f"-> {out_dir}")

    if args.combine:
        combine(tasks, args.params, ctx)
        return

    # Every task needs the baseline and runs it LAST, so the first to get there
    # writes it and later ones skip it (concurrent writes are atomic and identical).
    mine = [t for t in tasks if t["param"] in my_params] + [tasks[0]]
    print(f"sweeping : {', '.join(my_params)} ({len(mine)} runs incl. baseline)")

    for t in mine:
        t0 = time.time()
        status = run_task(t, ctx, args)
        print(f"  [{t['name']}] {status} [{time.time() - t0:.0f}s]", flush=True)

    combine(tasks, my_params, ctx)


if __name__ == "__main__":
    main()
