"""Step 4: stitch the step-3 per-block posteriors into one small file per run.

step3_analyze_bf_disp_posterior.py leaves one post_*.npz per covariance block.
This flattens them into a few MB of arrays with one row per p(theta) PEAK (so
two arrivals in one block stay separable by theta); each row's p_E, p_P and
joint log-likelihood are conditioned on that peak's own direction. p(theta)
itself is kept per block (LPT_all / output_time_all). E and P are in GPa / MPa.

Reads <results-dir>/posterior/<run>/blocks/post_*.npz (every run under
--results-dir, or those given with --posterior-dir), filtered on each run's
config.json by covariance, steering cache and scoring window. Writes to
<results-dir>/summary/ (default <results-dir> = outputs/bf_inv_disp):

    timeseries_stats_<steering>_<norm>_f<fmin>-<fmax>Hz_ll<scale>.npz
    theta_fixed_<run>_<tag>.npz    one row per BLOCK, for each theta_fix/<tag>/
                                   step 3 wrote with --theta-fix; includes
                                   theta_mass_all (window's share of p(theta))

The filters select among runs step 3 has already scored; a new scoring window
needs a new step-3 run, not a new stitch.

    python step4_stitch_bf_disp_series.py                      # every run
    python step4_stitch_bf_disp_series.py --is-coherence-norm true \
        --directional-floor 1.00 --directional-power 0.00 --fmin 0.10 --fmax 0.40
    python step4_stitch_bf_disp_series.py --theta-fix 322 --theta-fix-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from das_tools.beamforming import das_beamforming
from das_tools.utils import proj_path
from das_tools.utils import stat
from das_tools.utils.helpers import str2bool

# --------------------------------------------------------------------- #
# Config -- paths default via das_tools.utils.proj_path (DAS_LFI_* env vars).
# --------------------------------------------------------------------- #

RESULTS_DIR = proj_path.output_dir("bf_inv_disp")


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #

def read_config(post_dir: Path) -> dict:
    """A step-3 run's config.json, or {} if it has none."""
    path = Path(post_dir) / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_mode(cfg: dict, post_dir: Path) -> str:
    """'<steering>_<norm>' of the step-2 run behind a step-3 run, e.g. omni_coh.

    Without a recorded norm_tag, only the postfix (or directory name) is returned.
    """
    postfix = cfg.get("postfix")
    norm = cfg.get("norm_tag")
    if postfix and norm:
        return f"{postfix}_{norm}"
    return postfix or post_dir.name


def find_posterior_dirs(results_dir: Path) -> list[Path]:
    """Every step-3 output directory under <results_dir>/posterior, sorted by name."""
    root = Path(results_dir) / "posterior"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "blocks").is_dir())


def window_tag(cfg: dict) -> str:
    """A run's scoring window from config.json, e.g. 'f0.10-0.40Hz_llnone'."""
    fmin, fmax = cfg.get("fmin"), cfg.get("fmax")
    if fmin is None or fmax is None:
        return "f?-?"
    return f"f{fmin:.2f}-{fmax:.2f}Hz_ll{cfg.get('ll_scale', 'none')}"


def edge_matches(value, wanted) -> bool:
    """Match a band edge at 2 decimals, the precision of step 3's directory name.

    An unrecorded edge (None) never matches an explicit filter.
    """
    if wanted is None:
        return True
    if value is None:
        return False
    return round(float(value), 2) == round(float(wanted), 2)


def select_dirs(dirs: list[Path], norm_tags: list[str] | None,
                steering: list[str] | None, fmin: float | None = None,
                fmax: float | None = None,
                ll_scales: list[str] | None = None) -> list[Path]:
    """Keep the runs whose config.json matches the covariance / steering / window filters.

    A run missing a filtered field is dropped rather than matched on its name.
    """
    if not norm_tags and not steering and not ll_scales \
            and fmin is None and fmax is None:
        return dirs

    kept = []
    for d in dirs:
        cfg = read_config(d)
        if norm_tags and cfg.get("norm_tag") not in norm_tags:
            continue
        if steering and cfg.get("postfix") not in steering:
            continue
        if not edge_matches(cfg.get("fmin"), fmin):
            continue
        if not edge_matches(cfg.get("fmax"), fmax):
            continue
        if ll_scales and cfg.get("ll_scale") not in ll_scales:
            continue
        kept.append(d)

    return kept


def load_blocks(blocks_dir: Path) -> tuple[list[dict], dict]:
    """Every post_*.npz as a dict, time-ordered, plus the shared grids.

    The grids (E_all, C_all, theta_deg, freq_used, valid_disp) are identical in
    every block and are taken from the first readable one.
    """
    rows: list[dict] = []
    shared: dict = {}
    n_bad = 0

    for path in sorted(blocks_dir.glob("post_*.npz")):
        try:
            d = np.load(path, allow_pickle=False)
            row = {k: d[k] for k in d.files}
        except Exception as exc:
            print(f"    skipping unreadable {path.name}: {exc}")
            n_bad += 1
            continue

        if not shared:
            shared = {
                "E_all": row["E_all"],
                "C_all": row["C_all"],
                "theta_deg": row["theta_deg"],
                "freq_used": row["freq_used"],
            }
            if "valid_disp" in row:
                shared["valid_disp"] = row["valid_disp"]

        row["_t"] = pd.Timestamp(str(row["block_start"]))
        rows.append(row)

    if n_bad:
        print(f"    {n_bad} unreadable block(s) skipped")

    rows.sort(key=lambda r: r["_t"])
    return rows, shared


# --------------------------------------------------------------------- #
# Flatten blocks -> peaks
# --------------------------------------------------------------------- #

def flatten_peaks(rows: list[dict], shared: dict) -> dict:
    """One record per (block, peak), sorted by time; E in GPa, P in MPa.

    Re-sorted explicitly because downstream moving-window code needs a monotone
    time axis. Keys starting with '_' are diagnostics, not output.
    """
    E_all, C_all = shared["E_all"], shared["C_all"]
    has_peak_fields = "peak_p_E" in rows[0]

    rec: dict[str, list] = {k: [] for k in (
        "t", "th_deg", "BF", "pk_freq", "E_map", "P_map", "p_E", "p_P",
        "peak_height", "n_valid_snapshots", "LL",
    )}
    n_dropped = 0

    for row in rows:
        peak_deg = np.atleast_1d(row["peak_theta_deg"])
        peak_h = np.atleast_1d(row["peak_height"])

        for i, deg in enumerate(peak_deg):
            if not np.isfinite(deg):
                # circular_top_peaks pads when a block has fewer peaks than
                # requested; those are absences, not zero-strength arrivals.
                n_dropped += 1
                continue

            rec["t"].append(row["_t"])
            rec["th_deg"].append(float(deg) % 360.0)
            rec["peak_height"].append(float(peak_h[i]))
            rec["n_valid_snapshots"].append(int(row["n_valid_snapshots"]))

            if has_peak_fields:
                rec["BF"].append(float(row["peak_bf_db"][i]))
                rec["pk_freq"].append(float(row["peak_freq_hz"][i]))
                rec["E_map"].append(float(row["peak_E_map"][i]))
                rec["P_map"].append(float(row["peak_P_map"][i]))
                rec["p_E"].append(np.asarray(row["peak_p_E"][i], dtype=float))
                rec["p_P"].append(np.asarray(row["peak_p_P"][i], dtype=float))
                rec["LL"].append(np.asarray(row["peak_LL_curve"][i], dtype=np.float32))
            else:
                # No per-peak fields: every peak carries the block's marginals.
                rec["BF"].append(np.nan)
                rec["pk_freq"].append(np.nan)
                rec["E_map"].append(float(row["E_map"]))
                rec["P_map"].append(float(row["P_map"]))
                rec["p_E"].append(np.asarray(row["p_E"], dtype=float))
                rec["p_P"].append(np.asarray(row["p_P"], dtype=float))

    if not rec["t"]:
        raise RuntimeError("no finite peaks found in any block")

    t = np.array(rec["t"], dtype="datetime64[ns]")
    order = np.argsort(t, kind="stable")

    out = {
        "t_all_global": t[order],
        "th_all_deg_global": np.array(rec["th_deg"], dtype=float)[order],
        "BF_all_global": np.array(rec["BF"], dtype=float)[order],
        "pk_freq_all": np.array(rec["pk_freq"], dtype=float)[order],
        "peak_height_all": np.array(rec["peak_height"], dtype=float)[order],
        "n_valid_snapshots_all": np.array(rec["n_valid_snapshots"], dtype=int)[order],
        "E_map_all": np.array(rec["E_map"], dtype=float)[order] / 1e9,
        "P_map_all": np.array(rec["P_map"], dtype=float)[order] / 1e6,
        "p_E_all": np.vstack(rec["p_E"])[order],
        "p_P_all": np.vstack(rec["p_P"])[order],
    }

    if rec["LL"]:
        out["logL_EP_all"] = np.vstack(rec["LL"])[order]

    # 68/95% credible intervals and median from the marginals.
    (out["E_l68_all"], out["E_u68_all"], out["E_l95_all"], out["E_u95_all"],
     out["E_med_all"], out["E_grid_GPa"]) = stat.compute_stat(out["p_E_all"], E_all / 1e9)
    (out["P_l68_all"], out["P_u68_all"], out["P_l95_all"], out["P_u95_all"],
     out["P_med_all"], out["P_grid_MPa"]) = stat.compute_stat(out["p_P_all"], C_all / 1e6)

    out["_n_dropped"] = n_dropped
    out["_has_peak_fields"] = has_peak_fields
    return out


def theta_posterior_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(LPT_all, output_time_all): p(theta) per BLOCK (nBlocks, nTheta) and its times."""
    t = np.array([r["_t"] for r in rows], dtype="datetime64[ns]")
    order = np.argsort(t, kind="stable")
    LPT = np.vstack([np.asarray(r["p_theta"], dtype=np.float32) for r in rows])
    return LPT[order], t[order]


def theta_fix_dirs(post_dir: Path, args) -> list[Path]:
    """The theta_fix/<tag>/ directories of one step-3 run selected by --theta-fix.

    Matched on theta_fix.json to 0.05 deg; unset --theta-fix keeps them all.
    """
    root = Path(post_dir) / "theta_fix"
    found = sorted(p for p in root.iterdir() if (p / "blocks").is_dir()) if root.is_dir() else []
    if args.theta_fix is None:
        return found

    kept = []
    for d in found:
        meta = d / "theta_fix.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        dth = abs((m["theta_fix_deg"] - args.theta_fix + 180.0) % 360.0 - 180.0)
        if dth < 0.05 and abs(m["theta_half_width_deg"] - args.theta_half_width) < 0.05:
            kept.append(d)
    if not kept:
        print(f"    no theta_fix run at {args.theta_fix:g} +- {args.theta_half_width:g} deg; "
              f"present: {[d.name for d in found] or 'none'}")
    return kept


def stitch_theta_fixes(post_dir: Path, out_dir: Path, args) -> list[Path]:
    """Stitch every selected fixed-direction series of one step-3 run, one file each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [stitch_theta_fix(d, post_dir, out_dir, args) for d in theta_fix_dirs(post_dir, args)]
    return [p for p in written if p is not None]


def stitch_theta_fix(fix_dir: Path, post_dir: Path, out_dir: Path, args) -> Path | None:
    """One theta_fix/<tag>/ of a step-3 run -> theta_fixed_<run>_<tag>.npz."""
    recs = []
    for path in sorted((fix_dir / "blocks").glob("fix_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as d:
                recs.append({k: d[k] for k in d.files})
        except Exception as exc:
            print(f"    skipping unreadable {path.name}: {exc}")
    if not recs:
        print(f"    {fix_dir.name}: no readable blocks, skipping")
        return None

    t = np.array([pd.Timestamp(str(r["block_start"])) for r in recs], dtype="datetime64[ns]")
    order = np.argsort(t, kind="stable")

    def col(key, dtype=float):
        return np.array([r[key] for r in recs], dtype=dtype)[order]

    def rows(key, dtype=float):
        return np.vstack([np.asarray(r[key], dtype=dtype) for r in recs])[order]

    first = recs[0]
    E_all, C_all = first["E_all"], first["C_all"]
    out = {
        "output_time_all": t[order],
        "theta_mass_all": col("theta_mass"),
        "theta_best_deg_all": col("theta_best_deg"),
        "BF_all": col("bf_db"),
        "pk_freq_all": col("freq_hz"),
        "n_valid_snapshots_all": col("n_valid_snapshots", int),
        "E_map_all": col("E_map") / 1e9,
        "P_map_all": col("P_map") / 1e6,
        "p_E_all": rows("p_E"),
        "p_P_all": rows("p_P"),
    }
    if not args.no_joint:
        out["logL_EP_all"] = rows("LL_curve", np.float32)

    (out["E_l68_all"], out["E_u68_all"], out["E_l95_all"], out["E_u95_all"],
     out["E_med_all"], out["E_grid_GPa"]) = stat.compute_stat(out["p_E_all"], E_all / 1e9)
    (out["P_l68_all"], out["P_u68_all"], out["P_l95_all"], out["P_u95_all"],
     out["P_med_all"], out["P_grid_MPa"]) = stat.compute_stat(out["p_P_all"], C_all / 1e6)

    out.update(
        theta_fix_deg=float(first["theta_fix_deg"]),
        theta_half_width_deg=float(first["theta_half_width_deg"]),
        theta_bins_deg=first["theta_bins_deg"],
        valid_disp=first["valid_disp"],
        postfix=post_dir.name,
        theta_fix_tag=fix_dir.name,
    )

    out_path = out_dir / f"theta_fixed_{post_dir.name}_{fix_dir.name}.npz"
    np.savez_compressed(out_path, **out)
    print(f"    {fix_dir.name}: {len(recs)} blocks -> {out_path.name}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB), "
          f"median theta mass {np.nanmedian(out['theta_mass_all']):.2g}")
    return out_path


# --------------------------------------------------------------------- #
# One step-3 directory -> one stats file
# --------------------------------------------------------------------- #

def stitch_one(post_dir: Path, out_dir: Path, args) -> dict | None:
    print(f"\n=== {post_dir.name} ===")

    cfg = read_config(post_dir)
    mode = run_mode(cfg, post_dir)

    rows, shared = load_blocks(post_dir / "blocks")
    if not rows:
        print("    no readable blocks, skipping")
        return None

    flat = flatten_peaks(rows, shared)
    LPT_all, output_time_all = theta_posterior_series(rows)

    n_peaks = flat["t_all_global"].size
    print(f"    {len(rows)} blocks -> {n_peaks} peaks, "
          f"{flat['t_all_global'].min()} .. {flat['t_all_global'].max()}")
    if flat["_n_dropped"]:
        print(f"    {flat['_n_dropped']} requested peak slot(s) had no peak, dropped")
    if not flat["_has_peak_fields"]:
        print("    NOTE: blocks predate the per-peak fields -- BF_all_global and "
              "pk_freq_all are NaN, no logL_EP_all, and p_E/p_P are the block's "
              "direction-averaged marginals. Rerun stage 3 to fill them in.")

    payload = {k: v for k, v in flat.items() if not k.startswith("_")}
    payload.update(
        LPT_all=LPT_all,
        output_time_all=output_time_all,
        theta=np.deg2rad(shared["theta_deg"]),
        theta_deg=shared["theta_deg"],
        freq_used=shared["freq_used"],
        # Provenance; processing_mode is the step-2 run (steering + covariance).
        processing_mode=run_mode(cfg, post_dir),
        postfix=post_dir.name,
        steering_postfix=cfg.get("postfix", "unknown"),
        norm_tag=cfg.get("norm_tag", "unknown"),
        # -1 = not recorded (not the undecimated default 1).
        f_bin_skip=int(cfg.get("f_bin_skip", -1)),
        is_coherence_norm=bool(cfg.get("is_coherence_norm", False)),
        band=np.array([cfg.get("fmin", np.nan), cfg.get("fmax", np.nan)], dtype=float),
        ll_scale=cfg.get("ll_scale", "unknown"),
        source_band=cfg.get("band", "unknown"),
        config_json=json.dumps(cfg),
    )
    if "valid_disp" in shared:
        payload["valid_disp"] = shared["valid_disp"]

    if args.no_joint:
        payload.pop("logL_EP_all", None)

    if "logL_EP_all" in payload:
        n_valid = payload["logL_EP_all"].shape[1]
        expect = int(shared["valid_disp"].sum()) if "valid_disp" in shared else -1
        tag = "matches valid_disp" if n_valid == expect else f"width {n_valid}, valid_disp {expect}"
        print(f"    joint logL: {payload['logL_EP_all'].shape} ({tag})")
    elif not args.no_joint:
        print("    no joint logL available: only the per-block medians are stored")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"timeseries_stats_{post_dir.name}.npz"
    np.savez_compressed(out_path, **payload)

    mb = out_path.stat().st_size / 1e6
    print(f"    wrote {out_path.name}  ({mb:.1f} MB)")

    stitch_theta_fixes(post_dir, out_dir, args)

    th = flat["th_all_deg_global"]
    return {
        "path": out_path,
        "mode": mode,
        "window": f"{cfg.get('fmin', float('nan')):.2f}-{cfg.get('fmax', float('nan')):.2f}",
        "ll_scale": cfg.get("ll_scale", "?"),
        "n_peaks": int(n_peaks),
        "med_theta": float(np.median(th)) if th.size else float("nan"),
        "med_E": float(np.nanmedian(flat["E_med_all"])),
        "med_P": float(np.nanmedian(flat["P_med_all"])),
        "med_BF": float(np.nanmedian(flat["BF_all_global"])),
        "mb": mb,
    }


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def resolve_filters(args) -> tuple[list[str] | None, list[str] | None]:
    """(norm_tags, steering) from either the path tags or the step-2/3 style flags.

    --is-coherence-norm maps to --norm-tag, and --directional-floor/-power
    (+ --f-bin-skip) to --steering-postfix. Giving both spellings is an error.
    """
    norm_tags = list(args.norm_tag) if args.norm_tag else None
    steering = list(args.steering_postfix) if args.steering_postfix else None

    if args.is_coherence_norm is not None:
        if norm_tags:
            raise SystemExit(
                "Use either --norm-tag or --is-coherence-norm, not both."
            )
        norm_tags = [das_beamforming.normalization_tag(
            bool(args.is_coherence_norm)
        )]

    if args.directional_floor is not None or args.directional_power is not None:
        if steering:
            raise SystemExit(
                "Use either --steering-postfix or --directional-floor/--directional-power, "
                "not both."
            )
        if args.directional_floor is None or args.directional_power is None:
            raise SystemExit(
                "--directional-floor and --directional-power name one steering cache "
                "together; give both (--directional-power 0 selects the omni cache)."
            )
        # Unset --f-bin-skip: the skip step 1 recorded for this pair, else 1.
        skip = args.f_bin_skip
        if skip is None:
            try:
                _, meta, _ = das_beamforming.resolve_cache(
                    args.results_dir, args.directional_floor, args.directional_power)
                skip = int(meta["f_bin_skip"])
            except FileNotFoundError:
                skip = 1
                print("NOTE: no stage-1 steering metadata under --results-dir; "
                      "selecting f_bin_skip 1. Pass --f-bin-skip to choose another.")
        steering = [das_beamforming.steering_postfix(
            args.directional_floor, args.directional_power, skip
        )]

    return norm_tags, steering


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch stage-3 per-block posteriors into one transferable "
                    "stats file per scoring run (stage 4)."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Root holding posterior/; every run under it is stitched "
                             "unless --posterior-dir is given")
    parser.add_argument("--posterior-dir", type=Path, nargs="*", default=None,
                        help="One or more specific stage-3 output directories")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where the stats files go (default: <results-dir>/summary)")
    parser.add_argument("--norm-tag", nargs="*", default=None,
                        choices=("raw", "coh"),
                        help="Only stitch runs with these covariance treatments "
                             "(default: all of them)")
    parser.add_argument("--steering-postfix", nargs="*", default=None,
                        help="Only stitch runs against these steering caches, "
                             "e.g. omni dirSen_025_pwr150")
    # Same flags as steps 2 and 3; default None means "any".
    parser.add_argument("--is-coherence-norm", type=str2bool, default=None,
                        help="true/false picks the coherence-normalized "
                             "or the un-normalized run")
    parser.add_argument("--directional-floor", type=float, default=None,
                        help="Select the steering cache the way stages 1-3 do, instead of "
                             "by --steering-postfix; must be given with --directional-power")
    parser.add_argument("--directional-power", type=float, default=None,
                        help="With --directional-floor, names one steering cache; "
                             "0 selects the omni cache")
    parser.add_argument("--f-bin-skip", type=int, default=None,
                        help="Completes the cache name built from --directional-floor / "
                             "--directional-power: 3 selects the runs beamformed at "
                             "--f-bin-skip 3 (postfix ..._skip3). Default: the skip "
                             "stage 1 built for that pair. Ignored when the cache is "
                             "named directly with --steering-postfix")
    # These select among windows step 3 already scored; matched at 2 decimals.
    parser.add_argument("--fmin", type=float, default=None,
                        help="Only stitch runs scored from this frequency, e.g. 0.10 "
                             "(default: any). Selects among existing stage-3 runs; "
                             "to score a window nobody has run yet, rerun stage 3")
    parser.add_argument("--fmax", type=float, default=None,
                        help="Only stitch runs scored up to this frequency, e.g. 0.40 "
                             "(default: any)")
    parser.add_argument("--ll-scale", nargs="*", default=None,
                        choices=("none", "bandwidth"),
                        help="Only stitch runs scored with these likelihood scales "
                             "(default: all of them)")
    parser.add_argument("--theta-fix", type=float, default=None,
                        help="Only stitch the fixed-direction series stage 3 scored with "
                             "this --theta-fix [deg] (default: every one present)")
    parser.add_argument("--theta-half-width", type=float, default=0.0,
                        help="With --theta-fix, the --theta-half-width that series was "
                             "scored with (default 0, the nearest-bin series)")
    parser.add_argument("--theta-fix-only", action="store_true",
                        help="Write only the theta_fixed_*.npz files; leave the "
                             "timeseries_stats files as they are")
    parser.add_argument("--no-joint", action="store_true",
                        help="Drop logL_EP_all. Much smaller file, with only the per-block "
                             "medians, not the joint (E, P) log-likelihood.")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    dirs = [Path(p) for p in args.posterior_dir] if args.posterior_dir \
        else find_posterior_dirs(args.results_dir)

    if not dirs:
        raise SystemExit(
            f"No stage-3 output directories found under {args.results_dir / 'posterior'}. "
            "Run step3_analyze_bf_disp_posterior.py first, or name one with --posterior-dir."
        )

    norm_tags, steering = resolve_filters(args)

    ll_scales = list(args.ll_scale) if args.ll_scale else None
    window = (f"{args.fmin if args.fmin is not None else 'any'}-"
              f"{args.fmax if args.fmax is not None else 'any'} Hz"
              f" ll {ll_scales or 'any'}")

    n_found = len(dirs)
    dirs = select_dirs(dirs, norm_tags, steering, args.fmin, args.fmax, ll_scales)
    if not dirs:
        # List what does exist so a mistyped filter is easy to fix.
        available = sorted({f"{run_mode(cfg, d)}  {window_tag(cfg)}"
                            for d in find_posterior_dirs(args.results_dir)
                            for cfg in (read_config(d),)})
        raise SystemExit(
            f"No run matches covariance {norm_tags} / steering {steering} / "
            f"window {window}. Present:\n    " + "\n    ".join(available)
        )
    if len(dirs) != n_found:
        print(f"{n_found} run(s) found, {len(dirs)} kept by "
              f"covariance {norm_tags or 'any'} / steering {steering or 'any'} / "
              f"window {window}")

    out_dir = args.out_dir or (Path(args.results_dir) / "summary")
    print(f"{len(dirs)} scoring run(s) -> {out_dir}")

    if args.theta_fix_only:
        written = []
        for d in dirs:
            print(f"\n=== {d.name} ===")
            written += stitch_theta_fixes(d, out_dir, args)
        if not written:
            raise SystemExit("no fixed-direction series written")
        print(f"\n{len(written)} file(s). To fetch them:")
        print(f"    scp '<server>:{out_dir}/theta_fixed_*.npz' .")
        return

    done = [stitch_one(d, out_dir, args) for d in dirs]
    done = [r for r in done if r is not None]

    if not done:
        raise SystemExit("nothing written")

    # Side-by-side summary of the runs written.
    print(f"\n{'processing mode':<26}{'window Hz':>11}{'ll':>10}{'peaks':>7}"
          f"{'med th':>8}{'med E':>8}{'med P':>8}{'med BF':>8}{'MB':>7}")
    print("-" * 92)
    for r in sorted(done, key=lambda r: (r["mode"], r["window"])):
        print(f"{r['mode']:<26}{r['window']:>11}{r['ll_scale']:>10}{r['n_peaks']:>7}"
              f"{r['med_theta']:>8.1f}{r['med_E']:>8.2f}{r['med_P']:>8.3f}"
              f"{r['med_BF']:>8.1f}{r['mb']:>7.1f}")

    total = sum(r["path"].stat().st_size for r in done) / 1e6
    print(f"\n{len(done)} file(s), {total:.1f} MB total. To fetch them:")
    print(f"    scp '<server>:{out_dir}/timeseries_stats_*.npz' .")


if __name__ == "__main__":
    main()
