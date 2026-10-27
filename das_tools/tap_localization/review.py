"""Manual review of the tap-test arrival channel.

For every hammer, `generate_review_candidates` loads the FG window, proposes up
to ~10 candidate channels (amplitude peaks, lag-curve minima and the automatic
pick) and saves a waterfall with the candidates numbered. A person then picks
the channel in `review_candidates_interactive`; the automatic suggestions are
never used without that choice.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .picking import (
    PickConfig, channel_aperture, find_data_file, load_channel_window, load_hammer_table,
    pick_arrival_channel_xcorr, read_das_metadata, review_band, review_window,
)

# Bonus added to a candidate's score for each kind of evidence it carries.
_SOURCE_BONUS = {"xcorr_auto": 0.10, "xcorr_global_min": 0.10, "xcorr_raw_global_min": 0.06}
# Never pruned by candidate_max_count.
_PROTECTED = {"xcorr_global_min", "xcorr_raw_global_min"}


@dataclass
class CandidateSet:
    timestamp: str
    channel_start: int               # recorded channel of data[:, 0]
    time_s: np.ndarray
    data: np.ndarray                 # (time, channel), band-passed
    amplitude_score: np.ndarray
    relative_lag_s: np.ndarray
    relative_lag_smooth_s: np.ndarray
    candidates: np.ndarray           # recorded channels, ascending
    candidate_score: np.ndarray
    candidate_sources: list
    auto_channel: float
    xcorr_global_min_channel: float
    xcorr_raw_global_min_channel: float
    band_hz: tuple
    data_file: str

    @property
    def channels(self) -> np.ndarray:
        return np.arange(self.channel_start, self.channel_start + self.data.shape[1])


def _unit_interval(v: np.ndarray) -> np.ndarray:
    """Map the 5th..95th percentile range onto 0..1."""
    a = np.asarray(v, dtype=float)
    out = np.zeros_like(a)
    m = np.isfinite(a)
    if not np.any(m):
        return out
    lo, hi = np.nanpercentile(a[m], [5, 95])
    if hi > lo:
        out[m] = np.clip((a[m] - lo) / (hi - lo), 0.0, 1.0)
    return out


def channel_amplitude_score(data: np.ndarray) -> np.ndarray:
    """Per-channel packet strength: 95th percentile of the envelope, normalized."""
    env = np.abs(signal.hilbert(np.nan_to_num(np.asarray(data, dtype=float)), axis=0))
    return uniform_filter1d(_unit_interval(np.nanpercentile(env, 95.0, axis=0)), size=3, mode="nearest")


def _peaks(score: np.ndarray, distance: int, prominence_fraction: float) -> np.ndarray:
    y = np.asarray(score, dtype=float)
    if y.size < 3 or not np.any(np.isfinite(y)):
        return np.array([], dtype=int)
    y = np.nan_to_num(y, nan=np.nanmin(y))
    prom = max(1e-8, prominence_fraction * max(float(np.ptp(y)), 1e-8))
    return signal.find_peaks(y, distance=max(1, distance), prominence=prom)[0]


def build_candidate_set(x, t, fs, channel_start, row, cfg: PickConfig, *, band_hz, data_file) -> CandidateSet:
    """Candidate channels for one hammer, best evidence first, then sorted by channel.

    The global minima of the raw and smoothed lag curves are always kept; the
    rest are pruned to candidate_max_count, and candidates within 2 channels of
    a kept one are merged into it.
    """
    n = x.shape[1]
    amp = channel_amplitude_score(x)
    lag = lag_s = np.full(n, np.nan)
    auto = np.nan
    try:
        xp = pick_arrival_channel_xcorr(x, t, fs, cfg)
        lag, lag_s, auto = xp.relative_lag_times_s, xp.relative_lag_times_smooth_s, xp.channel
    except (RuntimeError, ValueError):
        pass   # no coherent lag graph: the amplitude candidates still go on the waterfall

    sources: dict[int, set] = {}
    for i in _peaks(amp, cfg.candidate_peak_min_distance_channels, cfg.candidate_amplitude_prominence_fraction):
        sources.setdefault(int(i), set()).add("amplitude")
    lag_score = _unit_interval(-lag_s)
    smooth_min = raw_min = None
    if np.any(np.isfinite(lag_s)):
        for i in _peaks(lag_score, cfg.candidate_peak_min_distance_channels, cfg.candidate_lag_prominence_fraction):
            sources.setdefault(int(i), set()).add("xcorr_minimum")
        sources.setdefault(int(round(auto)), set()).add("xcorr_auto")
        smooth_min = int(np.nanargmin(np.where(np.isfinite(lag_s), lag_s, np.inf)))
        sources.setdefault(smooth_min, set()).add("xcorr_global_min")
    if np.any(np.isfinite(lag)):
        raw_min = int(np.nanargmin(np.where(np.isfinite(lag), lag, np.inf)))
        sources.setdefault(raw_min, set()).add("xcorr_raw_global_min")
    # Always offer at least a few channels, strongest packets first.
    for i in np.argsort(amp)[::-1]:
        if len(sources) >= min(4, cfg.candidate_max_count):
            break
        sources.setdefault(int(i), set()).add("strong_amplitude")

    ranked = []
    for i, src in sources.items():
        if 0 <= i < n:
            score = 0.60 * amp[i] + 0.40 * lag_score[i] + sum(_SOURCE_BONUS.get(s, 0.0) for s in src)
            score += 0.10 if len(src) >= 2 else 0.0
            ranked.append((float(score), i, "+".join(sorted(src))))
    ranked.sort(reverse=True)

    kept = []
    for item in [r for r in ranked if set(r[2].split("+")) & _PROTECTED]:
        near = [k for k, q in enumerate(kept) if abs(item[1] - q[1]) <= 2]
        if near:
            j = min(near, key=lambda k: abs(item[1] - kept[k][1]))
            sc0, i0, src0 = kept[j]
            kept[j] = (max(sc0, item[0]), i0, "+".join(sorted(set(src0.split("+")) | set(item[2].split("+")))))
        else:
            kept.append(item)
    for item in [r for r in ranked if not set(r[2].split("+")) & _PROTECTED]:
        if len(kept) >= cfg.candidate_max_count:
            break
        if all(abs(item[1] - q[1]) > 2 for q in kept):
            kept.append(item)
    kept.sort(key=lambda q: q[1])

    def recorded(i):
        return float(channel_start + i) if i is not None and np.isfinite(i) else float("nan")

    return CandidateSet(
        timestamp=str(row["timestamp"]), channel_start=int(channel_start),
        time_s=np.asarray(t, dtype=float), data=x, amplitude_score=amp,
        relative_lag_s=lag, relative_lag_smooth_s=lag_s,
        candidates=np.asarray([channel_start + q[1] for q in kept], dtype=int),
        candidate_score=np.asarray([q[0] for q in kept]),
        candidate_sources=[q[2] for q in kept],
        auto_channel=recorded(auto), xcorr_global_min_channel=recorded(smooth_min),
        xcorr_raw_global_min_channel=recorded(raw_min),
        band_hz=tuple(map(float, band_hz)), data_file=str(data_file),
    )


def prepare_candidate_set(row, path: str, cfg: PickConfig):
    """Load the hammer's window from `path` and build its candidates."""
    meta = read_das_metadata(path)
    band = review_band(row, meta.sample_rate_hz, cfg)
    t1, t2 = review_window(row, cfg)
    c1, c2 = channel_aperture(row, meta, cfg)
    x, t = load_channel_window(path, t1, t2, band, cfg, c1, c2)
    return build_candidate_set(x, t, meta.sample_rate_hz, c1, row, cfg, band_hz=band, data_file=path), meta


def plot_candidate_waterfall(c: CandidateSet, cfg: PickConfig, *, title: str | None = None):
    """Waterfall, packet strength and lag curve, with the candidates numbered."""
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.6, 1.2, 1.2], hspace=0.16)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    # Each trace scaled by its own 95th percentile; amplitude is in panel 2.
    z = np.nan_to_num(c.data).copy()
    scale = np.nanpercentile(np.abs(z), 95.0, axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    z /= scale
    hi = np.nanpercentile(np.abs(z), cfg.waterfall_clip_percentile)
    ch = c.channels
    ax0.imshow(z, aspect="auto", interpolation="nearest", cmap="gray", vmin=-hi, vmax=hi,
               extent=[ch[0] - 0.5, ch[-1] + 0.5, c.time_s[-1], c.time_s[0]])
    ax0.set_ylabel("time [s]")
    ax0.set_title(title or f"FG waterfall / first-arrival candidate review: {c.timestamp}")

    ax1.plot(ch, c.amplitude_score, lw=1.3)
    ax1.set_ylabel("packet\nstrength")
    ax1.grid(alpha=0.2)

    if np.any(np.isfinite(c.relative_lag_smooth_s)):
        ax2.plot(ch, c.relative_lag_smooth_s, lw=1.5, label="xcorr lag curve")
        ax2.scatter(ch, c.relative_lag_s, s=5, alpha=0.20)
    ax2.set_ylabel("relative lag\n[s]")
    ax2.set_xlabel("DAS acquisition channel")
    ax2.grid(alpha=0.2)

    ylab = c.time_s[0] + 0.045 * (c.time_s[-1] - c.time_s[0])
    for n, cand in enumerate(c.candidates, start=1):
        for ax in (ax0, ax1, ax2):
            ax.axvline(cand, ls="--", lw=0.9, alpha=0.8)
        ax0.text(cand, ylab, str(n), ha="center", va="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))
    smooth, raw = c.xcorr_global_min_channel, c.xcorr_raw_global_min_channel
    if np.isfinite(smooth):
        ax2.axvline(smooth, ls="--", lw=1.8, label=f"smoothed global min {smooth:.0f}")
    if np.isfinite(raw) and not (np.isfinite(smooth) and abs(raw - smooth) <= 1):
        ax2.axvline(raw, ls="-.", lw=1.4, label=f"raw global min {raw:.0f}")
    if np.isfinite(c.auto_channel):
        ax2.axvline(c.auto_channel, ls=":", lw=1.4, label=f"xcorr auto {c.auto_channel:.1f}")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"channels {ch[0]}:{ch[-1]}; FG {c.band_hz[0]:.1f}-{c.band_hz[1]:.1f} Hz; candidates: "
        + ", ".join(f"{i + 1}={v}" for i, v in enumerate(c.candidates)),
        y=0.995, fontsize=9,
    )
    return fig, (ax0, ax1, ax2)


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def generate_review_candidates(hammer_csv, cfg: PickConfig, out_dir, *, verbose: bool = True) -> list[dict]:
    """Candidate waterfalls and a review CSV for every hammer in `hammer_csv`.

    Writes out_dir/waterfall_candidates/*.png and out_dir/candidates.csv. A
    hammer whose raw record is not on disk keeps its row (and any
    `arrival_channel` already in the table) without candidates.
    """
    out = Path(out_dir)
    plot_dir = out / "waterfall_candidates"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for n, row in enumerate(load_hammer_table(hammer_csv), start=1):
        rec = {k: v for k, v in row.items() if k != "timestamp_dt"}
        rec.setdefault("arrival_channel", "")
        path = find_data_file(row, cfg)
        if path is None:
            if verbose:
                print(f"{n:2d}: {row['timestamp']}: no raw record on disk; kept pick "
                      f"{rec['arrival_channel'] or '(none)'}")
            rows_out.append(rec)
            continue
        c, meta = prepare_candidate_set(row, path, cfg)
        plot_path = plot_dir / f"waterfall_{n:03d}_{row['timestamp'].replace(':', '-')}.png"
        fig, _ = plot_candidate_waterfall(c, cfg)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rec.update({
            "number_of_loci": meta.number_of_loci,
            "review_channel_start": int(c.channels[0]),
            "review_channel_end_inclusive": int(c.channels[-1]),
            "xcorr_auto_channel": c.auto_channel,
            "xcorr_global_min_channel": c.xcorr_global_min_channel,
            "xcorr_raw_global_min_channel": c.xcorr_raw_global_min_channel,
            "candidate_plot": str(plot_path),
        })
        for i, (cand, src) in enumerate(zip(c.candidates, c.candidate_sources), start=1):
            rec[f"candidate_{i}"] = int(cand)
            rec[f"candidate_{i}_source"] = src
        rows_out.append(rec)
        if verbose:
            print(f"{n:2d}: {row['timestamp']}: review {c.channels[0]}:{c.channels[-1]}; "
                  f"candidates {c.candidates.tolist()}")
    _write_csv(out / "candidates.csv", rows_out)
    return rows_out


def _parse_choice(ans: str, candidates: list[int], aperture: tuple[int, int] | None):
    """Channel for a typed answer, or an error message.

    A bare number is a channel if it falls inside the review aperture and a
    candidate number otherwise; ``c512`` forces channel 512 and ``#3`` forces
    candidate 3.
    """
    force_channel, force_candidate = ans.startswith("c"), ans.startswith("#")
    token = ans[1:] if (force_channel or force_candidate) else ans
    try:
        num = int(round(float(token)))
    except ValueError:
        return None, "Invalid choice."
    in_aperture = aperture is not None and aperture[0] <= num <= aperture[1]
    if force_channel or (in_aperture and not force_candidate):
        return num, None
    if 1 <= num <= len(candidates):
        return candidates[num - 1], None
    return None, (f"{num} is neither a candidate number (1..{len(candidates)}) nor a channel in "
                  f"{aperture}. Use c{num} to force it.")


def review_candidates_interactive(candidates_csv, reviewed_csv, *, redo: bool = False,
                                  show_plots: bool = True) -> list[dict]:
    """Pick the arrival channel of each hammer from its waterfall.

    Hammers that already have an `arrival_channel` are skipped unless `redo`,
    in which case pressing Enter keeps the existing pick. Answers: a candidate
    number or a recorded channel, ``c<channel>``, ``#<candidate>``, ``s`` to
    leave the hammer unpicked, ``q`` to save and stop. Progress is saved after
    every hammer, so a session can be resumed.
    """
    with open(candidates_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        current = str(r.get("arrival_channel", "")).strip()
        if not r.get("candidate_plot") or (current and not redo):
            continue
        cands = [int(float(v)) for k, v in sorted(
            ((k, v) for k, v in r.items() if k.startswith("candidate_") and k[10:].isdigit() and str(v).strip()),
            key=lambda kv: int(kv[0][10:]))]
        aperture = (int(float(r["review_channel_start"])), int(float(r["review_channel_end_inclusive"])))
        if show_plots:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(plt.imread(r["candidate_plot"]))
            ax.axis("off")
            plt.show(block=False)
        print(f"\n{r['timestamp']}")
        print("Candidates:", "  ".join(f"{i + 1}:{v}" for i, v in enumerate(cands)))
        print(f"Review aperture: {aperture[0]}..{aperture[1]}" + (f"; current pick {current}" if current else ""))
        quit_now = False
        while True:
            ans = input("Candidate #, channel, c<channel>, #<candidate>, s=skip, q=save+quit"
                        + (", Enter=keep: " if current else ": ")).strip().lower()
            if ans == "" and current:
                break
            if ans == "q":
                quit_now = True
                break
            if ans == "s":
                r["arrival_channel"] = ""
                break
            channel, err = _parse_choice(ans, cands, aperture)
            if err:
                print(err)
                continue
            r["arrival_channel"] = channel
            print(f"  -> channel {channel}")
            break
        if show_plots:
            plt.close("all")
        _write_csv(Path(reviewed_csv), rows)
        if quit_now:
            return rows
    _write_csv(Path(reviewed_csv), rows)
    return rows
