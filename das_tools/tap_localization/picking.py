"""Load a tap-test window from the raw DAS record and estimate the closest channel.

A hammer tap on the ice is heard first by the channel nearest to it, so the
relative arrival time across channels has a minimum there. The picker builds that
relative-arrival curve from cross-correlations between channel pairs (a sparse
lag graph solved by robust least squares) and takes its minimum. It only
proposes candidates; the channel used for the cable geometry is chosen by a
person looking at the waterfall (see `review`).

Each hammer row supplies the window: `t1FG..t2FG` (s from the file start) and
the `f1FG..f2FG` band of the flexural-gravity packet, plus `d1/d2`, the
approximate distance along the fibre (m) of the channels to search.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy import signal
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

from das_tools.processing import filter_dataset
from das_tools.readers import open_das

from .epochs import Rules


@dataclass
class PickConfig:
    """Where the raw records are and how the review windows are built.

    The defaults are the settings of the reviewed picks in data/tapTestLoc_picks.csv.
    """

    data_glob: str
    channel_epoch_shift_rules: Rules = ()
    file_time_format: str = "%Y-%m-%d_%H.%M.00"   # Onyx file names carry the minute

    # Causal band-pass: zero-phase filtering would smear energy ahead of the
    # first arrival and bias every pick early.
    filter_order: int = 4
    filter_zero_phase: bool = False
    fallback_band_hz: tuple[float, float] = (5.0, 80.0)
    window_padding_s: float = 0.0

    # Review aperture from the row's d1/d2, padded, and never below
    # minimum_review_channel.
    d1_d2_channel_padding: int = 6
    minimum_review_channel: int = 100
    maximum_review_channel: int | None = None

    # Lag graph.
    xcorr_channel_offsets: tuple[int, ...] = (1, 2, 4, 8, 16)
    xcorr_max_lag_samples: int = 500
    xcorr_min_peak_corr: float = 0.10
    xcorr_min_valid_pairs: int = 25
    xcorr_huber_scale_s: float = 0.020
    xcorr_irls_iterations: int = 5
    xcorr_lag_curve_smooth_channels: int = 11
    xcorr_minimum_refine_halfwidth_channels: int = 3
    xcorr_minimum_edge_guard_channels: int = 8

    # Candidates shown on each review waterfall.
    candidate_max_count: int = 10
    candidate_peak_min_distance_channels: int = 7
    candidate_amplitude_prominence_fraction: float = 0.08
    candidate_lag_prominence_fraction: float = 0.06
    waterfall_clip_percentile: float = 99.0


@dataclass
class DASMetadata:
    path: str
    number_of_loci: int
    sample_rate_hz: float
    spatial_sampling_m: float


@dataclass
class ArrivalPick:
    channel: float                    # sub-channel minimum of the smoothed lag curve
    quality: float                    # 0..1, a rough confidence only
    rmse_s: float
    relative_lag_times_s: np.ndarray = field(default_factory=lambda: np.array([]))
    relative_lag_times_smooth_s: np.ndarray = field(default_factory=lambda: np.array([]))
    n_valid_pairs: int = 0


# --------------------------------------------------------------------------
# hammer table and raw files
# --------------------------------------------------------------------------

_FLOAT_COLUMNS = {"lat", "lon", "d1", "d2", "t1FG", "t2FG", "f1FG", "f2FG",
                  "number_of_loci", "arrival_channel"}


def parse_timestamp(value: str) -> datetime:
    value = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")


def load_hammer_table(path) -> list[dict]:
    """Rows of a hammer CSV with numeric columns as floats and `timestamp_dt` added."""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            r = dict(row)
            r["timestamp_dt"] = parse_timestamp(r["timestamp"])
            for k in _FLOAT_COLUMNS & set(r):
                if str(r[k]).strip():
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def find_data_file(row, cfg: PickConfig) -> str | None:
    """The raw record holding this hammer, or None if it is not on disk."""
    stamp = row["timestamp_dt"].strftime(cfg.file_time_format)
    matches = [p for p in sorted(glob.glob(cfg.data_glob)) if stamp in os.path.basename(p)]
    if len(matches) > 1:
        raise RuntimeError(f"Several DAS files match {stamp!r}:\n" + "\n".join(matches))
    return matches[0] if matches else None


def read_das_metadata(path: str) -> DASMetadata:
    """Channel count, sample rate and channel spacing, reading one time sample."""
    ds = open_das(path, instrument="onyx", engine="h5py", time_slice=slice(0, 1))
    return DASMetadata(
        path=str(path),
        number_of_loci=int(ds.attrs.get("n_ch_total", ds.sizes["channel"])),
        sample_rate_hz=float(ds.attrs["fs"]),
        spatial_sampling_m=float(ds.attrs["dx"]),
    )


def review_window(row, cfg: PickConfig) -> tuple[float, float]:
    """[t1FG, t2FG] in s from the file start, padded by window_padding_s."""
    t1, t2 = float(row["t1FG"]), float(row["t2FG"])
    return max(0.0, t1 - cfg.window_padding_s), t2 + cfg.window_padding_s


def review_band(row, fs: float, cfg: PickConfig) -> tuple[float, float]:
    """[f1FG, f2FG] kept inside (0.5 Hz, 0.95 Nyquist), or the fallback band."""
    try:
        lo, hi = float(row["f1FG"]), float(row["f2FG"])
    except (KeyError, TypeError, ValueError):
        lo, hi = cfg.fallback_band_hz
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        lo, hi = cfg.fallback_band_hz
    nyq = 0.5 * fs
    lo = max(0.5, min(lo, 0.80 * nyq))
    hi = min(hi, 0.95 * nyq)
    if hi <= 1.05 * lo:
        lo, hi = max(0.5, 0.05 * nyq), 0.80 * nyq
    return lo, hi


def channel_aperture(row, meta: DASMetadata, cfg: PickConfig) -> tuple[int, int]:
    """[start, end) recorded channels to review, from the row's d1/d2 distances."""
    c1, c2 = 0, meta.number_of_loci
    d1, d2 = float(row.get("d1", np.nan)), float(row.get("d2", np.nan))
    if np.isfinite(d1) and np.isfinite(d2) and d2 > d1:
        c1 = int(np.floor(d1 / meta.spatial_sampling_m)) - cfg.d1_d2_channel_padding
        c2 = int(np.ceil(d2 / meta.spatial_sampling_m)) + cfg.d1_d2_channel_padding + 1
    c1 = max(0, c1, cfg.minimum_review_channel)
    if cfg.maximum_review_channel is not None:
        c2 = min(c2, cfg.maximum_review_channel + 1)
    c2 = min(c2, meta.number_of_loci)
    if c2 <= c1:
        # d1/d2 fell entirely below the minimum channel: review everything above it.
        c1, c2 = cfg.minimum_review_channel, meta.number_of_loci
    return c1, c2


def load_channel_window(path: str, t1: float, t2: float, band: tuple[float, float],
                        cfg: PickConfig, channel_start: int, channel_end: int):
    """Band-passed strain (time, channel) and t (s from file start) over [t1, t2].

    The whole record is filtered before trimming, so the filter's start-up
    transient does not sit on top of the first arrival.
    """
    ds = open_das(path, instrument="onyx", engine="h5py",
                  channel_slice=slice(channel_start, channel_end))
    fs = float(ds.attrs["fs"])
    ds = filter_dataset(ds, var="strain", fs=fs, fmin=band[0], fmax=min(band[1], 0.99 * 0.5 * fs),
                        order=cfg.filter_order, zero_phase=cfg.filter_zero_phase)
    x = np.asarray(ds["strain"].values, dtype=float)
    t = np.arange(x.shape[0], dtype=float) / fs
    keep = (t >= t1) & (t <= t2)
    return x[keep], t[keep]


# --------------------------------------------------------------------------
# cross-correlation lag graph
# --------------------------------------------------------------------------

def _standardized_envelope(data: np.ndarray) -> np.ndarray:
    x = np.abs(signal.hilbert(np.nan_to_num(np.asarray(data, dtype=float)), axis=0))
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return x / sd


def _best_lag_and_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    n = a.size
    corr = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(n, n, mode="full")
    m = np.abs(lags) <= max_lag
    k = int(np.argmax(corr[m]))
    lag = int(lags[m][k])
    # Inputs are standardized, so dividing by the overlap gives a Pearson-like score.
    return lag, float(np.clip(corr[m][k] / max(1, n - abs(lag)), -1.0, 1.0))


def sparse_channel_lags(data: np.ndarray, cfg: PickConfig):
    """Lag (samples) and peak correlation for channel pairs at the configured offsets.

    Several offsets tie distant channels together directly, so the solution does
    not drift the way integrating adjacent-channel lags would.
    """
    x = _standardized_envelope(data)
    n_ch = x.shape[1]
    ii, jj, ll, cc = [], [], [], []
    for off in sorted({o for o in cfg.xcorr_channel_offsets if 0 < o < n_ch}):
        for i in range(n_ch - off):
            lag, peak = _best_lag_and_corr(x[:, i], x[:, i + off], cfg.xcorr_max_lag_samples)
            ii.append(i); jj.append(i + off); ll.append(lag); cc.append(peak)
    return np.asarray(ii), np.asarray(jj), np.asarray(ll, float), np.asarray(cc, float)


def relative_arrival_times(n_channels: int, pi, pj, lag, corr, fs: float, cfg: PickConfig):
    """Robust least-squares relative arrival time per channel from the lag graph.

    SciPy's ``correlate(a, b)`` peaks at ``lag = -fs (tau_b - tau_a)``, so each
    accepted pair says ``tau_j - tau_i = -lag / fs``. Pairs below
    xcorr_min_peak_corr are dropped; if that splits the graph, only the largest
    connected region is solved and the other channels are NaN, so incoherent
    channels cannot become false minima. Returns (tau, per-channel mean
    correlation, weighted rms residual, number of pairs used).
    """
    good = np.isfinite(lag) & np.isfinite(corr) & (corr >= cfg.xcorr_min_peak_corr)
    pi, pj, lag, corr = pi[good], pj[good], lag[good], corr[good]
    if pi.size < 3:
        raise RuntimeError(f"Too few coherent channel pairs ({pi.size})")

    graph = coo_matrix((np.ones(2 * pi.size), (np.r_[pi, pj], np.r_[pj, pi])),
                       shape=(n_channels, n_channels)).tocsr()
    ncomp, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=ncomp)
    regions = np.flatnonzero(counts >= 3)
    if regions.size == 0:
        raise RuntimeError("No coherent region of >= 3 channels")
    # Widest region first, then the most summed correlation.
    chosen = max((counts[c], float(np.sum(np.clip(corr[(labels[pi] == c) & (labels[pj] == c)], 0, None))), c)
                 for c in regions)[2]
    nodes = np.flatnonzero(labels == chosen)
    inside = (labels[pi] == chosen) & (labels[pj] == chosen)
    pi, pj, lag, corr = pi[inside], pj[inside], lag[inside], corr[inside]
    m = pi.size

    local = -np.ones(n_channels, dtype=int)
    local[nodes] = np.arange(nodes.size)
    A = coo_matrix((np.tile([-1.0, 1.0], m),
                    (np.repeat(np.arange(m), 2), np.column_stack([local[pi], local[pj]]).ravel())),
                   shape=(m, nodes.size)).tocsr()
    b = -lag / float(fs)

    # IRLS with Huber weights on top of correlation-strength weights; one
    # heavily weighted row fixes the free constant.
    base_w = np.clip((corr - cfg.xcorr_min_peak_corr) / max(1e-9, 1.0 - cfg.xcorr_min_peak_corr), 0.02, 1.0) ** 2
    w = base_w.copy()
    gauge = csr_matrix(([100.0], ([0], [0])), shape=(1, nodes.size))
    for _ in range(max(1, cfg.xcorr_irls_iterations)):
        sw = np.sqrt(w)
        tau_local = lsqr(vstack([diags(sw) @ A, gauge], format="csr"), np.r_[sw * b, 0.0],
                         atol=1e-10, btol=1e-10, iter_lim=5000)[0]
        r = np.abs(A @ tau_local - b)
        scale = max(cfg.xcorr_huber_scale_s, 1e-6)
        w = base_w * np.where(r > scale, scale / np.maximum(r, 1e-30), 1.0)

    r = A @ tau_local - b
    rmse = float(np.sqrt(np.average(r * r, weights=np.maximum(base_w, 1e-6))))
    tau = np.full(n_channels, np.nan)
    tau[nodes] = tau_local - np.min(tau_local)
    qsum = np.bincount(np.r_[pi, pj], weights=np.r_[corr, corr], minlength=n_channels)
    qn = np.bincount(np.r_[pi, pj], minlength=n_channels)
    q = np.divide(qsum, qn, out=np.zeros(n_channels), where=qn > 0)
    return tau, q, rmse, int(m)


def _smooth_lag_curve(tau: np.ndarray, window_channels: int) -> np.ndarray:
    """Gap-filled Savitzky-Golay smoothing over the solved channel range."""
    out = np.full(tau.size, np.nan)
    finite = np.flatnonzero(np.isfinite(tau))
    if finite.size == 0:
        return out
    idx = np.arange(finite[0], finite[-1] + 1)
    yy = np.interp(idx, finite, tau[finite])
    if yy.size >= 5:
        win = max(5, int(window_channels)) | 1
        win = min(win, yy.size if yy.size % 2 else yy.size - 1)
        if win >= 5:
            yy = signal.savgol_filter(yy, window_length=win, polyorder=min(3, win - 2), mode="interp")
    out[idx] = yy
    return out


def _minimum_channel(curve: np.ndarray) -> float:
    """Discrete minimum, refined by a parabola through it and its two neighbours."""
    c0 = int(np.nanargmin(np.where(np.isfinite(curve), curve, np.inf)))
    if 0 < c0 < curve.size - 1 and np.all(np.isfinite(curve[c0 - 1:c0 + 2])):
        ym, y0, yp = curve[c0 - 1:c0 + 2]
        denom = ym - 2.0 * y0 + yp
        if denom > 0:
            dz = 0.5 * (ym - yp) / denom
            if abs(dz) <= 1.0:
                return float(c0 + dz)
    return float(c0)


def pick_arrival_channel_xcorr(x: np.ndarray, t: np.ndarray, fs: float, cfg: PickConfig) -> ArrivalPick:
    """Channel (index into x's columns) where the relative arrival time is smallest."""
    if x.ndim != 2 or x.shape[0] != t.size:
        raise ValueError("x must be (time, channel) and match t")
    if x.shape[0] < 8 or x.shape[1] < 5:
        raise ValueError("Window too short or too few channels for the lag graph")

    pi, pj, lag, corr = sparse_channel_lags(x, cfg)
    tau, channel_q, rmse, n_pairs = relative_arrival_times(x.shape[1], pi, pj, lag, corr, fs, cfg)
    smooth = _smooth_lag_curve(tau, cfg.xcorr_lag_curve_smooth_channels)
    center = _minimum_channel(smooth)

    finite = np.flatnonzero(np.isfinite(smooth))
    edge = float(min(center - finite[0], finite[-1] - center)) if finite.size else 0.0
    qgood = channel_q[channel_q > 0]
    quality = (
        (float(np.median(qgood)) if qgood.size else 0.0)
        * float(np.exp(-rmse / max(cfg.xcorr_huber_scale_s, 1e-6)))
        * min(1.0, n_pairs / max(float(cfg.xcorr_min_valid_pairs), 0.5 * x.shape[1]))
        * min(1.0, max(0.1, edge / max(1.0, float(cfg.xcorr_minimum_edge_guard_channels))))
    )
    return ArrivalPick(channel=center, quality=float(np.clip(quality, 0.0, 1.0)), rmse_s=rmse,
                       relative_lag_times_s=tau, relative_lag_times_smooth_s=smooth,
                       n_valid_pairs=n_pairs)
