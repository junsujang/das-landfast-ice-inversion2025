"""DAS spectrogram from the 5 Hz decimated hourly files, for every channel they keep.

The record is cut into blocks on a fixed clock grid (default 30 min, starting on
the hour and half hour). Each block is split into Welch segments (default 3 min:
Hann window, no overlap, mean removed); a segment containing any NaN -- a gap in
the record -- is dropped and the remaining ones are averaged. A block is kept if
at least --min-segments of them are complete (default 1), and `n_segments`
records how many went in, so a plot can mask blocks that rest on few segments.
One segment gives a noisier spectrum than ten, but a short stretch of good data
is no longer thrown away.

The channel axis is the decimated files' offset from the cable bend (-45..44, about
-144 to +140 m). The files move that window with the bend, so the same offset is
the same position relative to the bend at any time and a block spanning a bend
change needs no special handling.

Writes, under --results-dir (default outputs/das_spectrogram):
    das_spectrogram.npz
        block_start      (B,)       block start, ISO UTC
        freq_hz          (F,)       0 .. --fmax
        channel_offset   (C,)       channel offset from the bend
        offset_m         (C,)       the same in metres
        psd_db           (B, F, C)  10 log10 PSD of strain, (nm/m)^2/Hz, float32
        n_segments       (B,)       complete segments averaged into each block
        n_segments_max, segment_s, block_min, fs_hz

    python main_spectrogram.py
    python main_spectrogram.py --start 2025-04-26 --end 2025-04-27 --min-segments 3
See slurms/spectrogram.slurm for the server run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from das_tools.readers import decimated
from das_tools.utils import proj_path
from das_tools.utils.helpers import savez_atomic


def block_psd(block: np.ndarray, fs: float, seg_n: int, keep: np.ndarray, min_segments: int):
    """Mean PSD (F, C) over the NaN-free segments of a (T, C) block, or None.

    Returns (psd, n_segments).
    """
    n_seg = block.shape[0] // seg_n
    segs = block[: n_seg * seg_n].reshape(n_seg, seg_n, block.shape[1])
    complete = np.all(np.isfinite(segs), axis=(1, 2))
    if complete.sum() < min_segments:
        return None, int(complete.sum())
    _, pxx = signal.welch(segs[complete], fs=fs, window="hann", nperseg=seg_n, noverlap=0,
                          detrend="constant", scaling="density", axis=1)
    return pxx[:, keep, :].mean(axis=0), int(complete.sum())


def main(args) -> None:
    # The files' start_time attribute is naive UTC.
    hourly_map = {(h.tz_localize("UTC") if h.tzinfo is None else h): p
                  for h, p in decimated.index_hourly_files(args.data_dir).items()}
    if not hourly_map:
        raise FileNotFoundError(f"No hourly files in {args.data_dir}")
    hours = sorted(h for h in hourly_map
                   if (args.start is None or h >= args.start) and (args.end is None or h < args.end))
    print(f"{len(hours)} hourly files in {args.data_dir}")

    block = pd.Timedelta(minutes=args.block_min)
    if pd.Timedelta(hours=1) % block != pd.Timedelta(0):
        raise ValueError("--block-min must divide an hour")

    channel = fs = freq = keep = seg_n = None
    starts, psds, counts = [], [], []
    n_skipped = 0
    for hour in hours:
        ds = decimated.open_decimated(hourly_map[hour])
        if channel is None:
            channel = ds["channel"].values
            fs = float(ds.attrs["fs"])
            seg_n = int(round(args.segment_s * fs))
            freq = np.fft.rfftfreq(seg_n, d=1.0 / fs)
            keep = freq <= args.fmax + 1e-9
            dx = float(ds.attrs["dx"])
        elif not np.array_equal(ds["channel"].values, channel) or float(ds.attrs["fs"]) != fs:
            raise ValueError(f"{hourly_map[hour].name}: channel axis or fs differs from the first file")

        x = np.asarray(ds["strain"].values, dtype=float)
        t0 = pd.Timestamp(ds.attrs["start_time"])
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        n_block = int(round(block.total_seconds() * fs))
        n_kept = 0
        for b0 in pd.date_range(hour, hour + pd.Timedelta(hours=1), freq=block, inclusive="left"):
            i0 = int(round((b0 - t0).total_seconds() * fs))
            # Samples the file does not hold count as missing.
            blk = np.full((n_block, x.shape[1]), np.nan)
            lo, hi = max(i0, 0), min(i0 + n_block, x.shape[0])
            if hi > lo:
                blk[lo - i0: hi - i0] = x[lo:hi]
            psd, n = block_psd(blk, fs, seg_n, keep, args.min_segments)
            if psd is None:
                n_skipped += 1
                continue
            starts.append(b0.isoformat())
            psds.append(10.0 * np.log10(np.maximum(psd, 1e-30)).astype(np.float32))
            counts.append(n)
            n_kept += 1
        print(f"{hour:%Y-%m-%d %H:%M}: {n_kept} blocks")

    if not psds:
        raise RuntimeError("No block had enough complete segments")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "das_spectrogram.npz"
    counts = np.asarray(counts)
    savez_atomic(
        out_path,
        block_start=np.asarray(starts),
        freq_hz=freq[keep],
        channel_offset=channel,
        offset_m=channel * dx,
        psd_db=np.stack(psds),
        n_segments=counts,
        n_segments_max=np.int64(n_block // seg_n),
        segment_s=float(args.segment_s),
        block_min=float(args.block_min),
        fs_hz=fs,
        settings_json=json.dumps({k: str(v) for k, v in vars(args).items()}),
    )
    print(f"\n{len(starts)} blocks written to {out_path} ({n_skipped} blocks with fewer than "
          f"{args.min_segments} complete segments skipped)")
    print("segments per block:", dict(zip(*np.unique(counts, return_counts=True))))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Hourly 5 Hz decimated h5 files (default: proj_path.hourly_das_dir())")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Output directory (default: outputs/das_spectrogram)")
    p.add_argument("--block-min", type=float, default=30.0, help="Block length (min)")
    p.add_argument("--segment-s", type=float, default=180.0, help="Welch segment length (s)")
    p.add_argument("--min-segments", type=int, default=1,
                   help="Fewest complete segments for a block to be kept")
    p.add_argument("--fmax", type=float, default=1.0, help="Highest frequency kept (Hz)")
    p.add_argument("--start", type=pd.Timestamp, default=None, help="First hour, UTC (inclusive)")
    p.add_argument("--end", type=pd.Timestamp, default=None, help="Last hour, UTC (exclusive)")
    return p


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.data_dir = args.data_dir or proj_path.hourly_das_dir()
    args.results_dir = args.results_dir or proj_path.output_dir("das_spectrogram")
    for k in ("start", "end"):
        v = getattr(args, k)
        if v is not None and v.tzinfo is None:
            setattr(args, k, v.tz_localize("UTC"))
    main(args)
