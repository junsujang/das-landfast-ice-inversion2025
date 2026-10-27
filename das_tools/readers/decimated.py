from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .common import standardize_dataset, list_h5_files


def open_decimated(path, time_slice=None, channel_slice=None):
    """Open an hourly decimated h5 file written by decimate_merge.save_hour_block."""
    path = Path(path)

    with h5py.File(path, "r") as f:
        time_slice = time_slice or slice(None)
        channel_slice = channel_slice or slice(None)

        data = f["strain"][time_slice, channel_slice]
        time_ns = f["time"][time_slice]
        channel = f["channel"][channel_slice]
        distance = f["distance"][channel_slice]

        attrs = dict(f.attrs)

    time = pd.to_datetime(time_ns, unit="ns", utc=True)

    attrs["instrument"] = "decimated"
    attrs["source_file"] = str(path)

    return standardize_dataset(
        data_array=(("time", "channel"), data),
        data_name="strain",
        time=time,
        channel=channel,
        distance=distance,
        attrs=attrs,
    )


def index_hourly_files(input_dir: Path) -> dict[pd.Timestamp, Path]:
    """Map each hourly file's start time (floored to the hour) to its path."""
    out: dict[pd.Timestamp, Path] = {}
    for path in list_h5_files(input_dir):
        with h5py.File(path, "r") as f:
            start = pd.Timestamp(f.attrs["start_time"])
        out[start.floor("h")] = path
    return out


def load_block(hourly_map: dict[pd.Timestamp, Path], block_start: pd.Timestamp, block_end: pd.Timestamp):
    """Concatenate hourly files covering [block_start, block_end) and slice to it.

    Returns (data (T, M), channel (M,), fs), or None if any needed hour is missing.
    Channel offsets are always relative to that hour's bend/elbow, so the same
    offset means the same physical position relative to the elbow at any time --
    no per-block geometry tracking needed here.
    """
    hour = block_start.floor("h")
    hours_needed = []
    while hour < block_end:
        hours_needed.append(hour)
        hour += pd.Timedelta(hours=1)

    paths = [hourly_map.get(h) for h in hours_needed]
    if any(p is None for p in paths):
        return None

    datasets = [open_decimated(p) for p in paths]

    fs = float(datasets[0].attrs["fs"])
    channel = datasets[0]["channel"].values

    data = np.concatenate([ds["strain"].values for ds in datasets], axis=0)

    offset_samples = round((block_start - hours_needed[0]).total_seconds() * fs)
    n_samples = round((block_end - block_start).total_seconds() * fs)
    block = data[offset_samples: offset_samples + n_samples]

    if block.shape[0] != n_samples:
        return None

    return block, channel, fs
