from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from .common import (
    slice_distance_to_channel,
    make_time,
    standardize_dataset,
)


def _parse_onyx_start_time(path: Path) -> pd.Timestamp:
    name = path.name
    time_str = name[10:-7][:19]
    dt = datetime.strptime(time_str, "%Y-%m-%d_%H.%M.%S")
    return pd.Timestamp(dt)


def open_onyx(
    path,
    engine: str = "h5py",
    time_slice=None,
    channel_slice=None,
    distance_range=None,
):
    if engine == "h5py":
        return _open_onyx_h5py(
            path,
            time_slice=time_slice,
            channel_slice=channel_slice,
            distance_range=distance_range,
        )

    raise ValueError(f"Unknown engine: {engine}")


def _open_onyx_h5py(
    path,
    time_slice=None,
    channel_slice=None,
    distance_range=None,
    GL = 4.786,
):
    path = Path(path)

    with h5py.File(path, "r") as f:
        acq = f["Acquisition"]
        raw_group = acq["Raw[0]"]
        raw_dset = raw_group["RawData"]

        fs = float(raw_group.attrs.get("OutputDataRate", acq.attrs.get("PulseRate")))
        dx = float(acq.attrs["SpatialSamplingInterval"])

        if distance_range is not None:
            channel_slice = slice_distance_to_channel(distance_range, dx)

        time_slice = time_slice or slice(None)
        channel_slice = channel_slice or slice(None)

        data = raw_dset[time_slice, channel_slice]

        attrs = dict(acq.attrs)
        attrs.update(dict(raw_group.attrs))
        attrs.update(dict(raw_dset.attrs))

        full_shape = raw_dset.shape

    n_eff = 1.4682
    wl_vac = 1550.12e-9
    eps = 0.78 # strain optic coefficient
    raw_to_strain = wl_vac/(4*GL*n_eff*np.pi*eps)

    data *= raw_to_strain*1e9 # (nm/m)

    n_time_total, n_ch_total = full_shape

    full_channel = np.arange(n_ch_total)
    channel = full_channel[channel_slice]
    distance = channel * dx

    start_time = _parse_onyx_start_time(path)
    full_time = make_time(start_time, fs, n_time_total)
    time = full_time[time_slice]

    attrs={
        "instrument": "onyx",
        "fs": fs,
        "dx": dx,
        "source_file": str(path),
        "n_ch_total": n_ch_total
    }
    
    return standardize_dataset(
        data_array=(("time", "channel"), data),
        data_name="strain",
        time=time,
        channel=channel,
        distance=distance,
        attrs=attrs,
    )
