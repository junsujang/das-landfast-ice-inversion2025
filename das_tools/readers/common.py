import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path


def slice_distance_to_channel(distance_range, dx):
    d0, d1 = distance_range
    i0 = int(np.floor(d0 / dx))
    i1 = int(np.ceil(d1 / dx))
    return slice(i0, i1)


def make_time(start_time, fs, n_time):
    return pd.date_range(
        start=start_time,
        periods=n_time,
        freq=pd.to_timedelta(1.0 / fs, unit="s"),
    )


def standardize_dataset(
    data_array,
    *,
    data_name,
    time,
    channel,
    distance,
    attrs,
):
    ds = xr.Dataset(
        data_vars={data_name: data_array},
        coords={
            "time": time,
            "channel": channel,
            "distance": ("channel", distance),
        },
        attrs=attrs,
    )

    return ds


def list_h5_files(folder: Path) -> list[Path]:
    return sorted(Path(folder).glob("*.h5"))

