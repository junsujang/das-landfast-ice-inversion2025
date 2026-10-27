"""PISCES 2025 deployment specifics: the bend (hairpin apex) schedule and channel geometry.

The cable is a hairpin and the ice moves, so the channel index of the bend apex
drifts through the experiment. The bend schedule CSV records every change of that
index (`mid_idx`) with its time; decimate_merge uses it to keep a fixed window of
channels around the bend, and `load_leg_geometry` gives that window's coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

def load_bend_schedule(csv_path: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(csv_path)

    required = {"change_time", "mid_idx", "global_short_leg_nch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in bend csv: {sorted(missing)}"
        )

    df = df.copy()
    df["change_time"] = pd.to_datetime(df["change_time"], utc=True)
    df = df.sort_values("change_time").reset_index(drop=True)

    int_cols = [
        "mid_idx",
        "short_leg_nch",
        "global_short_leg_nch",
        "start_idx",
        "end_idx",
        "global_start_idx",
        "global_end_idx",
        "left_leg_nch",
        "right_leg_nch",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    global_short_leg_values = np.unique(df["global_short_leg_nch"].to_numpy())
    if global_short_leg_values.size != 1:
        raise ValueError(
            f"Expected exactly one global_short_leg_nch in CSV, got {global_short_leg_values}"
        )

    global_short_leg_nch = int(global_short_leg_values[0])
    return df, global_short_leg_nch


def lookup_bend_state(schedule_df: pd.DataFrame, block_dt: pd.Timestamp) -> dict:
    """Schedule row in force at ``block_dt`` (the first row if it predates them all)."""
    if schedule_df.empty:
        raise ValueError("Schedule dataframe is empty")

    active = schedule_df[schedule_df["change_time"] <= block_dt]

    if active.empty:
        active_row = schedule_df.iloc[0]
    else:
        active_row = active.iloc[-1]

    out = {
        "change_time": active_row["change_time"],
        "mid_idx": int(active_row["mid_idx"]),
        "global_short_leg_nch": int(active_row["global_short_leg_nch"]),
    }

    for col in [
        "short_leg_nch",
        "start_idx",
        "end_idx",
        "global_start_idx",
        "global_end_idx",
        "left_leg_nch",
        "right_leg_nch",
    ]:
        if col in active_row.index:
            out[col] = int(active_row[col])

    return out


def load_leg_geometry(geo_path: Path, channel: np.ndarray) -> tuple[np.ndarray, int]:
    """(ch_xy, m1) for the bend-relative channel axis of the hourly files.

    `geo_path` is the leg geometry file (channel_geometry_*_L.npz): ch_xy for each
    `channel_offset` from the bend that decimate_merge keeps, in the same order.
    decimate_merge slides its channel window with the bend, so this one geometry
    holds for every hour. The axis runs [leg 1, leg 2]: leg 1 is the m1 channels
    before the bend (offset < 0), leg 2 starts at the bend channel.
    """
    geo = np.load(geo_path)
    offset = geo["channel_offset"]
    if not np.array_equal(offset, channel):
        raise ValueError(
            f"{geo_path} covers channel offsets {offset[0]}..{offset[-1]}, but the "
            f"data hold {channel[0]}..{channel[-1]}"
        )
    return geo["ch_xy"], int(np.sum(offset < 0))


def retrieve_wind_swift(file_path):

    # Open dataset
    ds = xr.open_dataset(file_path)

    # Define time window in UTC
    start_time = pd.Timestamp("2025-04-23 00:00:00")
    end_time   = pd.Timestamp("2025-05-08 00:00:00")

    # Subset dataset by time
    ds_sel = ds.sel(time=slice(start_time, end_time))

    # Extract variables
    time_w = ds_sel["time"].values
    wind_speed = ds_sel["wind_speed"].values
    
    # In case these variables are relevant
    # wind_dir = ds_sel["winddirR"].values   # use this for wind direction
    # time_t = time_w
    # air_temp = ds_sel["air_temperature"].values

    return time_w, wind_speed


def retrieve_wind_ARM(file_path):
    # ============================================================
    # 2) LOAD WIND
    #    (from mainPlotWind.ipynb)
    # ============================================================
    # wind file
    start_time = pd.Timestamp("2025-04-23 00:00:00")
    end_time   = pd.Timestamp("2025-05-08 00:00:00")

    wind_df = pd.read_csv(file_path)

    time_col = "Timestamp"
    wspd_col = "ARM - nsametC1.b1 (nsametC1.b1) / Wind speed vector mean (wspd_vec_mean)"

    wind_df[time_col] = pd.to_datetime(wind_df[time_col])
    wind_df = wind_df.rename(columns={
        time_col: "time",
        wspd_col: "wind_speed",
    })
    
    wind_df = wind_df[["time", "wind_speed"]].copy()
    wind_df = wind_df[(wind_df["time"] >= start_time) & (wind_df["time"] <= end_time)].copy()
    wind_df = wind_df.sort_values("time")
    wind_speed = wind_df["wind_speed"]
    time_w = wind_df["time"]

    return time_w, wind_speed
