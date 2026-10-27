import numpy as np
import xarray as xr
from scipy.signal import butter, sosfilt, sosfiltfilt


def make_sos_filter(
    fs,
    *,
    fmin=0.0,
    fmax=0.0,
    order=4,
    btype=None,
):
    """
    Create a Butterworth SOS filter.

    Rules
    -----
    fmin > 0 and fmax > 0  -> bandpass
    fmin == 0 and fmax > 0 -> lowpass
    fmin > 0 and fmax == 0 -> highpass

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Lower cutoff frequency in Hz. Use 0 for no lower cutoff.
    fmax : float
        Upper cutoff frequency in Hz. Use 0 for no upper cutoff.
    order : int
        Butterworth filter order.
    btype : str, optional
        Explicit scipy filter type. Usually leave as None.

    Returns
    -------
    sos : ndarray
        Second-order sections filter coefficients.
    resolved_btype : str
        One of "lowpass", "highpass", or "bandpass".
    """
    fs = float(fs)
    fmin = float(fmin)
    fmax = float(fmax)

    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    nyq = fs / 2.0

    if btype is None:
        if fmin <= 0 and fmax <= 0:
            raise ValueError("At least one of fmin or fmax must be positive.")
        if fmin <= 0 and fmax > 0:
            btype = "lowpass"
            cutoff = fmax
        elif fmin > 0 and fmax <= 0:
            btype = "highpass"
            cutoff = fmin
        else:
            btype = "bandpass"
            cutoff = [fmin, fmax]
    else:
        btype = btype.lower()
        if btype in {"low", "lowpass"}:
            btype = "lowpass"
            cutoff = fmax
        elif btype in {"high", "highpass"}:
            btype = "highpass"
            cutoff = fmin
        elif btype in {"band", "bandpass"}:
            btype = "bandpass"
            cutoff = [fmin, fmax]
        else:
            raise ValueError(f"Unknown btype: {btype}")

    if btype == "lowpass":
        if not 0 < cutoff < nyq:
            raise ValueError(f"Lowpass cutoff must satisfy 0 < fmax < fs/2. Got fmax={cutoff}, fs={fs}")

    elif btype == "highpass":
        if not 0 < cutoff < nyq:
            raise ValueError(f"Highpass cutoff must satisfy 0 < fmin < fs/2. Got fmin={cutoff}, fs={fs}")

    elif btype == "bandpass":
        lo, hi = cutoff
        if not 0 < lo < hi < nyq:
            raise ValueError(
                f"Bandpass cutoffs must satisfy 0 < fmin < fmax < fs/2. "
                f"Got fmin={lo}, fmax={hi}, fs={fs}"
            )

    sos = butter(
        N=order,
        Wn=cutoff,
        btype=btype,
        fs=fs,
        output="sos",
    )

    return sos, btype


def filter_channels(
    data,
    *,
    fs=None,
    fmin=0.0,
    fmax=0.0,
    order=4,
    zero_phase=True,
    axis=0,
    keep_attrs=True,
):
    """
    Filter DAS data along the time axis for each channel.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        DAS data. Expected shape is usually (time, channel).
    fs : float, optional
        Sampling frequency in Hz. If data is an xarray object and fs is None,
        this function tries data.attrs["fs"].
    fmin : float
        Lower cutoff frequency in Hz. Use 0 for LPF.
    fmax : float
        Upper cutoff frequency in Hz. Use 0 for HPF.
    order : int
        Butterworth filter order.
    zero_phase : bool
        If True, use sosfiltfilt. If False, use causal sosfilt.
    axis : int
        Axis along which to filter. For standardized DAS data, use axis=0.
    keep_attrs : bool
        Preserve xarray attrs when input is xr.DataArray.

    Returns
    -------
    filtered : same type as input
        Filtered data.
    """
    is_xarray = isinstance(data, xr.DataArray)

    if fs is None:
        if is_xarray and "fs" in data.attrs:
            fs = data.attrs["fs"]
        else:
            raise ValueError("fs must be provided unless data.attrs['fs'] exists.")

    sos, resolved_btype = make_sos_filter(
        fs,
        fmin=fmin,
        fmax=fmax,
        order=order,
    )

    arr = data.data if is_xarray else np.asarray(data)

    if zero_phase:
        filtered = sosfiltfilt(sos, arr, axis=axis)
        phase_mode = "zero_phase"
    else:
        filtered = sosfilt(sos, arr, axis=axis)
        phase_mode = "causal"

    if not is_xarray:
        return filtered

    out = xr.DataArray(
        filtered,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs.copy() if keep_attrs else {},
        name=data.name,
    )

    out.attrs.update(
        {
            "filter_type": resolved_btype,
            "filter_fmin_hz": float(fmin),
            "filter_fmax_hz": float(fmax),
            "filter_order": int(order),
            "filter_phase": phase_mode,
        }
    )

    return out


def filter_dataset(
    ds,
    *,
    var=None,
    output_var=None,
    fs=None,
    fmin=0.0,
    fmax=0.0,
    order=4,
    zero_phase=True,
    time_dim="time",
):
    """
    Filter one variable in a standardized DAS xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing DAS variable.
    var : str, optional
        Variable to filter. If None, uses the only data variable.
    output_var : str, optional
        Name of output variable. If None, overwrites var in returned copy.
    fs : float, optional
        Sampling frequency. If None, tries ds.attrs["fs"].
    fmin, fmax, order, zero_phase
        Filter settings.
    time_dim : str
        Name of time dimension.

    Returns
    -------
    xr.Dataset
        Dataset with filtered variable.
    """
    if var is None:
        if len(ds.data_vars) != 1:
            raise ValueError(f"Dataset has multiple variables: {list(ds.data_vars)}. Please specify var=...")
        var = list(ds.data_vars)[0]

    if fs is None:
        if "fs" in ds.attrs:
            fs = ds.attrs["fs"]
        else:
            raise ValueError("fs must be provided unless ds.attrs['fs'] exists.")

    axis = ds[var].get_axis_num(time_dim)

    filtered = filter_channels(
        ds[var],
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        order=order,
        zero_phase=zero_phase,
        axis=axis,
    )

    out = ds.copy()

    if output_var is None:
        output_var = var

    out[output_var] = filtered

    out.attrs.update(
        {
            "last_filter_var": output_var,
            "last_filter_fmin_hz": float(fmin),
            "last_filter_fmax_hz": float(fmax),
            "last_filter_order": int(order),
            "last_filter_zero_phase": bool(zero_phase),
        }
    )

    return out


def downsample_array(x, old_fs, new_fs, axis=0):
    q = old_fs / new_fs

    if not q.is_integer():
        raise ValueError(
            f"Downsampling by slicing requires an integer factor: "
            f"old_fs / new_fs = {old_fs} / {new_fs} = {q}"
        )

    q = int(q)

    sl = [slice(None)] * x.ndim
    sl[axis] = slice(None, None, q)

    return x[tuple(sl)]
