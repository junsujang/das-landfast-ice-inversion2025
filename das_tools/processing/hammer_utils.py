import csv, os
import numpy as np
from scipy import signal

from pyproj import CRS,Transformer
from datetime import datetime

from das_tools.physics import dispersion_relations as dr


def get_all_tap_test_geoinfo_inv_all(data_dir):
    file_path = data_dir / "tapTestInv_all.csv"

    tap_data = []

    with open(file_path, 'r', newline='', encoding='utf-8-sig') as latlonFile:
        csv_reader = csv.DictReader(latlonFile)

        for row in csv_reader:
            timestamp = datetime.fromisoformat(row["timestamp"])

            tap_data.append({
                "eventNo": int(row["eventNo"]),
                "timestamp": timestamp, # timestamp.strptime(timestamp,"%Y-%m-%d_%H.%M.00"),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "d1FG": float(row["d1FG"]),
                "d2FG": float(row["d2FG"]),
                "t1": float(row["t1"]),
                "t2FG": float(row["t2FG"]),
                "f1FG": float(row["f1FG"]),
                "f2FG": float(row["f2FG"]),
                "t2L": float(row["t2L"]),
                "f1L": float(row["f1L"]),
                "f2L": float(row["f2L"]),
                "d1S": float(row["d1S"]),
                "d2S": float(row["d2S"]),
                "t1S": float(row["t1S"]),
                "t2S": float(row["t2S"]),
                "f1S": float(row["f1S"]),
                "f2S": float(row["f2S"]),
            })

    return tap_data


def find_das_file(data_files,target_time):
    matches = [
        f for f in data_files
        if target_time in os.path.basename(f)
    ]
    if not matches:
        raise FileNotFoundError(f"No file found for {target_time}")

    return matches[0]



def _local_aeqd_crs(lat0, lon0):
    # Local azimuthal equidistant projection centered at (lat0, lon0)
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def make_local_transformers(lat0, lon0):
    """
    Build forward (lon,lat -> x,y) and inverse (x,y -> lon,lat) transformers
    for a local ENU-like frame in meters centered at (lat0, lon0).
    """
    crs_geo = CRS.from_epsg(4326)  # WGS84 (lon,lat)
    crs_loc = _local_aeqd_crs(lat0, lon0)
    fwd = Transformer.from_crs(crs_geo, crs_loc, always_xy=True)  # (lon, lat) in
    inv = Transformer.from_crs(crs_loc, crs_geo, always_xy=True)  # (x, y) in
    return fwd, inv


def ll_to_xy(lon, lat, lat0, lon0):
    """
    Convert lon/lat (deg) to local x/y (m) relative to (lat0, lon0).
    Accepts scalars or arrays.
    """
    fwd, _ = make_local_transformers(lat0, lon0)
    x, y = fwd.transform(lon, lat)
    return x, y  # meters (x=East, y=North)

def get_window(row,w):
    if w == 0:
        t1 = row["t1"]
        t2 = row["t2FG"]
        d1 = row["d1FG"]
        d2 = row["d2FG"]
        f1 = row["f1FG"]
        f2 = row["f2FG"]
    elif w == 1:
        t1 = row["t1"]
        t2 = row["t2L"]
        d1 = row["d1FG"]
        d2 = row["d2FG"]
        f1 = row["f1L"]
        f2 = row["f2L"]
    elif w == 2:
        t1 = row["t1S"]
        t2 = row["t2S"]
        d1 = row["d1S"]
        d2 = row["d2S"]
        f1 = row["f1S"]
        f2 = row["f2S"]
    else:
        raise ValueError(f"Unknown wave index w={w}; 0 = FG, 1 = P, 2 = SH")

    return t1,t2,d1,d2,f1,f2


def compute_FK(x,Fs,k_grid,r):
    T, M1 = x.shape
    dt = 1/Fs
    # Apply windowing
    wt = signal.windows.hann(T)[:,None]
    wx1 = signal.windows.hann(M1)[None,:]
    w1  = wt * wx1
    xw1 = x * w1
    xw1 = xw1.T

    X = np.fft.rfft(xw1, axis=1)        # (n_receivers, n_freqs)
    freqs = np.fft.rfftfreq(T, dt)
    
    XFK = np.zeros((len(freqs), len(k_grid)))
    for ik, k_trial in enumerate(k_grid):
        # k_trial = k         # (n_freqs,)
        phase = np.exp(1j * np.outer(r, k_trial))  # (n_receivers, n_freqs)
        stack = np.sum(X*phase, axis=0)       # coherent sum over receivers
        XFK[:, ik] = np.abs(stack)**2

    return XFK, freqs


# Physical prior on where a tap-test ridge can lie: the dispersion curves of a
# soft and a stiff ice bound the k band. k_low comes from the stiff bound.
TAP_BAND_E_BOUNDS = {"FG": (0.5e9, 10.0e9), "P": (3.0e9, 10.0e9), "SH": (3.0e9, 10.0e9)}
TAP_BAND_CONSTANTS = dict(nu=0.3, h_i=1.0, h_w=10.0, rho_i=910.0, rho_w=1025.0, P=0.0)


def tap_test_band_mask(wave_type, freqs, **constants):
    """(k_low, k_high) per frequency: the physical k band `pick_ridge` searches.

    `constants` overrides TAP_BAND_CONSTANTS. FG needs f >= 1e-3, so the DC bin is
    evaluated at 1e-3 on a copy of `freqs`.
    """
    c = {**TAP_BAND_CONSTANTS, **constants}
    E_soft, E_stiff = TAP_BAND_E_BOUNDS[wave_type]
    f = np.array(freqs, dtype=float)
    if wave_type == "FG":
        f[np.abs(f) < 1e-3] = 1e-3
        k_of = lambda E: dr.fg_disp_eq_ml_c(f, E, c["nu"], c["h_i"], c["h_w"],
                                            c["rho_i"], c["rho_w"], c["P"])
    elif wave_type == "P":
        k_of = lambda E: dr.QS0(f, E, c["nu"], c["rho_i"])
    elif wave_type == "SH":
        k_of = lambda E: dr.SH0(f, E, c["nu"], c["rho_i"])
    else:
        raise ValueError(f"Unknown wave_type={wave_type!r}")
    return k_of(E_stiff), k_of(E_soft)
