"""Snapshot spectra -> averaged cross-spectral (covariance) matrices R(f), shape (F, M, M)."""

import numpy as np

from das_tools.processing import spectral
from das_tools.processing.filtering import filter_channels



def snapshot_covariance(x_win: np.ndarray, fs: float, band: dict,
                        filter_order: int = 4):
    """Filter + FFT + covariance for one snapshot, or None if it holds any NaN.

    A NaN anywhere in the window would spread across every channel through the
    filter and then across every frequency through the FFT, so a gapped window is
    dropped whole rather than patched.
    """
    if np.isnan(x_win).any():
        return None

    x_filt = filter_channels(
        x_win, fs=fs, fmin=band["fmin"], fmax=band["fmax"],
        order=filter_order, zero_phase=False, axis=0,
    )
    X, freq = spectral.bandlimited_windowed_rfft(x_filt, fs, band["fmin"], band["fmax"])
    Rk = np.einsum('fm,fn->fmn', X, X.conj())
    return Rk, freq


def block_covariance(block: np.ndarray, fs: float, band: dict,
                     min_valid_snapshots: int, filter_order: int = 4):
    """Average the snapshot covariances inside one covariance block.

    block is (T, M) covering band["cov_duration_h"]; snapshots are consecutive,
    non-overlapping band["window_s"] windows. A block short of min_valid_snapshots
    clean ones is not scored at all (returns None), since an average over one or
    two snapshots is a rank-deficient covariance that beamforms into confident
    nonsense.

    Returns (R (F, M, M), freq (F,), n_valid, n_windows_expected).
    """
    n_win_samples = round(band["window_s"] * fs)
    n_windows_expected = round(band["cov_duration_h"] * 3600.0 / band["window_s"])

    Rk_sum = None
    freq = None
    n_valid = 0

    for i in range(n_windows_expected):
        x_win = block[i * n_win_samples: (i + 1) * n_win_samples]
        if x_win.shape[0] < n_win_samples:
            break

        result = snapshot_covariance(x_win, fs, band, filter_order)
        if result is None:
            continue

        Rk_this, freq = result
        Rk_sum = Rk_this if Rk_sum is None else Rk_sum + Rk_this
        n_valid += 1

    if n_valid < min_valid_snapshots:
        return None

    return Rk_sum / n_valid, freq, n_valid, n_windows_expected


def normalize_to_coherence(Rk: np.ndarray, eps_D: float = 1e-12) -> np.ndarray:
    """R(f) -> D(f)^-1/2 R(f) D(f)^-1/2, D(f) = diag(R(f)) -- unit diagonal.

    eps_D floors the diagonal so low-power channels cannot blow up.
    """
    D = np.real(np.diagonal(Rk, axis1=-2, axis2=-1))       # (F, M)
    D = np.maximum(D, eps_D)
    inv_sqrt_D = 1.0 / np.sqrt(D)
    return Rk * inv_sqrt_D[..., :, None] * inv_sqrt_D[..., None, :]
