"""Snapshot spectra -> averaged cross-spectral (covariance) matrices R(f), shape (F, M, M)."""

import numpy as np

from das_tools.processing import spectral
from das_tools.processing.filtering import filter_channels


# def neighbor_coherence_weights(
#     X_sel,
#     eps=1e-12,
#     clip=(0.05, 1.0),
# ):
#     """
#     Compute per-channel reliability weights from adjacent-channel complex coherence.

#     X_sel: (Nsnap, Nf, M)

#     Returns w: (M,) in [clip[0], clip[1]]
#     """
#     X_sel = np.asarray(X_sel)
#     if X_sel.ndim != 3:
#         raise ValueError("X_sel must be (Nsnap,Nf,M), got {}".format(X_sel.shape))
#     _, _, M = X_sel.shape
#     if M < 2:
#         return np.ones(M, dtype=float)

#     Xi = X_sel[:, :, :-1]
#     Xj = X_sel[:, :, 1:]

#     Sij = np.mean(Xi * np.conj(Xj), axis=(0,1))
#     Sii = np.mean(np.abs(Xi)**2, axis=(0,1)) + eps
#     Sjj = np.mean(np.abs(Xj)**2, axis=(0,1)) + eps

#     coh = np.abs(Sij) / np.sqrt(Sii * Sjj)

#     w = np.zeros(M, dtype=float)
#     w[0] = coh[0]
#     w[-1] = coh[-1]
#     w[1:-1] = 0.5*(coh[:-1] + coh[1:])

#     w = np.clip(w, clip[0], clip[1])
#     w = w / np.max(w)
#     return w


# def compute_snapshot_covariance(X):
#     """Outer product X X^H per frequency, X (F, M), after neighbor-coherence weighting."""
#     # w_ch = neighbor_coherence_weights(X[np.newaxis, :, :])  # (M,)
#     # w_ch = np.asarray(w_ch, dtype=float)
#     # sw = np.sqrt(w_ch).astype(float)
#     # X = X * sw[None, :]

#     Rk = np.einsum('fm,fn->fmn', X, X.conj())  # (F,M,M)
#     return Rk


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
