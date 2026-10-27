from __future__ import annotations

import numpy as np


def pick_ridge(XFK, freqs, k_grid, f1, f2, k_low=None, k_high=None, *, floor_db=-20.0):
    """One dispersion curve k(f) from an FK power image, for f in [f1, f2].

    XFK is (n_freq, n_k) power. It is converted to dB, and where k lies outside
    the physical band [k_low(f), k_high(f)] (optional, one value per frequency)
    it is set to `floor_db`. At every frequency the pick is the positive-k
    wavenumber of maximum power (the forward model predicts the outgoing branch).

    Returns (k_obs, f_obs); both empty if f2 <= f1 or nothing falls in range.
    """
    XFK = np.asarray(XFK, float)
    freqs = np.asarray(freqs, float)
    k_grid = np.asarray(k_grid, float)
    empty = np.array([], float), np.array([], float)
    if f2 <= f1:
        return empty

    score = 10.0 * np.log10(np.maximum(XFK, np.finfo(float).tiny))
    if k_low is not None and k_high is not None:
        k_low = np.broadcast_to(np.asarray(k_low, float), freqs.shape)
        k_high = np.broadcast_to(np.asarray(k_high, float), freqs.shape)
        outside = (k_grid[None, :] < k_low[:, None]) | (k_grid[None, :] > k_high[:, None])
        score = np.where(outside, floor_db, score)

    fm = (freqs >= f1) & (freqs <= f2)
    km = k_grid > 0
    if not fm.any() or not km.any():
        return empty

    score = score[np.ix_(fm, km)]
    score = np.where(np.isfinite(score), score, -np.inf)
    return k_grid[km][np.argmax(score, axis=1)], freqs[fm]
