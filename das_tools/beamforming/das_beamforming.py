"""Plane-wave steering on a (k, theta) grid, the steering cache and the
conventional beamformer shared by the bf_inv_disp scripts.

Conventions:
    theta    back-azimuth, 0 = +y (north), clockwise positive (90 deg = east)
    k        kx = k sin(theta),  ky = k cos(theta)
    a_m      w_m(theta) exp(-i [kx x_m + ky y_m]) / sqrt(M), as stored in the cache
    B        b^H R b / (tr(R) |b|^2) with b = conj(a), the normalized beam response.
             With R = X X^H of rFFT snapshots, b matches a wave travelling away
             from theta, so theta is the direction the wave comes from.

Steering-matrix columns are ordered (k, theta) in C order:
[k0 theta0..thetaN, k1 theta0..thetaN, ...].
"""

from __future__ import annotations

import json

import numpy as np

from pathlib import Path


# --------------------------------------------------------------------------- #
# Band and steering-cache layout. The band and f_bin_skip are set only in
# step1_steer_vec_disp.py, recorded in steering_meta_<postfix>.json, and read
# back by later steps through resolve_cache().
# --------------------------------------------------------------------------- #

DEFAULT_BAND = {"fmin": 0.10, "fmax": 1.00, "window_s": 180.0, "cov_duration_h": 0.5}


def make_band(fmin: float, fmax: float, window_s: float, cov_duration_h: float) -> dict:
    """Band dict: [fmin, fmax] Hz, snapshot window [s], covariance-block length [h].

    The name ("0.10-1.00Hz") is derived from fmin/fmax so it cannot contradict them.
    """
    if not 0 < fmin < fmax:
        raise ValueError(f"Need 0 < fmin < fmax, got {fmin}, {fmax}")
    return {
        "name": f"{fmin:.2f}-{fmax:.2f}Hz",
        "fmin": float(fmin),
        "fmax": float(fmax),
        "window_s": float(window_s),
        "cov_duration_h": float(cov_duration_h),
    }


def steering_cache_dir(results_dir: Path, band_name: str) -> Path:
    return Path(results_dir) / "steering" / band_name.replace(" ", "_")


def steering_meta_path(results_dir: Path, band_name: str, postfix: str) -> Path:
    return steering_cache_dir(results_dir, band_name) / f"steering_meta_{postfix}.json"


def resolve_cache(results_dir: Path, directional_floor: float, directional_power: float,
                  f_bin_skip: int | None = None, band_name: str | None = None
                  ) -> tuple[dict, dict, str]:
    """(band, steering meta, postfix) of the cache step 1 built for this directional pair.

    band_name / f_bin_skip are needed only to choose among several built caches.
    """
    root = Path(results_dir) / "steering"
    base = steering_postfix(directional_floor, directional_power, 1)
    want = (None if f_bin_skip is None
            else steering_postfix(directional_floor, directional_power, f_bin_skip))

    built = sorted((p.parent.name, p.name[len("steering_meta_"):-len(".json")])
                   for p in root.glob("*/steering_meta_*.json"))
    matches = [(b, pf) for b, pf in built
               if (pf == want if want else pf == base or pf.startswith(base + "_skip"))
               and (band_name is None or b == band_name)]

    if len(matches) == 1:
        b, pf = matches[0]
        meta = json.loads(steering_meta_path(results_dir, b, pf).read_text())
        return meta["band"], meta, pf

    asked = want or f"{base}[_skipN]"
    if band_name:
        asked += f" for band {band_name!r}"
    if matches:
        raise SystemExit(
            f"Stage 1 built several caches matching {asked}: "
            f"{[f'{b} [{pf}]' for b, pf in matches]}. Choose with --band and/or --f-bin-skip."
        )
    raise FileNotFoundError(
        f"No steering cache {asked} under {root}. "
        f"Built: {[f'{b} [{pf}]' for b, pf in built] or 'nothing -- run step1_steer_vec_disp.py first'}. "
        "--directional-floor / --directional-power (and --band / --f-bin-skip, if "
        "given) must match what stage 1 built."
    )


def estimate_channel_orientation(ch_xy, half_window=2):
    """Local fiber tangent angle phi (M,) [rad, 0 = north, CW+] from ch_xy (2, M) = (east, north)."""
    xy = np.asarray(ch_xy).T  # (M, 2)
    M = xy.shape[0]
    phi = np.empty(M, dtype=float)

    for i in range(M):
        i0 = max(0, i - half_window)
        i1 = min(M - 1, i + half_window)

        if i1 == i0:
            if i == 0:
                i1 = 1
            else:
                i0 = i - 1

        dx = xy[i1, 0] - xy[i0, 0]  # east
        dy = xy[i1, 1] - xy[i0, 1]  # north

        # arctan2(east, north) matches kx = k sin(theta), ky = k cos(theta)
        phi[i] = np.arctan2(dx, dy)

    return phi


def directional_weight_table(
    ch_xy: np.ndarray,
    theta: np.ndarray,
    directional_floor: float,
    directional_power: float,
    orientation_half_window: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """(phi (M,), W_theta (M, nTheta)): fiber tangents and amplitude weights

        w_m(theta) = floor + (1 - floor) * |cos(theta - phi_m)| ** directional_power

    modelling the along-fiber strain sensitivity of a DAS channel. The weight is
    on amplitude, so beam power scales as its square. directional_power = 0 gives
    all ones (omnidirectional).
    """
    phi = estimate_channel_orientation(ch_xy, half_window=orientation_half_window)
    delta = np.asarray(theta, dtype=float)[None, :] - phi[:, None]

    # abs() before the power: a fractional power of a negative cosine is NaN.
    W_theta = np.power(np.abs(np.cos(delta)), directional_power)
    W_theta = directional_floor + (1.0 - directional_floor) * W_theta

    return phi, W_theta


def steering_matrix(
    k: np.ndarray,
    theta: np.ndarray,
    ch_xy: np.ndarray,
    W_theta: np.ndarray | None = None,
) -> np.ndarray:
    """Steering matrix A (M, nK * nTheta), columns in (k, theta) C order.

    k (nK,) [rad/m]; theta (nTheta,) [rad]; ch_xy (2, M) [m]. W_theta (M, nTheta),
    if given, is applied identically at every k.
    """
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    ch_xy = np.asarray(ch_xy, dtype=float)

    if k.ndim != 1:
        raise ValueError("k must have shape (nK,)")
    if theta.ndim != 1:
        raise ValueError("theta must have shape (nTheta,)")
    if ch_xy.ndim != 2 or ch_xy.shape[0] != 2:
        raise ValueError("ch_xy must have shape (2, M)")

    M = ch_xy.shape[1]
    r = ch_xy.T  # (M, 2)

    kx = k[:, None] * np.sin(theta)[None, :]
    ky = k[:, None] * np.cos(theta)[None, :]
    Kgrid = np.vstack([kx.ravel(order="C"), ky.ravel(order="C")])  # (2, nK * nTheta)

    A = np.exp(-1j * (r @ Kgrid)) / np.sqrt(M)

    if W_theta is not None:
        A *= np.tile(W_theta, (1, k.size))

    return A


def quadratic_form(Rf: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re(a_j^H R a_j) and ||a_j||^2 for every column a_j of A, R (M, M)."""
    num = np.real(np.sum(A.conj() * (Rf @ A), axis=0))
    a_norm2 = np.sum(np.abs(A) ** 2, axis=0).real
    return num, a_norm2


def precompute_A_per_frequency_directional(
    k_per_freq,
    ch_xy,
    theta,
    cache_dir,
    postfix,
    directional_floor=0.0,
    directional_power=0.0,
    orientation_half_window=2,
    round_decimals=12,
):
    """Write one steering file A_f_XXXX_<postfix>.npz per frequency row of k_per_freq.

    Each file holds A for that row's non-NaN wavenumbers with the directional
    weights baked in; existing files are skipped.
    k_per_freq (F, K), ch_xy (2, M), theta (nTheta,) [rad].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    k_per_freq = np.asarray(k_per_freq, dtype=float)
    theta = np.asarray(theta, dtype=float)
    F, _ = k_per_freq.shape
    M = ch_xy.shape[1]

    phi, W_theta = directional_weight_table(
        ch_xy=ch_xy,
        theta=theta,
        directional_floor=directional_floor,
        directional_power=directional_power,
        orientation_half_window=orientation_half_window,
    )

    print(f"{F} steering files expected in {cache_dir}")
    n_written = 0

    for f_idx in range(F):
        cache_file = cache_dir / f"A_f_{f_idx:04d}_{postfix}.npz"
        if cache_file.exists():
            continue

        krow = k_per_freq[f_idx, :]
        valid_mask = ~np.isnan(krow)

        if not valid_mask.any():
            np.savez(
                cache_file,
                A=np.zeros((M, 0), dtype=np.complex128),
                a_norm2=np.zeros((0,), dtype=float),
                valid_mask=valid_mask,
                K_valid=0,
            )
            n_written += 1
            continue

        k_valid = np.round(krow[valid_mask], round_decimals)
        A_f = steering_matrix(k_valid, theta, ch_xy, W_theta)    # (M, K_valid * A_len)
        a_norm2 = np.sum(np.abs(A_f) ** 2, axis=0).real

        np.savez(
            cache_file,
            A=A_f,
            a_norm2=a_norm2,
            valid_mask=valid_mask,
            K_valid=k_valid.size,
            phi=phi,
            W_theta=W_theta,
            directional_floor=directional_floor,
            directional_power=directional_power,
            orientation_half_window=orientation_half_window,
        )
        n_written += 1

    print(f"Wrote {n_written} new steering file(s); {F - n_written} already present")


def compute_steering_per_freq_directional(
    cache_dir: Path,
    k_disp_valid: np.ndarray,
    theta: np.ndarray,
    ch_xy: np.ndarray,
    f_bin_skip: int,
    postfix: str,
    directional_floor: float = 0.0,
    directional_power: float = 0.0,
    orientation_half_window: int = 2,
):
    """precompute_A_per_frequency_directional for every f_bin_skip-th bin of k_disp_valid (nK_valid, nF)."""
    if k_disp_valid.ndim != 2:
        raise ValueError("k_disp_valid must have shape (nK_valid, nF)")

    k_per_freq = k_disp_valid[:, ::f_bin_skip].T

    precompute_A_per_frequency_directional(
        k_per_freq=k_per_freq,
        ch_xy=ch_xy,
        theta=theta,
        cache_dir=cache_dir,
        postfix=postfix,
        directional_floor=directional_floor,
        directional_power=directional_power,
        orientation_half_window=orientation_half_window,
    )


def beamform_k_aoa_implicit_vec_2Dk_cached_from_cov(
    R, theta, cache_dir, cache_postfix,
    eps=1e-12, return_E3_sum=False
):
    """Conventional beamformer of R against the cached A_f_XXXX_<cache_postfix>.npz files.

    R (F, M, M) must be on the cache's frequency bins (A_f_0000 .. A_f_{F-1}).
    Returns (E3_per_f, B_per_f), each (F, K_len, A_len), NaN where k is invalid:
    B is the normalized response (module docstring) and
    E3 = M * (log(1 - B) + log tr R). return_E3_sum returns only E3 summed over F.
    """

    R = np.asarray(R)
    if R.ndim != 3:
        raise ValueError(f"Expected R to be (F,M,M). Got {R.shape}")

    F, M, M2 = R.shape

    if M2 != M:
        raise ValueError(f"R must be square in last dims. Got {R.shape}")

    A_len = len(theta)
    cache_dir = Path(cache_dir)

    first = np.load(cache_dir / f"A_f_0000_{cache_postfix}.npz")
    K_len = len(first["valid_mask"])

    E3_per_f = np.full((int(F), K_len, A_len), np.nan, dtype=float)
    B_per_f = np.full_like(E3_per_f, np.nan)
    print(f"{F} freq bins in R expected!")

    for out_idx in range(F):
        cache = np.load(cache_dir / f"A_f_{out_idx:04d}_{cache_postfix}.npz")
        valid_mask = cache["valid_mask"]
        K_valid = int(cache["K_valid"])

        if K_valid == 0:
            continue

        A_f = cache["A"].conj()   # (M, Nsteer) where Nsteer=K_valid*A_len
        Rf = R[out_idx, :, :]     # (M,M)
        num, a_norm2 = quadratic_form(Rf, A_f)

        x_norm2 = np.real(np.trace(Rf))

        denom = np.maximum(x_norm2 * a_norm2, eps)
        B_conv = np.exp(np.log(num)-np.log(denom))

        B = 1.0 - B_conv
        B = np.clip(B, eps, None)

        P_f = M * (np.log(B) + np.log(np.maximum(x_norm2, eps)))

        P_f_mat = P_f.reshape(K_valid, A_len, order="C")
        E3_per_f[out_idx, valid_mask, :] = P_f_mat
        B_per_f[out_idx, valid_mask, :] = B_conv.reshape(K_valid,A_len,order="C")

    if return_E3_sum:
        return np.nansum(E3_per_f, axis=0)
    return E3_per_f, B_per_f



def steering_postfix(directional_floor: float, directional_power: float,
                     f_bin_skip: int) -> str:
    """Cache postfix for a directional setting and frequency decimation.

        omni                                if directional_power <= 0
        dirSen_<floor*100>_pwr<power*100>   otherwise, 3 digits: dirSen_025_pwr150
        + _skip<N>                          if f_bin_skip > 1: omni_skip3

    The single naming rule for steering files and every downstream output, so a
    step asking for a cache nobody built finds none rather than the wrong one.
    f_bin_skip has no default on purpose: forgetting it would silently select the
    undecimated cache.
    """
    if f_bin_skip < 1:
        raise ValueError(f"f_bin_skip must be >= 1, got {f_bin_skip}")

    base = "omni" if directional_power <= 0.0 else (
        f"dirSen_{int(round(directional_floor * 100)):03d}"
        f"_pwr{int(round(directional_power * 100)):03d}"
    )
    return base if int(f_bin_skip) == 1 else f"{base}_skip{int(f_bin_skip)}"


def normalization_tag(is_coherence_norm: bool) -> str:
    """Output-path tag for the covariance normalization: "raw" or "coh".

    Keeps raw and coherence runs apart (the sweep skips existing outputs, so they
    would otherwise silently collide). Not part of the steering-cache name, which
    is independent of the normalization.
    """
    if is_coherence_norm:
        return "coh"
    return "raw"
