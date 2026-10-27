"""Dispersion relations for floating sea ice: QS0 (P), SH0 and flexural-gravity (FG).

The FG branch is a thin elastic plate with mass loading and in-plane compression
over water of finite depth:

    D k^4 - P h_i k^2 - rho_i h_i w^2 + rho_w g - rho_w w^2 coth(k h_w) / k = 0,
    D = E h_i^3 / (12 (1 - nu^2)).

Every FG wavenumber in the package -- the tap-test forward model, its cache, and the
beamforming dispersion grid -- comes from the one vectorized solver below
(`fg_disp_eq_ml_c_vec`). Above `SUPERCRITICAL_RATIO` of the buckling stress the
equation has several roots and no single FG branch; callers that sweep P must mask
those cells with `fg_is_supercritical`.
"""

import numpy as np

# Gravity
G = 9.81

# P / P_cr at and above which an FG branch is treated as undefined. The branch-
# tracing solver the beamforming grid was first built with used the same cut.
SUPERCRITICAL_RATIO = 0.8


def flexural_rigidity(E, nu, h_i):
    """D = E h_i^3 / (12 (1 - nu^2)) [N m]."""
    return np.asarray(E, float)*np.asarray(h_i, float)**3/(12*(1 - np.asarray(nu, float)**2))


def critical_compression(E, nu, h_i, rho_w, g=G):
    """Buckling stress P_cr = 2 sqrt(D rho_w g) / h_i [Pa] of the floating plate."""
    return 2.0*np.sqrt(flexural_rigidity(E, nu, h_i)*rho_w*g)/np.asarray(h_i, float)


def fg_is_supercritical(E, nu, h_i, rho_w, P, g=G):
    """True where P / P_cr >= SUPERCRITICAL_RATIO, i.e. where no FG branch is defined."""
    return np.asarray(P, float)/critical_compression(E, nu, h_i, rho_w, g) >= SUPERCRITICAL_RATIO


def fg_residual_and_slope(k, omega2, D, h_i, h_w, rho_i, rho_w, P, g=G):
    """FG dispersion residual F(k) (module docstring) and dF/dk, elementwise."""
    coth = 1.0/np.tanh(k*h_w)
    F = (D*k**4 - P*h_i*k**2 - h_i*rho_i*omega2 + rho_w*g) - rho_w*omega2*coth/k
    dF = 4*D*k**3 - 2*P*h_i*k + rho_w*omega2*(h_w*(coth**2 - 1)/k + coth/k**2)
    return F, dF


def fg_disp_eq_ml_c(f, E, nu, h_i, h_w, rho_i, rho_w, P):
    """FG wavenumber k(f) for one parameter set; a float for scalar ``f``.

    Kept for its call signature: it is `fg_disp_eq_ml_c_vec`, which every other
    FG caller in the package also uses.
    """
    k = fg_disp_eq_ml_c_vec(f, E, nu, h_i, h_w, rho_i, rho_w, P)
    return float(k) if np.ndim(f) == 0 else k


def fg_disp_eq_ml_c_vec(f, E, nu, h_i, h_w, rho_i, rho_w, P=0.0, g=G,
                        k_lo=1e-6, k_hi=3.0, xtol=2e-12, max_iter=200):
    """FG wavenumber k(f; E, nu, h_i, h_w, rho_i, rho_w, P): every argument broadcasts.

    Returns NaN wherever [k_lo, k_hi] has no sign change. All combinations are
    solved at once with a safeguarded Newton iteration: a Newton step is taken
    when it stays inside the current bracket, otherwise the bracket is bisected.
    This is what makes caching the FG branch over a full parameter grid cheap.

    Tolerance matches scipy brentq's default (xtol=2e-12 plus 4 ulp relative), and
    below the supercritical cut the result agrees with brentq on the same bracket
    and with branch tracing to < 1e-9 relative.
    """
    f = np.asarray(f, dtype=float)
    if np.any(f < 1e-3):
        raise ValueError(f"Frequencies need to be larger than 1e-3")
    omega2 = (2*np.pi*f)**2
    D = flexural_rigidity(E, nu, h_i)
    shape = np.broadcast_shapes(omega2.shape, D.shape, np.shape(h_i), np.shape(h_w),
                                np.shape(rho_i), np.shape(rho_w), np.shape(P))
    args = [np.broadcast_to(np.asarray(a, float), shape).ravel()
            for a in (omega2, D, h_i, h_w, rho_i, rho_w, P)]

    def residual_and_slope(k, *a):
        return fg_residual_and_slope(k, *a, g=g)

    n = args[0].size
    F_lo, _ = residual_and_slope(np.full(n, k_lo), *args)
    F_hi, _ = residual_and_slope(np.full(n, k_hi), *args)
    bracketed = np.sign(F_lo)*np.sign(F_hi) < 0

    # Track the end of the bracket where F < 0 and the end where F > 0, so the
    # update does not depend on which way F is oriented.
    k_neg = np.where(F_lo < 0, k_lo, k_hi)
    k_pos = np.where(F_lo < 0, k_hi, k_lo)
    # Start from the deep-water flexural limit D k^5 = rho_w w^2.
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.clip((args[5]*args[0]/args[1])**0.2, k_lo, k_hi)
    k = np.where(np.isfinite(k), k, 0.5*(k_lo + k_hi))

    eps4 = 4*np.finfo(float).eps
    idx = np.flatnonzero(bracketed)
    for _ in range(max_iter):
        if idx.size == 0:
            break
        ka, kn, kp = k[idx], k_neg[idx], k_pos[idx]
        F, dF = residual_and_slope(ka, *(a[idx] for a in args))

        neg = F < 0
        kn = np.where(neg, ka, kn)
        kp = np.where(neg, kp, ka)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_new = ka - F/dF
        lo, hi = np.minimum(kn, kp), np.maximum(kn, kp)
        inside = np.isfinite(k_new) & (k_new > lo) & (k_new < hi)
        k_new = np.where(inside, k_new, 0.5*(kn + kp))
        k_new = np.where(F == 0, ka, k_new)

        done = (F == 0) | (np.abs(k_new - ka) <= xtol + eps4*np.abs(k_new)) | (hi - lo <= xtol + eps4*hi)
        k[idx], k_neg[idx], k_pos[idx] = k_new, kn, kp
        idx = idx[~done]

    return np.where(bracketed, k, np.nan).reshape(shape)


# Dispersion relation (non-dispersive) for quasi-symmetric mode (QS0)
# k = \omega*\sqrt{\rho_i*(1-v^2)/E}
def QS0(f,E,nu,rho_i):
    omega = 2*np.pi*f
    kq = omega*np.sqrt(rho_i*(1-nu**2)/E)

    return kq


# Dispersion relation (non-dispersive) for shear-horizontal mode (SH0)
# k = \omega*\sqrt{2*\rho_i*(1+v)/E}
def SH0(f,E,nu,rho_i):
    omega = 2*np.pi*f
    ks = omega*np.sqrt(2*rho_i*(1+nu)/E)
    return ks
