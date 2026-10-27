from __future__ import annotations
import numpy as np


def gaussian_loglikelihood(residual, sigma_k, include_normalization=True):
    """Independent Gaussian log-likelihood, summed over the last axis of `residual`.

    residual : (..., n). Leading axes (e.g. a parameter grid) are kept, so the
        cached inversion evaluates every grid point with this same function.
    sigma_k : scalar, or (n,) per-observation standard deviations.

    Returns a float for a 1-D residual, otherwise an array over the leading axes.
    Any non-finite residual in a sum makes that entry -inf.
    """
    residual = np.asarray(residual, float)
    sigma = np.asarray(sigma_k, float)
    n = residual.shape[-1] if residual.ndim else residual.size
    if n == 0:
        return 0.0 if residual.ndim <= 1 else np.zeros(residual.shape[:-1])
    
    if sigma.ndim == 0:
        if float(sigma) <= 0 or not np.isfinite(float(sigma)):
            raise ValueError("sigma_k must be finite and positive.")
        log_sigma_sum = n*np.log(float(sigma))
    else:
        log_sigma_sum = np.sum(np.log(sigma))

    ll = -0.5*np.sum((residual/sigma)**2, axis=-1)
    if include_normalization:
        ll += -0.5*n*np.log(2*np.pi) - log_sigma_sum
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    return float(ll) if ll.ndim == 0 else ll
