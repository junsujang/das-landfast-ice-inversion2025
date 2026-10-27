from __future__ import annotations

import numpy as np

from das_tools.physics import dispersion_relations as dr
from .data import grid_param_names, posterior_from_logp
from .likelihood import gaussian_loglikelihood
from .model_cache import validate_fg_cache


def _interpolate_fg_cache(cache, frequency):
    """The cached FG k(f), linearly interpolated to `frequency`: (nE, nnu, nh, nObs)."""
    fgrid = cache["frequency"]
    f = np.asarray(frequency, float)
    if np.any(f < fgrid[0] - 1e-10) or np.any(f > fgrid[-1] + 1e-10):
        raise ValueError(
            f"FG observation frequencies {f.min():.2f}-{f.max():.2f} Hz "
            f"outside cache {fgrid[0]:.2f}-{fgrid[-1]:.2f} Hz."
        )

    hi = np.searchsorted(fgrid, f, side="right")
    hi = np.clip(hi, 1, len(fgrid) - 1)
    lo = hi - 1
    f0, f1 = fgrid[lo], fgrid[hi]
    weight = (f - f0) / (f1 - f0)
    return cache["k"][..., lo] * (1.0 - weight) + cache["k"][..., hi] * weight


def predict_k(cache, axes, wave_type, frequency):
    """Model k at `frequency` for every node of the (E, nu, h_i) axes.

    P (QS0) and SH0 are closed form and depend on (E, nu) only, so they are
    evaluated exactly and returned as (nE, nnu, 1, nObs), which broadcasts
    along h_i. FG is interpolated from the cache, (nE, nnu, nh, nObs).
    rho_i is the one the cache was built with.
    """
    f = np.asarray(frequency, float)
    if wave_type == "FG":
        return _interpolate_fg_cache(cache, f)
    if wave_type not in ("P", "SH"):
        raise ValueError(f"Unknown wave_type={wave_type!r}. Expected 'P', 'SH', or 'FG'.")
    mode = dr.QS0 if wave_type == "P" else dr.SH0
    E = np.asarray(axes["E"], float)[:, None, None, None]
    nu = np.asarray(axes["nu"], float)[None, :, None, None]
    return mode(f[None, None, None, :], E, nu, float(cache["metadata"]["model_constants"]["rho_i"]))


def event_logposterior_grid_cached(event, grid, cache, include_normalization=True):
    """Whole 3-D log-likelihood grid for one event, vectorized over parameters.

    The cache must already have been checked against `grid` (as `invert_cached` does).
    """
    names = grid_param_names(grid)
    axes = {n: getattr(grid, n) for n in names}
    logp = np.zeros(tuple(len(a) for a in axes.values()), dtype=float)

    for wt, wave in event.waves.items():
        f = np.asarray(wave.frequency, float)
        obs = np.asarray(wave.k_obs, float)
        mask = np.isfinite(f) & np.isfinite(obs) & (f > 0)
        if not np.any(mask):
            continue
        f, obs = f[mask], obs[mask]
        residual = obs - predict_k(cache, axes, wt, f)

        sigma = np.asarray(wave.sigma_k, float)
        sigma = sigma if sigma.ndim == 0 else sigma[mask]
        logp += gaussian_loglikelihood(residual, sigma, include_normalization)
    return logp


def invert_cached(events, grid, cache, include_normalization=True, per_event=True):
    """Per-event and global posteriors from the cache, in one pass over the events.

    Each event's log-likelihood grid is computed once; the global posterior is
    their sum. Returns (event_results, global_result): event_results maps
    event_id -> PosteriorResult, or is None with per_event=False, which saves
    holding every event's posterior grid when only the global one is needed.
    """
    events = list(events)
    validate_fg_cache(cache, grid, events=events)
    names = grid_param_names(grid)
    logp = np.zeros(tuple(len(getattr(grid, n)) for n in names), dtype=float)
    event_results = {} if per_event else None
    for event in events:
        logp_event = event_logposterior_grid_cached(event, grid, cache, include_normalization)
        logp += logp_event
        if per_event:
            event_results[event.event_id] = posterior_from_logp(
                logp_event, grid, f"event_{event.event_id}",
                {"cached": True, "event_id": event.event_id, "available_waves": list(event.waves)},
            )
    global_result = posterior_from_logp(logp, grid, "global", {"cached": True, "n_events": len(events)})
    return event_results, global_result
