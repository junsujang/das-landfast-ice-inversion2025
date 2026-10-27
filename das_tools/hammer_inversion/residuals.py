"""Residual correlation of the hammer inversion, and the sigma_m it implies.

The likelihood assumes

    k_hat[s, m](f_j) ~ N( k_m(f_j; theta), sigma_m^2 )

independently over records s and frequencies f_j. Frequency samples from one
f-k image are not independent: spectral leakage and the finite frequency
resolution of the FFT window correlate neighbouring frequencies. `residual_acf` measures that correlation within each record's
curve and turns it into a variance inflation factor and an effective sample
size,

    VIF = 1 + 2 sum_L (1 - L/n) rho_L ,      n_eff = n / VIF ,

and the sigma_m to use in the inversion is sigma_rms * sqrt(VIF), sigma_rms
being the root-mean-square residual, so that a correlated curve does not count
as VIF times more evidence than it is.

`drop_outlier_picks` removes picks whose residual is more than n robust
standard deviations (1.4826 MAD) from their mode's median, before sigma_m
is estimated.

`plot_sigma_and_gaussian` shows the residuals against frequency and their
pooled distribution against a Gaussian.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .cached_inversion import predict_k

from .data import grid_param_names



def _make_predictor(*, cache, result=None, map_params=None, grid=None):
    """Build k_pred(wave_type, frequency) at one grid node.

    The same model evaluation as the inversion itself (`predict_k`), so residuals cannot
    disagree with the likelihood. The node is the MAP of `result`, or
    `map_params` snapped to `grid`.
    """

    if result is not None:
        axes = dict(result.parameter_axes or {})
        params = dict(result.map_params)
    elif grid is not None and map_params is not None:
        axes = {n: np.asarray(getattr(grid, n), float) for n in grid_param_names(grid)}
        params = dict(map_params)
    else:
        raise ValueError("Pass either `result` or both `grid` and `map_params`.")
    if not axes:
        raise ValueError("Result carries no parameter axes; pass `grid` explicitly.")

    names = tuple(axes.keys())
    idx = tuple(int(np.argmin(np.abs(np.asarray(axes[n], float) - params[n]))) for n in names)
    snapped = {n: float(np.asarray(axes[n], float)[i]) for n, i in zip(names, idx)}

    shape = tuple(len(axes[n]) for n in names)

    def predict(wave_type, frequency):
        pred_all = predict_k(cache, axes, wave_type, frequency)
        pred_all = np.broadcast_to(pred_all, shape + pred_all.shape[-1:])  # P/SH have h_i of length 1
        return np.asarray(pred_all[idx + (slice(None),)], float)

    return predict, snapped


def standardized_residuals(events, *, cache, result=None, map_params=None, grid=None):
    """Per-observation residual rows for every record and mode.

    Returns a list of dicts (DataFrame-ready) with one row per (record, mode,
    frequency): k_obs, k_pred, residual, sigma, and the standardized residual.
    """
    predict, params = _make_predictor(cache=cache, result=result, map_params=map_params, grid=grid)

    rows = []
    for ev in events:
        for wt, w in ev.waves.items():
            f = np.asarray(w.frequency, float)
            obs = np.asarray(w.k_obs, float)
            m = np.isfinite(f) & np.isfinite(obs) & (f > 0)
            if not m.any():
                continue
            f, obs = f[m], obs[m]
            pred = predict(wt, f)

            sigma = np.asarray(w.sigma_k, float)
            sigma = np.full(f.shape, float(sigma)) if sigma.ndim == 0 else sigma[m]

            residual = obs - pred
            rows.extend({
                "event_id": ev.event_id,
                "wave": wt,
                "frequency": float(fj),
                "k_obs": float(oj),
                "k_pred": float(pj),
                "residual": float(dj),
                "sigma": float(sj),
                "r": float(dj / sj),
            } for fj, oj, pj, dj, sj in zip(f, obs, pred, residual, sigma))

    # Record which fit these residuals came from, as scalar columns so the
    # rows stay a flat, pandas-friendly table.
    for row in rows:
        for name, value in params.items():
            row[f"fit_{name}"] = float(value)
    return rows


def _split_by(rows, *keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return groups


def _curve_arrays(group):
    """Frequency-sorted (f, r) for one (record, mode) curve."""
    f = np.array([g["frequency"] for g in group], float)
    r = np.array([g["r"] for g in group], float)
    order = np.argsort(f)
    return f[order], r[order]


def residual_acf(rows, max_lag=8, spacing_tolerance=0.5):
    """Frequency-lag autocorrelation of standardized residuals, per mode.

    Correlations are computed *within* one record's curve only, pairing samples
    that are genuinely `lag` frequency bins apart: a pair is used only if its
    frequency separation is within `spacing_tolerance` bins of lag*df, where df
    is that curve's own median bin spacing. This keeps gaps in a curve (from QC
    removal, or the non-uniform native FK axis) from being silently treated as
    adjacent samples.

    Also returns the variance inflation factor

        VIF = 1 + 2 * sum_lag (1 - lag/n) * rho_lag ,

    the factor by which the iid likelihood overstates the information content
    of one curve, and the implied effective sample size n_eff = n / VIF.
    """
    out = {}
    for (wave,), group in _split_by(rows, "wave").items():
        curves = [_curve_arrays(g) for g in _split_by(group, "event_id", "wave").values()]
        curves = [c for c in curves if c[0].size >= 3]
        if not curves:
            continue

        n_total = sum(c[0].size for c in curves)
        mean_len = float(np.mean([c[0].size for c in curves]))

        acf = np.full(max_lag + 1, np.nan)
        npairs = np.zeros(max_lag + 1, dtype=int)
        acf[0], npairs[0] = 1.0, n_total

        for lag in range(1, max_lag + 1):
            a, b = [], []
            for f, r in curves:
                if f.size <= lag:
                    continue
                df = np.median(np.diff(f))
                if not np.isfinite(df) or df <= 0:
                    continue
                sep = f[lag:] - f[:-lag]
                ok = np.abs(sep - lag * df) <= spacing_tolerance * df
                a.append(r[:-lag][ok])
                b.append(r[lag:][ok])
            if not a:
                continue
            a = np.concatenate(a)
            b = np.concatenate(b)
            npairs[lag] = a.size
            if a.size >= 5 and np.std(a) > 0 and np.std(b) > 0:
                acf[lag] = float(np.corrcoef(a, b)[0, 1])

        lags = np.arange(max_lag + 1)
        usable = min(max_lag, max(1, int(mean_len) - 1))
        terms = [(1.0 - lag / mean_len) * acf[lag]
                 for lag in range(1, usable + 1) if np.isfinite(acf[lag])]
        vif = float(max(1.0, 1.0 + 2.0 * np.sum(terms))) if terms else 1.0

        out[wave] = {
            "lag": lags,
            "acf": acf,
            "n_pairs": npairs,
            "n_points": n_total,
            "n_curves": len(curves),
            "mean_curve_length": mean_len,
            # 95% white-noise band for a correlation from n_pairs samples.
            "white_noise_band": np.where(npairs > 0, 1.96 / np.sqrt(np.maximum(npairs, 1)), np.nan),
            "vif": vif,
            "n_effective": float(n_total / vif),
        }
    return out


def drop_outlier_picks(events, rows, n_sigma=5.0, min_points_per_wave=2):
    """Remove picks whose residual is an outlier for its wave type.

    For each wave type, a pick is an outlier if its residual (from `rows`, the
    output of `standardized_residuals` for a preliminary fit) lies more than
    `n_sigma` robust standard deviations from that wave type's median residual,
    the robust standard deviation being 1.4826 * MAD (the median absolute
    deviation; equal to sigma for Gaussian data, but not inflated by the
    outliers themselves). These are the picks the per-frequency peak picker took
    from an FK sidelobe.

    Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). 
    Detecting outliers: Do not use standard deviation around the mean, 
    use absolute deviation around the median. Journal of Experimental Social
    Psychology, 49(4), 764-766. doi:10.1016/j.jesp.2013.03.013

    Returns (cleaned events, report): copies of the events without the outlier
    picks -- a wave left with fewer than `min_points_per_wave` picks is dropped,
    and an event left with no waves -- and one report row per wave type.
    """
    from .data import EventData, WaveObservation

    drop, report = set(), []
    for (wave,), group in sorted(_split_by(rows, "wave").items()):
        d = np.array([g["residual"] for g in group], float)
        med = float(np.median(d))
        robust_std = 1.4826 * float(np.median(np.abs(d - med)))
        out = np.abs(d - med) > n_sigma * robust_std
        drop |= {(g["event_id"], wave, g["frequency"]) for g, o in zip(group, out) if o}
        report.append({"wave": wave, "n": int(d.size), "n_removed": int(out.sum()),
                       "median": med, "robust_std": robust_std,
                       "threshold": n_sigma * robust_std})

    cleaned = []
    for event in events:
        waves = {}
        for wt, w in event.waves.items():
            f = np.asarray(w.frequency, float)
            keep = np.array([(event.event_id, wt, float(fj)) not in drop for fj in f])
            if keep.sum() < min_points_per_wave:
                continue
            sigma = np.asarray(w.sigma_k, float)
            waves[wt] = WaveObservation(wt, f[keep], np.asarray(w.k_obs, float)[keep],
                                        float(sigma) if sigma.ndim == 0 else sigma[keep],
                                        dict(w.metadata))
        if waves:
            cleaned.append(EventData(event.event_id, waves, dict(event.metadata)))
    return cleaned, report


def plot_sigma_and_gaussian(rows, waves=("FG", "P", "SH"), *, figsize=(13, 7), ylim=0.12,
                            hist_range=None):
    """Two-row summary of the error model, one column per mode.

    Top: every residual (rad/m) against frequency, where drift or fanning of the
    error with frequency shows up directly, on a common +/- `ylim`. The assumed
    sigma_m is drawn as +/- lines when it fits inside that range, otherwise
    quoted in the legend as a multiple of the residual RMS.

    Bottom: every residual of that mode pooled across frequency and record,
    with a Gaussian fitted to the pooled sample. The fitted width is what
    sigma_m should have been; departures of the histogram from the curve are
    the non-Gaussianity. `hist_range` defaults to a symmetric window sized from
    the fitted width, shared across the modes.

    Returns the figure.
    """

    names = {"FG": "QS", "P": "QS0", "SH": "SH0"}
    waves = [wt for wt in waves if any(row["wave"] == wt for row in rows)]
    if not waves:
        raise ValueError("None of the requested waves appear in `rows`.")

    fig, ax = plt.subplots(2, len(waves), figsize=figsize, squeeze=False)

    for col, wt in enumerate(waves):
        name = names.get(wt, wt)
        group = [row for row in rows if row["wave"] == wt]
        fg = np.array([g["frequency"] for g in group], float)
        dg = np.array([g["residual"] for g in group], float)

        # --- top: residuals against frequency ------------------------------
        a = ax[0, col]
        a.scatter(fg, dg, s=7, alpha=0.30, color="0.35", linewidths=0,
                  label=f"residuals (n={dg.size})")
        a.axhline(0.0, color="k", lw=0.8)
        a.set_xlabel("Frequency (Hz)")
        if col == 0:
            a.set_ylabel("Residual $k_{obs}-k_{pred}$ (rad/m)")
        a.set_title(f"{name}: residuals vs frequency")
        a.set_ylim(-ylim, ylim)
        a.grid(alpha=0.3, lw=0.5)
        a.legend(fontsize=14, loc="upper left", framealpha=0.9)

        # --- bottom: pooled residuals vs the Gaussian model ---------------
        # Unstandardized, so the width reads directly as a sigma_k in rad/m.
        a = ax[1, col]
        r = dg[np.isfinite(dg)]
        mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
        span = hist_range or (-(4 * sd + abs(mu)), 4 * sd + abs(mu))
        a.hist(np.clip(r, *span), bins=np.linspace(span[0], span[1], 41), density=True,
               color="steelblue", alpha=0.65, edgecolor="white", linewidth=0.4,
               label=f"residuals (n={r.size})")
        xs = np.linspace(span[0], span[1], 400)
        a.plot(xs, stats.norm.pdf(xs, mu, sd), "-", color="crimson", lw=2.0,
               label=f"Gaussian fit")
        a.set_xlim(*span)
        a.set_xlabel("Residual $k_{obs}-k_{pred}$ (rad/m)")
        if col == 0:
            a.set_ylabel("Density")
        a.set_title(f"{name}: pooled residuals vs Gaussian")
        a.grid(alpha=0.25, lw=0.5)
        a.legend(fontsize=14, loc="upper left")
        print(f"{name} Gaussian fit: $\\mu$={mu:.4f}, $\\hat{{\\sigma}}$={sd:.4f} rad/m")
    
    # Shared limits on the bottom row: the modes are only comparable if their axes are.
    half = max(abs(v) for a in ax[1] for v in a.get_xlim())
    for a in ax[1]:
        a.set_xlim(-half, half)

    fig.tight_layout()
    return fig
