from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def grid_param_names(grid) -> tuple[str, ...]:
    """Names of the free search parameters carried by a parameter-grid dataclass.

    Works for any frozen grid dataclass, so the rest of the package never
    hardcodes which parameters are being searched over.
    """
    return tuple(f.name for f in dataclasses.fields(grid))


@dataclass
class WaveObservation:
    """One extracted dispersion curve for one event and one wave type."""

    wave_type: str
    frequency: np.ndarray
    k_obs: np.ndarray
    sigma_k: float | np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, dtype=float)
        self.k_obs = np.asarray(self.k_obs, dtype=float)

        if self.frequency.ndim != 1 or self.k_obs.ndim != 1:
            raise ValueError("frequency and k_obs must be 1-D arrays.")
        if self.frequency.shape != self.k_obs.shape:
            raise ValueError("frequency and k_obs must have the same shape.")
        if self.frequency.size == 0:
            raise ValueError("Dispersion curve cannot be empty.")

        sigma = np.asarray(self.sigma_k, dtype=float)
        if sigma.ndim == 0:
            if float(sigma) <= 0:
                raise ValueError("sigma_k must be positive.")
        elif sigma.shape != self.frequency.shape:
            raise ValueError("Array sigma_k must match frequency shape.")
        elif np.any(sigma <= 0):
            raise ValueError("All sigma_k values must be positive.")


@dataclass
class EventData:
    """All available wave observations for one source/event."""

    event_id: str | int
    waves: dict[str, WaveObservation]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available_wave_types(self) -> tuple[str, ...]:
        return tuple(self.waves.keys())


@dataclass(frozen=True)
class ParameterGrid:
    """Rectangular posterior grid for (E, nu, h_i)."""

    E: np.ndarray
    nu: np.ndarray
    h_i: np.ndarray

    @classmethod
    def from_bounds(
        cls,
        E_bounds: tuple[float, float],
        nu_bounds: tuple[float, float],
        h_i_bounds: tuple[float, float],
        n_E: int = 41,
        n_nu: int = 41,
        n_h_i: int = 41,
    ) -> "ParameterGrid":
        return cls(
            E=np.linspace(*E_bounds, n_E),
            nu=np.linspace(*nu_bounds, n_nu),
            h_i=np.linspace(*h_i_bounds, n_h_i),
        )

@dataclass
class PosteriorResult:
    label: str
    map_params: dict[str, float]
    map_log_posterior: float
    log_posterior: np.ndarray | None = None
    posterior: np.ndarray | None = None
    parameter_axes: dict[str, np.ndarray] | None = None
    marginals: dict[str, np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def marginal_mean(self, name: str) -> float:
        if self.marginals is None or self.parameter_axes is None:
            raise ValueError("This result does not contain grid marginals.")
        pdf = self.marginals[name]
        axis = self.parameter_axes[name]
        return float(np.sum(axis * pdf) / np.sum(pdf))

    def marginal_std(self, name: str) -> float:
        if self.marginals is None or self.parameter_axes is None:
            raise ValueError("This result does not contain grid marginals.")
        pdf = self.marginals[name]
        axis = self.parameter_axes[name]
        mean = np.sum(axis * pdf) / np.sum(pdf)
        variance = np.sum((axis - mean) ** 2 * pdf) / np.sum(pdf)
        return float(np.sqrt(max(variance, 0.0)))

    def marginal_pmf(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (axis, normalized marginal probability mass) for a parameter."""
        if self.marginals is None or self.parameter_axes is None:
            raise ValueError("This result does not contain grid marginals.")
        axis = np.asarray(self.parameter_axes[name], float)
        pmf = np.asarray(self.marginals[name], float)
        total = pmf.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Marginal for '{name}' has no positive mass.")
        order = np.argsort(axis)
        return axis[order], pmf[order] / total

    def marginal_quantile(self, name: str, q) -> float | np.ndarray:
        """Quantile(s) of the marginal posterior, by interpolating the CDF.

        The grid marginal is a probability mass on cell centers; the CDF is
        evaluated at cell centers with the midpoint convention
        (cdf_i = sum_{j<i} p_j + p_i/2) and linearly interpolated.
        """
        axis, pmf = self.marginal_pmf(name)
        cdf = np.cumsum(pmf) - 0.5 * pmf
        out = np.interp(np.atleast_1d(np.asarray(q, float)), cdf, axis)
        return float(out[0]) if np.isscalar(q) or np.ndim(q) == 0 else out

    def marginal_credible_interval(
        self,
        name: str,
        level: float = 0.95,
        method: str = "equal_tailed",
    ) -> tuple[float, float]:
        """Credible interval of the 1-D marginal posterior.

        Parameters
        ----------
        name : parameter name.
        level : credible mass, e.g. 0.95 for a 95% interval.
        method : ``"equal_tailed"`` (default) puts (1-level)/2 in each tail and
            interpolates the CDF; ``"hpd"`` returns the highest-posterior-density
            interval, found by including grid cells in decreasing probability
            until ``level`` is reached (resolution-limited by the grid).
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1).")
        if method == "equal_tailed":
            tail = (1.0 - level) / 2.0
            lo, hi = self.marginal_quantile(name, [tail, 1.0 - tail])
            return float(lo), float(hi)
        if method == "hpd":
            axis, pmf = self.marginal_pmf(name)
            order = np.argsort(pmf)[::-1]
            csum = np.cumsum(pmf[order])
            k = int(np.searchsorted(csum, level) + 1)
            k = min(k, len(order))
            kept = axis[np.sort(order[:k])]
            return float(kept[0]), float(kept[-1])
        raise ValueError("method must be 'equal_tailed' or 'hpd'.")


def posterior_from_logp(logp, grid, label, metadata=None):
    """Normalized posterior, MAP and 1-D marginals from a log-posterior on `grid`.

    Shared by the pointwise (invert.py) and cached (cached_inversion.py) paths.
    """
    names = grid_param_names(grid)
    axes = [np.asarray(getattr(grid, n)) for n in names]
    finite = np.isfinite(logp)
    if not finite.any():
        raise RuntimeError("No finite posterior values.")
    maxlog = np.max(logp[finite])
    weights = np.zeros_like(logp, dtype=float)
    weights[finite] = np.exp(logp[finite] - maxlog)
    z = weights.sum()
    if not np.isfinite(z) or z <= 0:
        raise RuntimeError("Posterior normalization failed.")
    weights /= z
    idx = np.unravel_index(np.argmax(logp), logp.shape)
    all_axes = tuple(range(len(names)))
    return PosteriorResult(
        label=label,
        map_params={n: float(a[i]) for n, a, i in zip(names, axes, idx)},
        map_log_posterior=float(logp[idx]),
        log_posterior=logp,
        posterior=weights,
        parameter_axes={n: a for n, a in zip(names, axes)},
        marginals={n: weights.sum(axis=tuple(ax for ax in all_axes if ax != pos)) for pos, n in enumerate(names)},
        metadata=metadata or {},
    )


def set_wave_sigmas(events, sigma_by_wave):
    """Return copies of events with a chosen scalar sigma_k per wave type."""
    out = []
    for event in events:
        waves = {}
        for wt, wave in event.waves.items():
            sigma = sigma_by_wave.get(wt, wave.sigma_k)
            waves[wt] = WaveObservation(
                wt,
                np.asarray(wave.frequency).copy(),
                np.asarray(wave.k_obs).copy(),
                sigma,
                dict(wave.metadata),
            )
        out.append(EventData(event.event_id, waves, dict(event.metadata)))
    return out
