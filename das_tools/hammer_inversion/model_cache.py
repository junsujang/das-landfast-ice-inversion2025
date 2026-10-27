"""Flexural-gravity forward model precomputed over the parameter grid.

FG has no closed form: k(f) comes from a root solve, so it is evaluated once for
every (E, nu, h_i) grid node on a uniform frequency grid and interpolated to each
event's observed frequencies during the inversion. P (QS0) and SH0 are closed form
and are evaluated directly (see `cached_inversion.predict_k`), so they are not
cached.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from das_tools.physics import dispersion_relations as dr

MODEL_CONSTANTS_KEYS = ("h_w", "rho_i", "rho_w", "P")


def build_fg_cache(grid, model_constants, frequency, dtype=np.float32, chunk_elements=2_000_000):
    """k_FG(f) for every node of `grid` (E, nu, h_i), shape (nE, nnu, nh, nf).

    `model_constants` fixes h_w, rho_i, rho_w and P. Solved with the vectorized
    root finder (`dispersion_relations.fg_disp_eq_ml_c_vec`) in chunks along E,
    so the solver's temporary arrays stay bounded; a ~5e5-node grid takes
    seconds. Stored as float32 (k to ~1e-7 relative, far below sigma_k).
    """
    missing = set(MODEL_CONSTANTS_KEYS) - set(model_constants)
    if missing:
        raise ValueError(f"model_constants must fix {sorted(missing)}.")
    f = np.asarray(frequency, dtype=float)
    if f.ndim != 1 or len(f) < 2 or np.any(np.diff(f) <= 0):
        raise ValueError("frequency must be a 1-D increasing grid.")

    E, nu, h_i = (np.asarray(getattr(grid, n), float) for n in ("E", "nu", "h_i"))
    c = {k: float(model_constants[k]) for k in MODEL_CONSTANTS_KEYS}
    k = np.empty((E.size, nu.size, h_i.size, f.size), dtype=dtype)
    step = max(1, int(chunk_elements // (nu.size * h_i.size * f.size)))
    for start in range(0, E.size, step):
        s = slice(start, min(start + step, E.size))
        k[s] = dr.fg_disp_eq_ml_c_vec(
            f[None, None, None, :], E[s, None, None, None], nu[None, :, None, None],
            h_i[None, None, :, None], c["h_w"], c["rho_i"], c["rho_w"], c["P"],
        )

    metadata = {
        "grid": {"E": E.tolist(), "nu": nu.tolist(), "h_i": h_i.tolist()},
        "model_constants": c,
    }
    return {"frequency": f, "k": k, "metadata": metadata}


def save_fg_cache(cache, filepath, overwrite=False):
    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"{filepath} exists. Use overwrite=True to rebuild.")
    np.savez_compressed(filepath, frequency=cache["frequency"], k=cache["k"],
                        metadata=np.array(json.dumps(cache["metadata"])))
    print(f"Saved FG cache to {filepath}")


def load_fg_cache(filepath):
    """Read a cache written by `save_fg_cache`."""
    with np.load(Path(filepath), allow_pickle=False) as data:
        if "k" not in data.files:
            raise ValueError(f"{filepath} is not an FG cache written by save_fg_cache (an older "
                             "format?); rebuild it with get_fg_cache.")
        return {"frequency": data["frequency"], "k": data["k"],
                "metadata": json.loads(data["metadata"].item())}


def validate_fg_cache(cache, grid, events=None, model_constants=None):
    """Raise ValueError if `cache` was not built for this grid / constants, or
    an event's FG picks fall outside the cached frequency range."""
    meta = cache["metadata"]
    for name in ("E", "nu", "h_i"):
        cached = np.asarray(meta["grid"][name], dtype=float)
        current = np.asarray(getattr(grid, name), dtype=float)
        if cached.shape != current.shape or not np.allclose(cached, current, rtol=0, atol=1e-12):
            raise ValueError(f"Cache parameter grid mismatch for {name}; rebuild the cache.")

    for key, value in (model_constants or {}).items():
        cached = meta["model_constants"].get(key)
        if cached is None or not np.isclose(float(cached), float(value), rtol=0, atol=1e-12):
            raise ValueError(f"Cache/model mismatch for {key}: cache={cached}, requested={value}.")

    f = cache["frequency"]
    for event in events or []:
        if "FG" not in event.waves:
            continue
        fe = np.asarray(event.waves["FG"].frequency, float)
        fe = fe[np.isfinite(fe) & (fe > 0)]
        if fe.size and (fe.min() < f[0] - 1e-9 or fe.max() > f[-1] + 1e-9):
            raise ValueError(
                f"Event {event.event_id} FG spans {fe.min():.2f}-{fe.max():.2f} Hz, "
                f"outside cache {f[0]:.2f}-{f[-1]:.2f} Hz. Widen the cache frequency grid."
            )


def get_fg_cache(filepath, grid, model_constants, frequency, rebuild=False, save=True):
    """The FG cache for (grid, model_constants, frequency): loaded or built.

    The file at `filepath` is reused only if it was built for exactly this
    parameter grid, these constants and this frequency grid; otherwise -- or
    with rebuild=True -- it is built with `build_fg_cache` and, with save=True,
    written to `filepath`. A cache can therefore never be silently reused after
    the grid, a constant or the band has changed.
    """
    filepath = Path(filepath)
    frequency = np.asarray(frequency, dtype=float)

    if filepath.exists() and not rebuild:
        try:
            cache = load_fg_cache(filepath)
            validate_fg_cache(cache, grid, model_constants=model_constants)
            if np.array_equal(cache["frequency"], frequency):
                return cache
        except (KeyError, ValueError):
            pass
        print(f"{filepath.name} was built for a different grid, constants or band; rebuilding")

    cache = build_fg_cache(grid, model_constants, frequency)
    if save:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        save_fg_cache(cache, filepath, overwrite=True)
    return cache
