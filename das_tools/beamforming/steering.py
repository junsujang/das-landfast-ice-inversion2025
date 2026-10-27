from pathlib import Path
import numpy as np
from dataclasses import dataclass

from das_tools.physics import dispersion_relations as dr


@dataclass
class IceConfig:
    nu: float = 0.22
    rho_i: float = 910.0
    rho_w: float = 1025.0
    h_w: float = 10.0
    h_i: float = 0.98
    E_min: float = 0.3e9
    E_max: float = 10.0e9
    E_step: float = 0.1e9
    C_min: float = 0.0
    C_max: float = 3.0e6
    C_step: float = 0.1e6

    def get_E_All(self):
        return np.arange(self.E_min, self.E_max, self.E_step, dtype='float')

    def get_C_All(self):
        return np.arange(self.C_min, self.C_max, self.C_step, dtype='float')


def compute_disp_curves(cache_path: Path,
                        freq: np.ndarray,
                        theta_rad: np.ndarray,
                        cfg: IceConfig):
    """FG dispersion curves k(f) for every (E, P) on the cfg grid.

    A curve is valid when it is subcritical (dr.fg_is_supercritical) and finite at
    every frequency. All curves are solved at once by dr.fg_disp_eq_ml_c_vec.
    """
    freq = np.asarray(freq, dtype=float)
    theta_rad = np.asarray(theta_rad, dtype=float)

    if freq.ndim != 1 or freq.size == 0:
        raise ValueError("freq must be a non-empty 1D array")
    if theta_rad.ndim != 1 or theta_rad.size == 0:
        raise ValueError("theta_rad must be a non-empty 1D array")
    if np.any(freq <= 0):
        raise ValueError("All frequencies must be > 0 for the dispersion solver.")

    E_all = cfg.get_E_All()
    P_all = cfg.get_C_All()

    # (nE, nC, nF)
    k_disp = dr.fg_disp_eq_ml_c_vec(
        freq[None, None, :], E_all[:, None, None], cfg.nu, cfg.h_i, cfg.h_w,
        cfg.rho_i, cfg.rho_w, P_all[None, :, None],
    )
    supercritical = dr.fg_is_supercritical(E_all[:, None], cfg.nu, cfg.h_i, cfg.rho_w,
                                           P_all[None, :])
    k_disp[supercritical] = np.nan
    valid_disp = ~supercritical & np.all(np.isfinite(k_disp), axis=-1)

    valid_pairs = np.argwhere(valid_disp).astype(np.int32)
    k_disp_valid = k_disp[valid_disp]
    if cache_path != None:
        np.savez(
            cache_path,
            nu=cfg.nu,
            rho_i=cfg.rho_i,
            rho_w=cfg.rho_w,
            h_i=cfg.h_i,
            h_w=cfg.h_w,
            E_all=E_all,
            P_all=P_all,
            theta=theta_rad,
            freq=freq,
            k_disp=k_disp,
            valid_disp=valid_disp,
            valid_pairs=valid_pairs,
            k_disp_valid=k_disp_valid,
        )

    return {
        "nu": cfg.nu,
        "rho_i": cfg.rho_i,
        "rho_w": cfg.rho_w,
        "h_i": cfg.h_i,
        "h_w": cfg.h_w,
        "E_all": E_all,
        "P_all": P_all,
        "theta": theta_rad,
        "freq": freq,
        "k_disp": k_disp,
        "valid_disp": valid_disp,
        "valid_pairs": valid_pairs,
        "k_disp_valid": k_disp_valid,
    }


def _disp_cache_matches(cached: dict, freq: np.ndarray, cfg: IceConfig) -> bool:
    """True if a cached dispersion grid was built for exactly this band + ice config."""
    try:
        if cached["freq"].shape != freq.shape:
            return False
        if not np.allclose(cached["freq"], freq):
            return False
        for name, want in (
            ("nu", cfg.nu),
            ("rho_i", cfg.rho_i),
            ("rho_w", cfg.rho_w),
            ("h_i", cfg.h_i),
            ("h_w", cfg.h_w),
        ):
            if not np.isclose(float(cached[name]), want):
                return False
        for name, want in (("E_all", cfg.get_E_All()), ("P_all", cfg.get_C_All())):
            if cached[name].shape != want.shape or not np.allclose(cached[name], want):
                return False
    except KeyError:
        return False
    return True


def load_or_build_disp_curves(cache_path: Path,
                              freq: np.ndarray,
                              theta_rad: np.ndarray,
                              cfg: IceConfig) -> dict:
    """Dispersion-curve grid for one band, cached to disk.

    The grid depends only on (freq bins, ice config), never on the data, so it is
    built once per band and reused by every time block. The cache is rebuilt if
    the frequency axis or any ice parameter has changed.
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        with np.load(cache_path) as npz:
            cached = {k: npz[k] for k in npz.files}
        if _disp_cache_matches(cached, freq, cfg):
            return cached
        print(f"[disp] {cache_path.name} does not match current config -- rebuilding")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return compute_disp_curves(cache_path, freq, theta_rad, cfg)


def load_dispersion_cache(cache_path: Path) -> dict:
    """Read a grid written by compute_disp_curves; P_all is returned as C_all."""
    with np.load(cache_path) as data:
        required = {"theta", "freq", "k_disp_valid", "valid_pairs"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Dispersion cache missing required fields: {sorted(missing)}"
            )

        return {
            "theta": data["theta"],
            "freq": data["freq"],
            "E_all": data["E_all"],
            "C_all": data["P_all"],
            "k_disp": data["k_disp"],
            "valid_disp": data["valid_disp"],
            "k_disp_valid": data["k_disp_valid"],
            "valid_pairs": data["valid_pairs"],
        }
