import numpy as np


def compute_stat(param_all,param_grid):

    # New Edition
    P_grid_MPa = np.asarray(param_grid, dtype=float)

    P_l68_new = []
    P_u68_new = []
    P_l95_new = []
    P_u95_new = []
    P_med_new = []

    for p_P in param_all:
        p_P = np.asarray(p_P, dtype=float)

        # Normalize defensively
        total = np.nansum(p_P)
        if not np.isfinite(total) or total <= 0:
            P_l68_new.append(np.nan)
            P_u68_new.append(np.nan)
            P_l95_new.append(np.nan)
            P_u95_new.append(np.nan)
            P_med_new.append(np.nan)
            continue

        p_P = p_P / total
        cdf_P = np.cumsum(p_P)

        P_l68_new.append(np.interp(0.16, cdf_P, P_grid_MPa))
        P_u68_new.append(np.interp(0.84, cdf_P, P_grid_MPa))

        P_l95_new.append(np.interp(0.025, cdf_P, P_grid_MPa))
        P_u95_new.append(np.interp(0.975, cdf_P, P_grid_MPa))

        P_med_new.append(np.interp(0.50, cdf_P, P_grid_MPa))

    P_l68_all = np.array(P_l68_new)
    P_u68_all = np.array(P_u68_new)
    P_l95_all = np.array(P_l95_new)
    P_u95_all = np.array(P_u95_new)
    P_med_all = np.array(P_med_new)
    # END Edition
    return P_l68_all, P_u68_all, P_l95_all, P_u95_all, P_med_all, P_grid_MPa