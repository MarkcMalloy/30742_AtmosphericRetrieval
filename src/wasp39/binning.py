from __future__ import annotations

import numpy as np
from typing import Tuple


def bin_time_series(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    binning_factor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple uniform binning in time order.
    Returns binned t, y, yerr (propagated as sqrt(sum(err^2))/N in each bin).
    """
    n = len(t)
    n_bins = n // binning_factor
    t_b = np.zeros(n_bins)
    y_b = np.zeros(n_bins)
    e_b = np.zeros(n_bins)

    for i in range(n_bins):
        start = i * binning_factor
        end = start + binning_factor
        t_b[i] = np.mean(t[start:end])
        y_b[i] = np.mean(y[start:end])
        # average of independent measurements
        e_b[i] = np.sqrt(np.sum(yerr[start:end] ** 2)) / (end - start)

    return t_b, y_b, e_b


def make_wavelength_bins(wavelength_um: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Create wavelength bin edges over the wavelength grid.

    Returns:
        edges: shape (n_bins+1,)
    """
    wl_min = float(np.min(wavelength_um))
    wl_max = float(np.max(wavelength_um))
    return np.linspace(wl_min, wl_max, n_bins + 1)


def make_binned_lightcurve_for_wlbin(
    flux_2d: np.ndarray,
    fluxerr_2d: np.ndarray,
    i0: int,
    i1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin/integrate the spectroscopic light curve across wavelength indices [i0, i1).
    Uses inverse-variance weighting for the mean flux in each integration.

    Returns:
        flux_bin: (n_int,)
        err_bin: (n_int,)
    """
    f = flux_2d[:, i0:i1]
    e = fluxerr_2d[:, i0:i1]
    w = np.where(e > 0, 1.0 / (e ** 2), 0.0)
    wsum = np.sum(w, axis=1)
    flux_bin = np.sum(w * f, axis=1) / np.where(wsum > 0, wsum, np.nan)
    err_bin = np.sqrt(1.0 / np.where(wsum > 0, wsum, np.nan))
    return flux_bin, err_bin
