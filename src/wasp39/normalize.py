from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def normalize_by_oot(
    flux: np.ndarray,
    flux_err: np.ndarray,
    oot_index: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize flux by the median out-of-transit flux.
    If oot_index is None, normalizes by the global median (fallback).
    """
    if oot_index is None or len(oot_index) == 0:
        med = np.nanmedian(flux)
    else:
        med = np.nanmedian(flux[oot_index])
    return flux / med, flux_err / med


def normalize_white_light(
    flux_2d: np.ndarray,
    fluxerr_2d: np.ndarray,
    oot_index: Optional[np.ndarray] = None,
    wl_bin_start: int = 83,
    wl_bin_end: int = 339,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    STEP 1 — Construct a white-light curve by summing flux over wavelength bins,
    then normalize using out-of-transit points (OOT).

    Returns:
        white_norm, white_err_norm  (shape: Ntime,)
    """
    # Sum over wavelength range
    white = np.nansum(flux_2d[:, wl_bin_start:wl_bin_end], axis=1)

    # Propagate errors in quadrature for summed flux
    white_err = np.sqrt(np.nansum(fluxerr_2d[:, wl_bin_start:wl_bin_end] ** 2, axis=1))

    # Normalize by OOT
    white_norm, white_err_norm = normalize_by_oot(white, white_err, oot_index)
    return white_norm, white_err_norm
