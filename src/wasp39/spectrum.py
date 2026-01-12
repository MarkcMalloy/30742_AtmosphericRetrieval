from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from .binning import make_wavelength_bins, make_binned_lightcurve_for_wlbin, bin_time_series
from .normalize import normalize_by_oot
from .mcmc import fit_one_bin_mcmc
from .lightcurve import TransitConfig


def construct_transmission_spectrum(
    bjd: np.ndarray,
    wavelength_um: np.ndarray,
    flux_2d: np.ndarray,
    fluxerr_2d: np.ndarray,
    cfg: TransitConfig,
    n_wavelength_bins: int = 30,
    oot_index: Optional[np.ndarray] = None,
    rp_init: float = 0.1457,
    *,
    # New knobs for "Proj_WASP-like" behavior + visibility
    progress: bool = True,
    verbose: bool = True,
    time_bin_factor: Optional[int] = None,
    min_wl_pixels: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    STEP 5 — Transmission Spectrum Construction
      5.1 Divide the light curve into wavelength bins
      5.2 Fit each bin via MCMC (extract Rp/R*)
      5.3 Construct transmission spectrum (depth = (Rp/R*)^2)

    Args:
        bjd: time array, shape (Ntime,)
        wavelength_um: wavelength array, shape (Nwl,)
        flux_2d: flux, shape (Ntime, Nwl)
        fluxerr_2d: flux errors, shape (Ntime, Nwl)
        cfg: TransitConfig (t0, period, a/R*, inc, limb darkening)
        n_wavelength_bins: number of wavelength bins
        oot_index: optional indices for out-of-transit points (same length as bjd)
        rp_init: initial Rp/R* for MCMC walkers

        progress: if True, show emcee progress for each bin
        verbose: if True, print bin-by-bin status
        time_bin_factor: optionally bin the time series before MCMC (e.g. 10)
        min_wl_pixels: skip bins that contain fewer wavelength pixels than this

    Returns:
        wl_centers_um, depth, depth_err_lo, depth_err_hi
        where depth = (Rp/R*)^2 and errors are 16–84 percentile half-intervals.
    """
    edges = make_wavelength_bins(wavelength_um, n_wavelength_bins)

    wl_centers = []
    depths = []
    depth_err_lo = []
    depth_err_hi = []

    if verbose:
        print(f"  STEP 5.1 — Wavelength binning: {n_wavelength_bins} bins")
        print(f"  STEP 5.2 — Per-bin MCMC: Rp/R* (progress={'on' if progress else 'off'})")
        if time_bin_factor is not None:
            print(f"  STEP 5.2b — Time binning enabled: factor={time_bin_factor}")

    for k in range(n_wavelength_bins):
        wl0, wl1 = float(edges[k]), float(edges[k + 1])

        # Find indices in wavelength array for this bin
        i0 = int(np.searchsorted(wavelength_um, wl0, side="left"))
        i1 = int(np.searchsorted(wavelength_um, wl1, side="right"))
        n_pix = i1 - i0

        if n_pix < min_wl_pixels:
            if verbose:
                print(f"    - Bin {k+1:02d}/{n_wavelength_bins}: {wl0:.3f}–{wl1:.3f} µm "
                      f"SKIP (only {n_pix} wavelength pixels)")
            continue

        # Build a 1D light curve for this wavelength range
        f_bin, e_bin = make_binned_lightcurve_for_wlbin(flux_2d, fluxerr_2d, i0, i1)

        # Normalize by out-of-transit (OOT)
        f_norm, e_norm = normalize_by_oot(f_bin, e_bin, oot_index)

        # Optional time binning (often helps speed/stability)
        if time_bin_factor is not None and time_bin_factor > 1:
            t_use, f_use, e_use = bin_time_series(bjd, f_norm, e_norm, time_bin_factor)
        else:
            t_use, f_use, e_use = bjd, f_norm, e_norm

        if verbose:
            print(f"    - Bin {k+1:02d}/{n_wavelength_bins}: {wl0:.3f}–{wl1:.3f} µm "
                  f"(pixels={n_pix}, points={len(t_use)})")

        # Run MCMC to fit Rp/R* for this bin
        samples_rp, _ = fit_one_bin_mcmc(
            t_use, f_use, e_use, cfg,
            rp_init=rp_init,
            progress=progress,
        )

        # Convert posterior to depth posterior properly
        depth_samples = samples_rp ** 2
        depth_med = np.median(depth_samples)
        d16, d84 = np.percentile(depth_samples, [16, 84])

        wl_centers.append(0.5 * (wl0 + wl1))
        depths.append(depth_med)
        depth_err_lo.append(depth_med - d16)
        depth_err_hi.append(d84 - depth_med)

    return (
        np.array(wl_centers, dtype=float),
        np.array(depths, dtype=float),
        np.array(depth_err_lo, dtype=float),
        np.array(depth_err_hi, dtype=float),
    )
