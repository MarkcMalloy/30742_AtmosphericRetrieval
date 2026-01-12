from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from .binning import make_binned_lightcurve_for_wlbin, bin_time_series
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
    # Explicit finite wavelength span (Proj_WASP-style)
    wl_min: float = 0.5,
    wl_max: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    STEP 5 — Transmission Spectrum Construction
      5.1 Divide the light curve into wavelength bins
      5.2 Fit each bin via MCMC (extract Rp/R*)
      5.3 Construct transmission spectrum (depth = (Rp/R*)^2)

    This version is robust to NaNs in the wavelength array by:
      - filtering non-finite wavelengths
      - sorting wavelengths if needed
      - applying the same mask/order to flux_2d and fluxerr_2d
      - constructing bin edges over an explicit finite wavelength range [wl_min, wl_max]

    Returns:
        wl_centers_um, depth, depth_err_lo, depth_err_hi
        where depth = (Rp/R*)^2 and errors are 16–84 percentile half-intervals.
    """

    # ------------------------------------------------------
    # 5.1 — Clean wavelength axis to avoid NaN bin edges
    # ------------------------------------------------------
    wl_raw = np.asarray(wavelength_um, dtype=float)

    finite = np.isfinite(wl_raw)
    n_finite = int(np.count_nonzero(finite))
    if n_finite < 10:
        raise ValueError(
            f"Too few finite wavelength points: {n_finite}/{wl_raw.size}. "
            "Wavelength array appears mostly NaN/inf."
        )

    wl = wl_raw[finite]
    f2 = np.asarray(flux_2d, dtype=float)[:, finite]
    e2 = np.asarray(fluxerr_2d, dtype=float)[:, finite]

    # Ensure wavelength is increasing; if not, sort and permute columns
    if np.any(np.diff(wl) < 0):
        order = np.argsort(wl)
        wl = wl[order]
        f2 = f2[:, order]
        e2 = e2[:, order]

    # Clip explicit span to what exists in data (so we don't create bins entirely out of range)
    data_min = float(np.min(wl))
    data_max = float(np.max(wl))
    wl_lo = max(float(wl_min), data_min)
    wl_hi = min(float(wl_max), data_max)

    if not np.isfinite(wl_lo) or not np.isfinite(wl_hi) or wl_hi <= wl_lo:
        raise ValueError(
            f"Invalid wavelength span after clipping: wl_lo={wl_lo}, wl_hi={wl_hi}. "
            f"(data range: {data_min}–{data_max} µm)"
        )

    # Explicit finite bin edges (Proj_WASP-style)
    edges = np.linspace(wl_lo, wl_hi, n_wavelength_bins + 1)

    # We'll use these cleaned arrays from here on
    wavelength_um = wl
    flux_2d = f2
    fluxerr_2d = e2

    wl_centers: list[float] = []
    depths: list[float] = []
    depth_err_lo: list[float] = []
    depth_err_hi: list[float] = []

    if verbose:
        print(f"  STEP 5.1 — Wavelength binning: {n_wavelength_bins} bins")
        print(f"  Wavelength sanity: finite={n_finite}/{wl_raw.size}, "
              f"data_range={data_min:.4f}–{data_max:.4f} µm, "
              f"using_range={wl_lo:.4f}–{wl_hi:.4f} µm")
        print(f"  STEP 5.2 — Per-bin MCMC: Rp/R* (progress={'on' if progress else 'off'})")
        if time_bin_factor is not None:
            print(f"  STEP 5.2b — Time binning enabled: factor={time_bin_factor}")

    # ------------------------------------------------------
    # 5.2 — Loop wavelength bins, build LC, normalize, MCMC fit
    # ------------------------------------------------------
    for k in range(n_wavelength_bins):
        w0, w1 = float(edges[k]), float(edges[k + 1])

        # Skip bins that fall outside available wavelength range (should be rare due to clipping)
        if w1 <= wavelength_um[0] or w0 >= wavelength_um[-1]:
            if verbose:
                print(f"    - Bin {k+1:02d}/{n_wavelength_bins}: {w0:.3f}–{w1:.3f} µm "
                      f"SKIP (outside data range)")
            continue

        # Find indices in wavelength array for this bin
        i0 = int(np.searchsorted(wavelength_um, w0, side="left"))
        i1 = int(np.searchsorted(wavelength_um, w1, side="right"))
        n_pix = i1 - i0

        if n_pix < min_wl_pixels:
            if verbose:
                print(f"    - Bin {k+1:02d}/{n_wavelength_bins}: {w0:.3f}–{w1:.3f} µm "
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
            print(f"    - Bin {k+1:02d}/{n_wavelength_bins}: {w0:.3f}–{w1:.3f} µm "
                  f"(pixels={n_pix}, points={len(t_use)})")

        # Run MCMC to fit Rp/R* for this bin
        samples_rp, _ = fit_one_bin_mcmc(
            t_use, f_use, e_use, cfg,
            rp_init=rp_init,
            progress=progress,
        )

        # 5.3 — Build transmission depth posterior
        depth_samples = samples_rp ** 2
        depth_med = float(np.median(depth_samples))
        d16, d84 = np.percentile(depth_samples, [16, 84])

        wl_centers.append(0.5 * (w0 + w1))
        depths.append(depth_med)
        depth_err_lo.append(depth_med - float(d16))
        depth_err_hi.append(float(d84) - depth_med)

    return (
        np.array(wl_centers, dtype=float),
        np.array(depths, dtype=float),
        np.array(depth_err_lo, dtype=float),
        np.array(depth_err_hi, dtype=float),
    )
