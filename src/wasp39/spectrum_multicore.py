from __future__ import annotations

import os
import numpy as np
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional: clean progress bar for completed bins
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from .binning import make_binned_lightcurve_for_wlbin, bin_time_series
from .normalize import normalize_by_oot
from .mcmc import fit_white_light_mcmc
from .lightcurve import TransitConfig


def _fit_bin_depth_worker(
    k: int,
    w0: float,
    w1: float,
    t_use: np.ndarray,
    f_use: np.ndarray,
    e_use: np.ndarray,
    cfg: TransitConfig,
    rp_init: float,
) -> Tuple[int, float, float, float, float, float]:
    """Worker: fit Rp/R* for a single wavelength bin and return depth statistics.

    Returns:
        k, w0, w1, depth_med, depth_err_lo, depth_err_hi
    """
    # Import inside worker for Windows spawn safety
    from .mcmc import fit_white_light_mcmc

    samples_rp, _ = fit_white_light_mcmc(
        t_use,
        f_use,
        e_use,
        cfg,
        rp_init=rp_init,
        progress=False,  # avoid progress bars from multiple processes
    )

    depth_samples = samples_rp ** 2
    depth_med = float(np.median(depth_samples))
    d16, d84 = np.percentile(depth_samples, [16, 84])

    return k, w0, w1, depth_med, depth_med - float(d16), float(d84) - depth_med



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
    # Parallelism
    n_jobs: Optional[int] = None,
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
    # 5.2 — Loop wavelength bins, build LC, normalize, (parallel) MCMC fit
    # ------------------------------------------------------
    tasks: list[tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray]] = []

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

        # Defer the expensive per-bin MCMC to (optional) process pool
        tasks.append((k, w0, w1, np.asarray(t_use, float), np.asarray(f_use, float), np.asarray(e_use, float)))

    if len(tasks) == 0:
        raise RuntimeError("No valid wavelength bins were constructed (all bins skipped).")

    # ------------------------------------------------------
    # 5.2b — Run per-bin fits (parallel across bins)
    # ------------------------------------------------------
    results: dict[int, tuple[float, float, float, float, float]] = {}

    # Choose workers
    if n_jobs is None:
        # Leave one core free by default
        n_jobs_eff = max(1, (os.cpu_count() or 2) - 1)
    else:
        n_jobs_eff = max(1, int(n_jobs))

    # Don't spawn more workers than tasks
    n_jobs_eff = min(n_jobs_eff, len(tasks))

    do_parallel = (n_jobs_eff > 1)

    if verbose:
        mode = "parallel" if do_parallel else "serial"
        print(f"  STEP 5.2c — Per-bin execution: {mode} (workers={n_jobs_eff})")

    if do_parallel:
        # Windows-safe: worker is top-level + imports inside
        with ProcessPoolExecutor(max_workers=n_jobs_eff) as ex:
            futs = {
                ex.submit(_fit_bin_depth_worker, k, w0, w1, t_use, f_use, e_use, cfg, rp_init): k
                for (k, w0, w1, t_use, f_use, e_use) in tasks
            }
            iterator = as_completed(futs)
            if progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futs), desc="Bins", unit="bin")
            for fut in iterator:
                k, w0, w1, depth_med, elo, ehi = fut.result()
                results[k] = (w0, w1, depth_med, elo, ehi)
                if verbose:
                    print(f"      ✓ Bin {k+1:02d}/{n_wavelength_bins} done: depth={depth_med:.6f}")
    else:
        iterator = tasks
        if progress and tqdm is not None:
            iterator = tqdm(tasks, total=len(tasks), desc="Bins", unit="bin")
        for (k, w0, w1, t_use, f_use, e_use) in iterator:
            k, w0, w1, depth_med, elo, ehi = _fit_bin_depth_worker(k, w0, w1, t_use, f_use, e_use, cfg, rp_init)
            results[k] = (w0, w1, depth_med, elo, ehi)
            if verbose:
                print(f"      ✓ Bin {k+1:02d}/{n_wavelength_bins} done: depth={depth_med:.6f}")

    # ------------------------------------------------------
    # 5.3 — Assemble outputs in wavelength-bin order
    # ------------------------------------------------------
    for k in sorted(results.keys()):
        w0, w1, depth_med, elo, ehi = results[k]
        wl_centers.append(0.5 * (w0 + w1))
        depths.append(depth_med)
        depth_err_lo.append(elo)
        depth_err_hi.append(ehi)

    return (
        np.array(wl_centers, dtype=float),
        np.array(depths, dtype=float),
        np.array(depth_err_lo, dtype=float),
        np.array(depth_err_hi, dtype=float),
    )