from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

# Optional progress bar (falls back gracefully if not installed)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from .binning import make_binned_lightcurve_for_wlbin, bin_time_series
from .normalize import normalize_by_oot
from .mcmc import fit_white_light_mcmc
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
    # Binning strategy
    #  - 'equal_pixels': each wavelength bin contains ~equal number of wavelength columns
    #  - 'uniform_wavelength': bins are uniform in wavelength span
    binning_mode: str = "equal_pixels",
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

    # Restrict to requested wavelength range (keeps binning honest)
    in_range = (wl >= wl_lo) & (wl <= wl_hi)
    wl = wl[in_range]
    f2 = f2[:, in_range]
    e2 = e2[:, in_range]

    # We'll use these cleaned arrays from here on
    wavelength_um = wl
    flux_2d = f2
    fluxerr_2d = e2

    # Build wavelength-bin definitions
    # Each entry: (i0, i1, w0, w1, w_center)
    bins: list[tuple[int, int, float, float, float]] = []

    mode = str(binning_mode).strip().lower()
    if mode not in {"equal_pixels", "uniform_wavelength"}:
        raise ValueError(
            f"Unknown binning_mode='{binning_mode}'. Expected 'equal_pixels' or 'uniform_wavelength'."
        )

    n_cols = int(wavelength_um.size)
    if n_cols < max(1, min_wl_pixels):
        raise ValueError(
            f"Too few wavelength columns in range {wl_lo}–{wl_hi} µm: {n_cols}. "
            f"Try widening wl_min/wl_max or lowering min_wl_pixels."
        )

    if mode == "uniform_wavelength":
        # Uniform wavelength edges over [wl_lo, wl_hi]
        edges = np.linspace(wl_lo, wl_hi, n_wavelength_bins + 1)
        for k in range(n_wavelength_bins):
            w0, w1 = float(edges[k]), float(edges[k + 1])
            i0 = int(np.searchsorted(wavelength_um, w0, side="left"))
            i1 = int(np.searchsorted(wavelength_um, w1, side="right"))
            if i1 - i0 < min_wl_pixels:
                continue
            w_center = float(np.mean(wavelength_um[i0:i1]))
            bins.append((i0, i1, w0, w1, w_center))
    else:
        # Equal-pixel binning: split wavelength columns into ~equal-sized groups.
        # This avoids "holes" when some wavelength regions are sparsely sampled.
        max_bins = max(1, n_cols // max(1, int(min_wl_pixels)))
        nbins_eff = int(min(int(n_wavelength_bins), max_bins))
        if nbins_eff < 1:
            nbins_eff = 1

        groups = np.array_split(np.arange(n_cols, dtype=int), nbins_eff)
        for g in groups:
            if g.size == 0:
                continue
            i0 = int(g[0])
            i1 = int(g[-1]) + 1
            if i1 - i0 < min_wl_pixels:
                # Should be rare due to nbins_eff selection; skip if it happens
                continue
            w0 = float(wavelength_um[i0])
            w1 = float(wavelength_um[i1 - 1])
            w_center = float(np.mean(wavelength_um[i0:i1]))
            bins.append((i0, i1, w0, w1, w_center))

    if len(bins) < 3:
        raise ValueError(
            f"Too few usable wavelength bins after '{mode}' binning: {len(bins)}. "
            f"Try lowering n_wavelength_bins or min_wl_pixels."
        )

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
    it = enumerate(bins, start=1)
    if progress and (tqdm is not None):
        # Materializing to a list keeps tqdm happy across environments.
        it = tqdm(list(it), total=len(bins), desc="Spectrum bins", unit="bin")

    for idx, (i0, i1, w0, w1, w_center) in it:
        n_pix = int(i1 - i0)
        if progress and (tqdm is not None) and hasattr(it, "set_postfix_str"):
            it.set_postfix_str(f"{w0:.3f}-{w1:.3f} um")

        # Build a 1D light curve for this wavelength range
        f_bin, e_bin = make_binned_lightcurve_for_wlbin(flux_2d, fluxerr_2d, i0, i1)

        # Normalize by out-of-transit (OOT)
        f_norm, e_norm = normalize_by_oot(f_bin, e_bin, oot_index)
        if (not np.all(np.isfinite(f_norm))) or (not np.all(np.isfinite(e_norm))):
            if verbose:
                print(f"      ! Bin {idx:02d}: non-finite after OOT normalize; skipping")
            continue

        # Optional time binning (often helps speed/stability)
        if time_bin_factor is not None and int(time_bin_factor) > 1:
            t_use, f_use, e_use = bin_time_series(bjd, f_norm, e_norm, int(time_bin_factor))
        else:
            t_use, f_use, e_use = bjd, f_norm, e_norm

        if verbose:
            print(f"    - Bin {idx:02d}/{len(bins)}: {w0:.3f}–{w1:.3f} µm "
                  f"(pixels={n_pix}, points={len(t_use)})")

        # Filter invalid points (prevents NaNs in log-likelihood)
        good = np.isfinite(t_use) & np.isfinite(f_use) & np.isfinite(e_use) & (e_use > 0)
        t_use, f_use, e_use = t_use[good], f_use[good], e_use[good]
        if t_use.size < 20:
            if verbose:
                print(f"      ! Bin {idx:02d}: too few valid points after filtering; skipping")
            continue

        # Error floor to prevent inv_sigma2 blow-ups
        e_use = np.maximum(e_use, 1e-8)

        # Run MCMC to fit Rp/R* for this bin
        try:
            chain, labels, best_params, best_model = fit_white_light_mcmc(
                t_use, f_use, e_use, cfg,
                rp_init=rp_init,
                progress=progress,
            )
        except ValueError as ex:
            # emcee can throw if logprob returns NaN; skip this bin
            if verbose:
                print(f"      ! Bin {idx:02d}: MCMC failed ({ex}); skipping")
            continue

        try:
            rp_idx = list(labels).index("rp")
        except ValueError:
            rp_idx = 4

        samples_rp = np.asarray(chain[:, rp_idx], dtype=float)
        samples_rp = samples_rp[np.isfinite(samples_rp)]
        if samples_rp.size < 50:
            if verbose:
                print(f"      ! Bin {idx:02d}: too few finite rp samples ({samples_rp.size}); skipping")
            continue

        # 5.3 — Build transmission depth posterior
        depth_samples = samples_rp ** 2
        depth_med = float(np.median(depth_samples))
        d16, d84 = np.percentile(depth_samples, [16, 84])

        wl_centers.append(float(w_center))
        depths.append(depth_med)
        depth_err_lo.append(depth_med - float(d16))
        depth_err_hi.append(float(d84) - depth_med)

    return (
        np.array(wl_centers, dtype=float),
        np.array(depths, dtype=float),
        np.array(depth_err_lo, dtype=float),
        np.array(depth_err_hi, dtype=float),
    )
