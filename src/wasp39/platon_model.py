# src/wasp39/platon_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from platon.constants import R_sun, M_jup
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.abundance_getter import AbundanceGetter


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PlatonPlanetStar:
    # White-light / literature inputs
    rp_over_rs: float = 0.15
    rstar_rsun: float = 0.895
    mplanet_mjup: float = 0.281

    # Atmosphere / spectrum inputs
    temperature_k: float = 1175.0
    logZ: float = 0.2                 # Used for equilibrium chemistry OR as seed for AbundanceGetter
    CO_ratio: float = 0.55            # Used for equilibrium chemistry OR as seed for AbundanceGetter
    cloudtop_pressure_pa: float = 1e3


def load_binned_spectrum(path: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load binned transmission spectrum.

    Accepts:
      - 4 cols: wavelength_um, depth, err_lo, err_hi
      - 3 cols: wavelength_um, depth, err
      - 2 cols: wavelength_um, depth  (err set to nan)

    Returns:
      wl_um, depth, err (1-sigma symmetric; for 4-col uses mean of lo/hi)
    """
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected spectrum format in {path}: shape={arr.shape}")

    wl_um = arr[:, 0].astype(float)
    depth = arr[:, 1].astype(float)

    if arr.shape[1] >= 4:
        err = 0.5 * (np.abs(arr[:, 2]) + np.abs(arr[:, 3])).astype(float)
    elif arr.shape[1] == 3:
        err = np.abs(arr[:, 2]).astype(float)
    else:
        err = np.full_like(depth, np.nan, dtype=float)

    return wl_um, depth, err


def _build_custom_abundances_from_getter(
    *,
    logZ: float,
    CO_ratio: float,
    overrides_vmr: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    Use PLATON AbundanceGetter to produce correctly-shaped abundance arrays,
    then apply constant-with-altitude overrides.

    overrides_vmr: species -> scalar VMR. Example:
        {"He":0.06, "H2O":1e-4, "CO2":1e-5, "CO":3e-4, "CH4":1e-6}

    Policy:
    - We keep the AbundanceGetter grid (correct shapes).
    - For each species in overrides_vmr, we set that species everywhere to the scalar.
    - If "H2" is NOT provided, we fill the remaining VMR into H2 (clipped at >=0).
    - If "H2" IS provided, we do not auto-fill.
    """
    getter = AbundanceGetter()
    abund = getter.get(logZ, CO_ratio)  # species -> ndarray with PLATON-required shape

    # Validate the override species exist
    for sp in overrides_vmr:
        if sp not in abund:
            known = ", ".join(sorted(abund.keys()))
            raise ValueError(f"Unknown species '{sp}' for PLATON. Known species include: {known}")

    # Apply overrides
    for sp, vmr in overrides_vmr.items():
        abund[sp] = np.zeros_like(abund[sp]) + float(vmr)

    # Optionally fill remainder into H2
    if "H2" in abund and "H2" not in overrides_vmr:
        # Compute sum across all species at each grid element
        total = None
        for arr in abund.values():
            if total is None:
                total = np.zeros_like(arr, dtype=float)
            total = total + np.asarray(arr, dtype=float)

        remainder = 1.0 - total
        remainder = np.clip(remainder, 0.0, 1.0)
        abund["H2"] = abund["H2"] + remainder

    return abund


def compute_platon_transit_depths(
    cfg: PlatonPlanetStar = PlatonPlanetStar(),
    wavelengths_um: Optional[np.ndarray] = None,
    n_wavelengths: int = 600,
    wl_min_um: float = 0.6,
    wl_max_um: float = 5.2,
    out_txt: Optional[PathLike] = None,
    out_npz: Optional[PathLike] = None,
    out_png: Optional[PathLike] = None,
    abundance_overrides_vmr: Optional[Dict[str, float]] = None,
    zero_opacities: Optional[list[str]] = None,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute wavelength-dependent transit depths using PLATON TransitDepthCalculator.

    Modes:
    - Equilibrium chemistry: abundance_overrides_vmr is None -> uses cfg.logZ & cfg.CO_ratio
    - Custom abundances via AbundanceGetter: abundance_overrides_vmr provided (scalar VMRs)
      -> internally uses AbundanceGetter().get(cfg.logZ, cfg.CO_ratio) and then applies overrides,
         then calls compute_depths with logZ=None and CO_ratio=None (PLATON requirement).

    Returns:
      (wavelength_um, depth)
    """

    # Convert white-light Rp/R* -> physical radii (SI)
    rstar_m = cfg.rstar_rsun * R_sun
    rplanet_m = cfg.rp_over_rs * rstar_m
    mplanet_kg = cfg.mplanet_mjup * M_jup

    if debug:
        print("Rp/R* target:", cfg.rp_over_rs)
        print("Depth target (rp^2):", cfg.rp_over_rs**2)
        print("Rstar [m]:", rstar_m)
        print("Rplanet [m]:", rplanet_m)
        print("Rplanet/Rstar:", rplanet_m / rstar_m)

    if wavelengths_um is None:
        wavelengths_um = np.linspace(wl_min_um, wl_max_um, n_wavelengths)
    wl_req_um = np.asarray(wavelengths_um, dtype=float)

    calc = TransitDepthCalculator()

    # Build kwargs for PLATON
    kwargs = dict(
        star_radius=rstar_m,
        planet_mass=mplanet_kg,
        planet_radius=rplanet_m,
        temperature=cfg.temperature_k,
        cloudtop_pressure=cfg.cloudtop_pressure_pa,
        full_output=False,
    )

    if zero_opacities:
        kwargs["zero_opacities"] = list(zero_opacities)

    if abundance_overrides_vmr is None:
        # Equilibrium chemistry
        kwargs["logZ"] = cfg.logZ
        kwargs["CO_ratio"] = cfg.CO_ratio
    else:
        # Custom abundances: AbundanceGetter -> overrides -> pass as custom_abundances
        abund = _build_custom_abundances_from_getter(
            logZ=cfg.logZ,
            CO_ratio=cfg.CO_ratio,
            overrides_vmr=abundance_overrides_vmr,
        )
        kwargs["logZ"] = None
        kwargs["CO_ratio"] = None
        kwargs["custom_abundances"] = abund

    # Call PLATON (return signature varies by version)
    result = calc.compute_depths(**kwargs)

    if isinstance(result, (tuple, list)):
        if len(result) < 2:
            raise RuntimeError(f"PLATON compute_depths returned too few values: len={len(result)}")
        wl_native_m = np.asarray(result[0], dtype=float)
        depth_native = np.asarray(result[1], dtype=float)
    elif isinstance(result, dict):
        wl_native_m = np.asarray(result["wavelengths"], dtype=float)
        depth_native = np.asarray(result["depths"], dtype=float)
    else:
        raise RuntimeError(f"Unexpected compute_depths return type: {type(result)}")

    wl_native_um = wl_native_m * 1e6

    # Interpolate onto requested wavelengths
    sort_idx = np.argsort(wl_native_um)
    wl_native_um = wl_native_um[sort_idx]
    depth_native = depth_native[sort_idx]
    depth_req = np.interp(wl_req_um, wl_native_um, depth_native)

    if debug:
        med = float(np.nanmedian(depth_req))
        print("Depth model median:", med)
        print("Implied rp from median depth:", float(np.sqrt(med)) if med > 0 else float("nan"))

    # Optional outputs
    if out_txt is not None:
        out_txt = Path(out_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_txt,
            np.c_[wl_req_um, depth_req],
            header="wavelength_um transit_depth",
            comments="",
        )

    if out_npz is not None:
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_npz,
            wavelength_um=wl_req_um,
            depth=depth_req,
        )

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.plot(wl_req_um, depth_req, lw=1.5)
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Transit depth $(R_p/R_*)^2$")
        plt.title("PLATON Forward Model — Transit Depth")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    return wl_req_um, depth_req

def platon_list_opacity_names() -> None:
    calc = TransitDepthCalculator()
    names = None
    for attr in ["opacity_names", "absorber_names", "gas_names", "species_names", "opacities"]:
        if hasattr(calc.atm, attr):
            names = getattr(calc.atm, attr)
            print(f"calc.atm.{attr} =", names)
            break
    if names is None:
        # Fallback: introspect attributes
        print("Could not find a standard names attribute on calc.atm. Available attrs:")
        print([a for a in dir(calc.atm) if "name" in a or "opac" in a or "absorb" in a])

def platon_overlay_binned(
    binned_txt: PathLike,
    out_png: PathLike,
    cfg: PlatonPlanetStar = PlatonPlanetStar(),
    abundance_overrides_vmr: Optional[Dict[str, float]] = None,
    zero_opacities: Optional[list[str]] = None,
    plot_raw_platon: bool = False,
    plot_matched_platon: bool = True,
    debug: bool = False,
) -> None:
    wl_um, depth_obs, err_obs = load_binned_spectrum(binned_txt)

    # Evaluate PLATON exactly at the binned wavelength centers
    wl_um_model, depth_model = compute_platon_transit_depths(
        cfg=cfg,
        wavelengths_um=wl_um,
        out_txt=None,
        out_npz=None,
        out_png=None,
        abundance_overrides_vmr=abundance_overrides_vmr,
        zero_opacities=zero_opacities,
        debug=debug,
    )

    obs_med = float(np.nanmedian(depth_obs))
    obs_rp = float(np.sqrt(obs_med)) if np.isfinite(obs_med) and obs_med > 0 else float("nan")

    mod_med = float(np.nanmedian(depth_model))
    mod_rp = float(np.sqrt(mod_med)) if np.isfinite(mod_med) and mod_med > 0 else float("nan")

    # Baseline-match (shape comparison)
    offset = float(np.nanmedian(depth_model - depth_obs))
    depth_model_matched = depth_model - offset

    matched_med = float(np.nanmedian(depth_model_matched))
    matched_rp = float(np.sqrt(matched_med)) if np.isfinite(matched_med) and matched_med > 0 else float("nan")

    print("[Overlay summary]")
    print(f"  JWST median depth      = {obs_med:.6g}  (implied rp={obs_rp:.6g})")
    print(f"  PLATON median depth    = {mod_med:.6g}  (implied rp={mod_rp:.6g})")
    print(f"  Baseline offset (model - obs) = {offset:.6g}")
    print(f"  PLATON matched median  = {matched_med:.6g}  (implied rp={matched_rp:.6g})")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    have_err = np.isfinite(err_obs).any()
    plt.errorbar(
        wl_um,
        depth_obs,
        yerr=err_obs if have_err else None,
        fmt="o",
        markersize=4,
        capsize=2,
        label="JWST binned",
    )

    if plot_raw_platon:
        plt.plot(wl_um_model, depth_model, lw=2, label="PLATON (raw @ bin centers)")

    if plot_matched_platon:
        plt.plot(wl_um_model, depth_model_matched, lw=2, label="PLATON (baseline-matched)")

    plt.xlabel("Wavelength [µm]")
    plt.ylabel("Transit depth $(R_p/R_*)^2$")
    plt.title("JWST binned spectrum vs PLATON forward model")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
