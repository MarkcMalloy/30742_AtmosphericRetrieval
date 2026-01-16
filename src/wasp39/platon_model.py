"""Small wrapper around PLATON for forward transmission spectra.

Goals (mirrors the course PLATON notebook):
- Keep planet parameters in SI units (meters, kg, Kelvin).
- Accept wavelength bin centers in microns and compute (Rp/Rs)^2 in those bins.
- Provide:
    * simple "equilibrium chemistry" knobs (logZ, C/O)
    * optional explicit composition control via PLATON's `custom_abundances`
      (like the notebook's AbundanceGetter example).

Notes:
- PLATON may download its data the first time it is used. Your main.py should set
  PLATON_DATA_DIR before importing PLATON.
- All inputs to PLATON are SI units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


# -----------------------------
# Atmosphere parameter bundle
# -----------------------------
@dataclass(frozen=True)
class PlatonAtmosphereParams:
    # --- Chemistry (equilibrium mode) ---
    # If `custom_abundances` is provided, logZ/CO_ratio are ignored (set to None in PLATON call).
    logZ: float = 0.0
    CO_ratio: float = 0.53
    CH4_mult: float = 1.0

    # --- Opacity toggles ---
    add_gas_absorption: bool = True
    add_H_minus_absorption: bool = False
    add_scattering: bool = True
    add_collisional_absorption: bool = True

    # --- Cloud deck ---
    # PLATON uses Pa. Example: 10 mbar = 1000 Pa
    cloudtop_pressure: float = np.inf

    # --- Haze / Rayleigh-like scattering controls ---
    scattering_factor: float = 1.0
    scattering_slope: float = 4.0
    scattering_ref_wavelength: float = 1e-6  # meters

    # --- Explicit composition options ---
    # (A) Notebook-style: provide PLATON `custom_abundances` dict
    #     e.g. abundances = AbundanceGetter().get(logZ, CO_ratio)
    #     modify abundances[...] then pass custom_abundances=abundances
    custom_abundances: Optional[Dict[str, float]] = None

    # (B) Alternate: override PLATON chemistry with explicit gases/vmrs.
    # Provide BOTH or neither.
    gases: Optional[Sequence[str]] = None
    vmrs: Optional[Sequence[float]] = None


def build_custom_abundances(
    logZ: float,
    CO_ratio: float,
    *,
    set_all_to_zero: bool = False,
    overrides: Optional[Dict[str, float]] = None,
    renormalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Notebook-like helper using PLATON AbundanceGetter.

    Returns:
        Dict[str, np.ndarray] mapping species -> abundance profile array
        (this is what PLATON expects for custom_abundances).
    """
    from platon.abundance_getter import AbundanceGetter

    getter = AbundanceGetter()
    abundances = getter.get(float(logZ), float(CO_ratio))  # species -> arrays

    # Pick a reference "shape" to broadcast scalar overrides into
    any_key = next(iter(abundances.keys()))
    ref = np.asarray(abundances[any_key])
    if ref.ndim == 0:
        # Extremely defensive: force a 1D shape
        ref = ref.reshape(1)

    def _as_profile(value, like: np.ndarray) -> np.ndarray:
        """Convert scalar/array to a profile array matching `like`."""
        arr = np.asarray(value)
        if arr.ndim == 0:
            return np.ones_like(like, dtype=float) * float(arr)
        # If it's already an array, try to broadcast to like
        return np.asarray(arr, dtype=float)

    # Ensure everything is arrays, same dtype
    for k in list(abundances.keys()):
        a = np.asarray(abundances[k], dtype=float)
        if a.ndim == 0:
            a = np.ones_like(ref, dtype=float) * float(a)
        abundances[k] = a

    # Zero all species if requested
    if set_all_to_zero:
        for k in abundances:
            abundances[k] = np.zeros_like(ref, dtype=float)

    # Apply overrides: scalar -> constant profile
    if overrides:
        unknown = [k for k in overrides.keys() if k not in abundances]
        if unknown:
            print("WARNING: unknown species in overrides:", unknown)

        for k, v in overrides.items():
            if k in abundances:
                abundances[k] = _as_profile(v, ref)

    # Renormalize elementwise so sum(species) = 1 at each layer
    if renormalize:
        s = np.zeros_like(ref, dtype=float)
        for v in abundances.values():
            s = s + np.asarray(v, dtype=float)

        # Avoid division by zero
        mask = s > 0
        for k in abundances:
            a = np.asarray(abundances[k], dtype=float)
            a2 = np.zeros_like(a, dtype=float)
            a2[mask] = a[mask] / s[mask]
            abundances[k] = a2

    return abundances



def make_atm_params(atm: PlatonAtmosphereParams) -> Dict[str, Any]:
    """Convert our dataclass into kwargs for TransitDepthCalculator.compute_depths()."""
    kwargs: Dict[str, Any] = dict(
        CH4_mult=float(atm.CH4_mult),
        add_gas_absorption=bool(atm.add_gas_absorption),
        add_H_minus_absorption=bool(atm.add_H_minus_absorption),
        add_scattering=bool(atm.add_scattering),
        add_collisional_absorption=bool(atm.add_collisional_absorption),
        cloudtop_pressure=float(atm.cloudtop_pressure),
        scattering_factor=float(atm.scattering_factor),
        scattering_slope=float(atm.scattering_slope),
        scattering_ref_wavelength=float(atm.scattering_ref_wavelength),
    )

    # --- Composition handling priority ---
    # 1) custom_abundances (notebook style)
    if atm.custom_abundances is not None:
        kwargs["logZ"] = None
        kwargs["CO_ratio"] = None
        kwargs["custom_abundances"] = {str(k): np.asarray(v, dtype=float) for k, v in atm.custom_abundances.items()}
        return kwargs

    # 2) equilibrium chemistry knobs
    kwargs["logZ"] = float(atm.logZ)
    kwargs["CO_ratio"] = float(atm.CO_ratio)

    # 3) explicit gases/vmrs override (optional)
    if (atm.gases is None) and (atm.vmrs is None):
        return kwargs

    if (atm.gases is None) != (atm.vmrs is None):
        raise ValueError("PlatonAtmosphereParams: gases and vmrs must be provided together (or neither).")

    gases = list(atm.gases)  # type: ignore[arg-type]
    vmrs = list(atm.vmrs)    # type: ignore[arg-type]
    if len(gases) != len(vmrs):
        raise ValueError(f"PlatonAtmosphereParams: len(gases)={len(gases)} != len(vmrs)={len(vmrs)}")

    kwargs["gases"] = gases
    kwargs["vmrs"] = vmrs
    return kwargs


# -----------------------------
# Wavelength utilities
# -----------------------------
def _centers_to_edges_um(wl_um: np.ndarray) -> np.ndarray:
    """Convert bin centers (micron) to bin edges (micron) by midpoint rule."""
    wl_um = np.asarray(wl_um, dtype=float)
    if wl_um.ndim != 1 or wl_um.size < 2:
        raise ValueError("wl_um must be a 1D array with at least 2 elements")

    mid = 0.5 * (wl_um[1:] + wl_um[:-1])
    edges = np.empty(wl_um.size + 1, dtype=float)
    edges[1:-1] = mid
    edges[0] = wl_um[0] - (mid[0] - wl_um[0])
    edges[-1] = wl_um[-1] + (wl_um[-1] - mid[-1])
    return edges


# -----------------------------
# Planet presets (SI units)
# -----------------------------
def super_jupiter_defaults_si() -> Dict[str, float]:
    """A 'hot super-Jupiter' toy planet (SI) for bracketing models."""
    from platon.constants import R_sun, R_jup, M_jup
    import numpy as np

    T_star = 5400.0
    Rs = 0.895 * R_sun
    a_AU = 0.0486

    AU_m = 1.495978707e11
    a_m = a_AU * AU_m

    Teq = T_star * np.sqrt(Rs / (2.0 * a_m))  # A=0, full redistribution
    print("Teq =", Teq)
    return dict(
        Rs=Rs,
        T_star=T_star,
        a_AU=a_AU,

        mass=0.275 * M_jup,
        radius=1.27 * R_jup,

        Teq=Teq,

        # optional: cloud deck for forward models (Pa)
        cloudtop_pressure=1e4,   # or 1e4 for 0.1 bar clouds
    )




# -----------------------------
# Forward model evaluation
# -----------------------------
def platon_transit_depths_at_wavelengths(
    wl_um: np.ndarray,
    planet_params: Dict[str, float],
    atm: PlatonAtmosphereParams,
    *,
    wl_min_um: float = 0.5,
    wl_max_um: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """Evaluate PLATON transit depth (Rp/Rs)^2 in the given wavelength bins.

    Args:
        wl_um: wavelength BIN CENTERS in microns.
        planet_params: dict with keys Rs, mass, radius, T (SI units).
        atm: PlatonAtmosphereParams
        wl_min_um, wl_max_um: only evaluate within this range; outside = NaN.

    Returns:
        depth array with same shape as wl_um.
    """
    wl_um = np.asarray(wl_um, dtype=float)
    out_depth = np.full_like(wl_um, np.nan, dtype=float)

    ok = np.isfinite(wl_um) & (wl_um >= float(wl_min_um)) & (wl_um <= float(wl_max_um))
    if not np.any(ok):
        return out_depth

    wl_use = wl_um[ok]

    # PLATON expects wavelength bins as (start, end) in meters.
    edges_um = _centers_to_edges_um(wl_use)
    edges_m = edges_um * 1e-6
    bins_m = np.column_stack([edges_m[:-1], edges_m[1:]])

    # Lazy import so main.py can set PLATON_DATA_DIR first.
    from platon.transit_depth_calculator import TransitDepthCalculator

    calc = TransitDepthCalculator()
    calc.change_wavelength_bins(bins_m)


    atm_kwargs = make_atm_params(atm)
    _temperature = float(
        planet_params.get(
            "temperature",
            planet_params.get(
                "Teq",
                planet_params.get("T", 1200.0),  # last-resort legacy fallback
            ),
        )
    )
    print("[PLATON model] using temperature =", _temperature, "K",
          "| T_star =", planet_params.get("T_star", None),
          "| cloudtop_pressure =", planet_params.get("cloudtop_pressure", None))
    res = calc.compute_depths(
        star_radius=float(planet_params["Rs"]),
        planet_mass=float(planet_params["mass"]),
        planet_radius=float(planet_params["radius"]),
        temperature=_temperature,
        full_output=False,
        **atm_kwargs,
    )

    # PLATON versions differ: sometimes returns (wavelengths, depths), sometimes just depths.
    if isinstance(res, tuple) and len(res) >= 2:
        depths = np.asarray(res[1], dtype=float)
    else:
        depths = np.asarray(res, dtype=float)

    depths[~np.isfinite(depths)] = np.nan

    if depths.shape[0] != wl_use.shape[0]:
        # Safe fallback if PLATON returns its own wavelengths
        if isinstance(res, tuple) and len(res) >= 2:
            wl_m = np.asarray(res[0], dtype=float)
            wl_um_pl = wl_m * 1e6
            if wl_um_pl.ndim == 1 and wl_um_pl.size == depths.size:
                depths = np.interp(wl_use, wl_um_pl, depths, left=np.nan, right=np.nan)
            else:
                raise RuntimeError("PLATON output shape mismatch; cannot map depths to requested bins.")
        else:
            raise RuntimeError("PLATON output shape mismatch; cannot map depths to requested bins.")

    out_depth[ok] = depths

    if verbose:
        try:
            d0 = (float(planet_params["radius"]) / float(planet_params["Rs"])) ** 2
            _ = d0
        except Exception:
            pass

    return out_depth
