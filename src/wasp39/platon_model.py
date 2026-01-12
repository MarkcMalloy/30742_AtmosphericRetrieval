# platon_model.py

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup

@dataclass(frozen=True)
class PlatonAtmosphereParams:
    logZ: float = 0.0
    CO_ratio: float = 0.53

    # Optional: if provided, overrides PLATON’s internal chemistry
    gases: Optional[Tuple[str, ...]] = None
    vmrs: Optional[Tuple[float, ...]] = None

    cloudtop_pressure: float = np.inf  # Pa, np.inf = no cloud deck

    add_gas_absorption: bool = True
    add_scattering: bool = True
    add_collisional_absorption: bool = True
    add_H_minus_absorption: bool = False


def make_atm_params(atm: PlatonAtmosphereParams) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        logZ=float(atm.logZ),
        CO_ratio=float(atm.CO_ratio),
        add_gas_absorption=bool(atm.add_gas_absorption),
        add_scattering=bool(atm.add_scattering),
        add_collisional_absorption=bool(atm.add_collisional_absorption),
        add_H_minus_absorption=bool(atm.add_H_minus_absorption),
        cloudtop_pressure=float(atm.cloudtop_pressure),
    )

    # Only pass gases/vmrs if explicitly set
    if atm.gases is not None and atm.vmrs is not None:
        kwargs["gases"] = list(atm.gases)
        kwargs["vmrs"] = list(atm.vmrs)

    return kwargs



def _centers_to_edges_um(wl_um: np.ndarray) -> np.ndarray:
    """
    Convert wavelength bin centers (micron) to bin edges (micron).
    Needed because PLATON bins are configured via edges.
    """
    wl_um = np.asarray(wl_um, dtype=float)
    if wl_um.ndim != 1 or wl_um.size < 2:
        raise ValueError("wl_um must be a 1D array with at least 2 elements")

    mid = 0.5 * (wl_um[1:] + wl_um[:-1])
    edges = np.empty(wl_um.size + 1, dtype=float)
    edges[1:-1] = mid
    # extrapolate end edges
    edges[0] = wl_um[0] - (mid[0] - wl_um[0])
    edges[-1] = wl_um[-1] + (wl_um[-1] - mid[-1])
    return edges


def platon_transit_depths_at_wavelengths(
    wl_um: np.ndarray,
    planet_params: Dict[str, float],
    atm: PlatonAtmosphereParams,
) -> np.ndarray:
    """
    Evaluate PLATON transit depth (Rp/R*)^2 on the same wavelength bins as wl_um (micron).
    Uses PLATON's compute_depths(star_radius, planet_mass, planet_radius, temperature, ...).

    planet_params should contain:
      - "Rs" (m)  : stellar radius
      - "mass" (kg): planet mass
      - "radius" (m): planet radius
      - "T" (K)   : isothermal temperature (Teq-ish)
    """
    wl_um = np.asarray(wl_um, dtype=float)
    finite = np.isfinite(wl_um)
    if not np.all(finite):
        wl_um = wl_um[finite]

    calc = TransitDepthCalculator()

    # Configure wavelength bins (PLATON uses meters)
    edges_um = _centers_to_edges_um(wl_um)
    edges_m = edges_um * 1e-6
    calc.change_wavelength_bins(edges_m)

    atm_kwargs = make_atm_params(atm)

    out = calc.compute_depths(
        star_radius=float(planet_params["Rs"]),
        planet_mass=float(planet_params["mass"]),
        planet_radius=float(planet_params["radius"]),
        temperature=float(planet_params["T"]),
        **atm_kwargs,
        full_output=False,
    )

    # PLATON typically returns (wavelengths_m, depths) OR (wavelengths_m, depths, info)
    if isinstance(out, tuple) and len(out) >= 2:
        depths = out[1]
    else:
        raise RuntimeError("Unexpected return type from PLATON compute_depths()")

    depths = np.asarray(depths, dtype=float)

    # depths should already align with bins; just return it
    return depths


def wasp39b_defaults_si() -> Dict[str, float]:
    """
    Hardcoded planet/star parameters in SI units.
    Adjust these numbers if you want WASP-38b instead.
    """
    return dict(
        # Star radius (m) — put your preferred value here
        Rs=0.90 * R_sun,

        # Planet mass/radius (SI)
        mass=0.28 * M_jup,
        radius=1.27 * R_jup,

        # Isothermal temperature (K)
        T=1116.0,
    )
