from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup

import inspect

def make_atm_params(*, logZ: float, CO_ratio: float, cloudtop_pressure: float | None = None):
    """
    Create PlatonAtmosphereParams robustly even if the dataclass uses different field names.
    """
    sig = inspect.signature(PlatonAtmosphereParams)
    param_names = set(sig.parameters.keys())

    # Map "canonical" names -> possible field names in your dataclass
    aliases = {
        "logZ": ["logZ", "log_z", "log_metallicity", "metallicity_logZ", "logZ_over_solar"],
        "CO_ratio": ["CO_ratio", "co_ratio", "c_o", "C_O", "COratio"],
        "cloudtop_pressure": ["cloudtop_pressure", "cloudtop_P", "P_cloudtop", "cloud_top_pressure"],
    }

    kwargs = {}

    # logZ
    for name in aliases["logZ"]:
        if name in param_names:
            kwargs[name] = float(logZ)
            break

    # CO_ratio
    for name in aliases["CO_ratio"]:
        if name in param_names:
            kwargs[name] = float(CO_ratio)
            break

    # cloudtop_pressure (optional)
    if cloudtop_pressure is not None:
        for name in aliases["cloudtop_pressure"]:
            if name in param_names:
                kwargs[name] = float(cloudtop_pressure)
                break

    return PlatonAtmosphereParams(**kwargs)


@dataclass(frozen=True)
class PlatonPlanetParams:
    """
    All values must be SI (meters, kg, Kelvin).
    """
    Rs: float          # stellar radius [m]
    Mp: float          # planet mass [kg]
    Rp: float          # planet radius [m]
    Teq: float         # equilibrium temperature [K]


@dataclass(frozen=True)
class PlatonAtmosphereParams:
    """
    PLATON "knobs" for simple forward models.
    logZ: float = 0.0          # metallicity relative to solar (log10)
    CO_ratio: float = 0.53     # C/O ratio
    cloudtop_P: Optional[float] = None  # [Pa] if you want a gray cloud deck (optional)
    """

def wasp39b_defaults_si() -> PlatonPlanetParams:
    # Commonly cited: Mp ~ 0.28 Mj, Rp ~ 1.27 Rj, Teq ~ 1116 K. :contentReference[oaicite:2]{index=2}
    # Star radius is model-dependent; this is a starter. (Replace with archive value when you wire it in.)
    Rs = 0.90 * R_sun
    Mp = 0.28 * M_jup
    Rp = 1.27 * R_jup
    Teq = 1116.0
    return PlatonPlanetParams(Rs=Rs, Mp=Mp, Rp=Rp, Teq=Teq)


def platon_transit_depths_at_wavelengths(
    wl_um: np.ndarray,
    planet: PlatonPlanetParams,
    atm: PlatonAtmosphereParams,
    calculator: Optional[TransitDepthCalculator] = None,
) -> np.ndarray:
    """
    Compute PLATON transit depths and interpolate onto wl_um (micron).

    Returns:
        depth_model (Rp/Rs)^2 at wl_um
    """
    if calculator is None:
        calculator = TransitDepthCalculator()

    # PLATON returns its own wavelength grid + depths
    # compute_depths is the documented forward-model entry point. :contentReference[oaicite:3]{index=3}
    wl_model_m, depth_model = calculator.compute_depths(
        planet.Rs, planet.Mp, planet.Rp, planet.Teq,
        logZ=atm.logZ,
        CO_ratio=atm.CO_ratio,
        cloudtop_pressure=atm.cloudtop_P,
    )

    wl_model_um = (np.asarray(wl_model_m) * 1e6).astype(float)
    depth_model = np.asarray(depth_model).astype(float)

    # Interpolate onto requested bins (only where PLATON is defined)
    # If your wl_um is already limited to 0.5–5.0, this behaves well.
    return np.interp(wl_um, wl_model_um, depth_model, left=np.nan, right=np.nan)
