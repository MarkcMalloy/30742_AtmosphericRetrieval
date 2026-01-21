#!/usr/bin/env python3
"""
PLATON forward models overlay:
- Family A: composition-defined (explicit abundance overrides)
- Family B: metallicity-defined (logZ/CO_ratio based, optionally with some overrides)

Reads:  output/transmission_spectrum_unbinned.txt
Writes: output/platon_model_<family>_<name>.txt
Plots:  output/platon_overlay_elemental.png
        output/platon_overlay_metallicity.png
        output/platon_overlay_all.png (optional)

Usage:
  python platon_forwardmodel_families.py --spec output/transmission_spectrum_unbinned.txt --outdir output
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Model specs
# -----------------------------
@dataclass(frozen=True)
class ModelCase:
    name: str
    cfg_overrides: Dict[str, float]               # e.g. {"logZ": 0.2, "CO_ratio": 0.55}
    abundance_overrides_vmr: Optional[Dict[str, float]] = None
    zero_opacities: Optional[List[str]] = None
    baseline_match: bool = True                   # subtract median(model-data) for shape comparison


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def _parse_zero_opacities(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = []
    for tok in s.replace(" ", ",").split(","):
        tok = tok.strip().strip("'").strip('"')
        if tok:
            parts.append(tok)
    return parts or None


def compute_case(
    *,
    case: ModelCase,
    base_cfg_kwargs: Dict[str, float],
    wl_um: np.ndarray,
    depth_obs: np.ndarray,
    compute_platon_transit_depths,
    PlatonPlanetStar,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns: (wl_um, depth_model_for_plot, applied_offset)
    """
    # Merge cfg overrides onto base cfg kwargs
    cfg_kwargs = dict(base_cfg_kwargs)
    cfg_kwargs.update(case.cfg_overrides)

    cfg = PlatonPlanetStar(**cfg_kwargs)

    _, depth_model = compute_platon_transit_depths(
        cfg=cfg,
        wavelengths_um=wl_um,
        abundance_overrides_vmr=case.abundance_overrides_vmr,
        zero_opacities=case.zero_opacities,
        out_txt=None,
        out_npz=None,
        out_png=None,
        debug=False,
    )

    offset = 0.0
    depth_plot = depth_model.copy()
    if case.baseline_match:
        offset = float(np.nanmedian(depth_model - depth_obs))
        depth_plot = depth_model - offset

    return wl_um, depth_plot, offset


def plot_family(
    *,
    family_name: str,
    data_label: str,
    wl_um: np.ndarray,
    depth_obs: np.ndarray,
    err_obs: np.ndarray,
    cases_results: List[Tuple[ModelCase, np.ndarray, float]],
    out_png: Path,
    no_errorbars: bool,
) -> None:
    plt.figure(figsize=(10, 5))

    have_err = (not no_errorbars) and np.isfinite(err_obs).any()
    plt.errorbar(
        wl_um,
        depth_obs,
        yerr=err_obs if have_err else None,
        fmt="o",
        markersize=4,
        capsize=2,
        label=data_label,
    )

    for case, depth_plot, offset in cases_results:
        if case.baseline_match:
            lbl = f"{case.name} (offset {offset:.3g})"
        else:
            lbl = case.name
        plt.plot(wl_um, depth_plot, lw=2, label=lbl)

    plt.xlabel("Wavelength [µm]")
    plt.ylabel("Transit depth $(R_p/R_*)^2$")
    plt.title(f"PLATON forward models — {family_name}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, default="output/transmission_spectrum_unbinned.txt")
    ap.add_argument("--outdir", type=str, default="output")

    # Geometry/stellar/planet defaults (shared across all cases unless overridden per-case via cfg_overrides)
    ap.add_argument("--rp-over-rs", type=float, default=0.15)
    ap.add_argument("--rstar-rsun", type=float, default=0.895)
    ap.add_argument("--mplanet-mjup", type=float, default=0.281)
    ap.add_argument("--T", type=float, default=1150.0)
    ap.add_argument("--cloudtop-pa", type=float, default=1e2)

    # Optional: apply the same opacity toggles to *all* cases (each case can still override)
    ap.add_argument("--global-zero-opacities", type=str, default=None)

    # Plot options
    ap.add_argument("--no-errorbars", action="store_true")
    ap.add_argument("--no-combined-plot", action="store_true")

    # Debug
    ap.add_argument("--list-opacities", action="store_true")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import your helper module
    try:
        from platon_model import (
            PlatonPlanetStar,
            load_binned_spectrum,
            compute_platon_transit_depths,
            platon_list_opacity_names,
        )
    except Exception as e:
        raise SystemExit(
            "Could not import platon_model.py. Put this script in the same folder as platon_model.py "
            "or ensure that folder is on PYTHONPATH.\n"
            f"Import error: {e}"
        )

    if args.list_opacities:
        platon_list_opacity_names()

    # Load spectrum
    wl_um, depth_obs, err_obs = load_binned_spectrum(spec_path)
    wl_um = np.asarray(wl_um, float)
    depth_obs = np.asarray(depth_obs, float)
    err_obs = np.asarray(err_obs, float)

    global_zero = _parse_zero_opacities(args.global_zero_opacities)

    # Base cfg shared by all models
    base_cfg = dict(
        rp_over_rs=float(args.rp_over_rs),
        rstar_rsun=float(args.rstar_rsun),
        mplanet_mjup=float(args.mplanet_mjup),
        temperature_k=float(args.T),
        cloudtop_pressure_pa=float(args.cloudtop_pa),
        # NOTE: logZ and CO_ratio are intentionally not set here so each metallicity case can define them.
        # For elemental cases we can still set some defaults if you want.
        logZ=0.2,
        CO_ratio=0.55,
    )

    # ------------------------------------------------------------
    # DEFINE YOUR MODEL FAMILIES HERE
    # ------------------------------------------------------------

    # Family A: composition-defined (explicit abundances)
    # These are just examples; tune to match your paper’s intent.
    elemental_cases: List[ModelCase] = [
        ModelCase(
            name="100% H2O-like",
            cfg_overrides={},  # use base_cfg geometry + T + clouds
            abundance_overrides_vmr={
                # Keep some background; PLATON expects a physically sensible mix.
                # You can push toward a single-species signature by dominating the active absorber.
                "He": 0.06,
                "H2O": 1e-3,
                "CO2": 1e-12,
                "CO": 1e-12,
                "CH4": 1e-12,
            },
            zero_opacities=None,
            baseline_match=True,
        ),
        ModelCase(
            name="100% CO2-like",
            cfg_overrides={},
            abundance_overrides_vmr={
                "He": 0.06,
                "H2O": 1e-12,
                "CO2": 1e-3,
                "CO": 1e-12,
                "CH4": 1e-12,
            },
            baseline_match=True,
        ),
        ModelCase(
            name="100% CH4-like",
            cfg_overrides={},
            abundance_overrides_vmr={
                "He": 0.06,
                "H2O": 1e-12,
                "CO2": 1e-12,
                "CO": 1e-12,
                "CH4": 1e-3,
            },
            baseline_match=True,
        ),
        ModelCase(
            name="N2+CH4 mix (example)",
            cfg_overrides={},
            abundance_overrides_vmr={
                "He": 0.06,
                "CH4": 4e-4,
                # N2 handling depends on PLATON species availability;
                # If "N2" exists in your install, use it; otherwise remove this line.
                "N2": 9.6e-3,
                "H2O": 1e-12,
                "CO2": 1e-12,
                "CO": 1e-12,
            },
            baseline_match=True,
        ),
    ]

    # Family B: metallicity-defined (like 1x solar vs 100x solar)
    metallicity_cases: List[ModelCase] = [
        ModelCase(
            name="Metal 1× solar",
            cfg_overrides={"logZ": 0.0, "CO_ratio": 0.55},
            abundance_overrides_vmr=None,
            # Example: if you want to “turn off” some opacities like in your earlier runs:
            zero_opacities=["CO", "CO2"],
            baseline_match=True,
        ),
        ModelCase(
            name="Metal 100× solar",
            cfg_overrides={"logZ": 2.0, "CO_ratio": 0.55},
            abundance_overrides_vmr=None,
            zero_opacities=["CO", "CO2"],
            baseline_match=True,
        ),
    ]

    # Apply global opacities toggle if provided (case-specific still wins)
    if global_zero:
        elemental_cases = [
            ModelCase(
                **{**case.__dict__, "zero_opacities": case.zero_opacities or global_zero}
            )
            for case in elemental_cases
        ]
        metallicity_cases = [
            ModelCase(
                **{**case.__dict__, "zero_opacities": case.zero_opacities or global_zero}
            )
            for case in metallicity_cases
        ]

    # ------------------------------------------------------------
    # Compute + save + plot each family
    # ------------------------------------------------------------
    def run_family(family_key: str, cases: List[ModelCase]) -> List[Tuple[ModelCase, np.ndarray, float]]:
        results: List[Tuple[ModelCase, np.ndarray, float]] = []
        for case in cases:
            _, depth_plot, offset = compute_case(
                case=case,
                base_cfg_kwargs=base_cfg,
                wl_um=wl_um,
                depth_obs=depth_obs,
                compute_platon_transit_depths=compute_platon_transit_depths,
                PlatonPlanetStar=PlatonPlanetStar,
            )
            results.append((case, depth_plot, offset))

            # Save model
            out_txt = out_dir / f"platon_model_{family_key}_{_slug(case.name)}.txt"
            np.savetxt(
                out_txt,
                np.c_[wl_um, depth_plot],
                header="wavelength_um transit_depth_model_for_plot",
                comments="",
            )
        return results

    elemental_results = run_family("elemental", elemental_cases)
    metallicity_results = run_family("metallicity", metallicity_cases)

    # Family plots
    plot_family(
        family_name="Elemental composition-defined",
        data_label="Data",
        wl_um=wl_um,
        depth_obs=depth_obs,
        err_obs=err_obs,
        cases_results=elemental_results,
        out_png=out_dir / "platon_overlay_elemental.png",
        no_errorbars=args.no_errorbars,
    )
    plot_family(
        family_name="Metallicity-defined",
        data_label="Data",
        wl_um=wl_um,
        depth_obs=depth_obs,
        err_obs=err_obs,
        cases_results=metallicity_results,
        out_png=out_dir / "platon_overlay_metallicity.png",
        no_errorbars=args.no_errorbars,
    )

    # Combined plot (optional)
    if not args.no_combined_plot:
        combined = []
        # Prefix labels to make legend clearer
        for case, depth_plot, offset in elemental_results:
            combined.append((ModelCase(name=f"[Elem] {case.name}", cfg_overrides=case.cfg_overrides,
                                       abundance_overrides_vmr=case.abundance_overrides_vmr,
                                       zero_opacities=case.zero_opacities, baseline_match=case.baseline_match),
                             depth_plot, offset))
        for case, depth_plot, offset in metallicity_results:
            combined.append((ModelCase(name=f"[Z] {case.name}", cfg_overrides=case.cfg_overrides,
                                       abundance_overrides_vmr=case.abundance_overrides_vmr,
                                       zero_opacities=case.zero_opacities, baseline_match=case.baseline_match),
                             depth_plot, offset))

        plot_family(
            family_name="All models",
            data_label="Data",
            wl_um=wl_um,
            depth_obs=depth_obs,
            err_obs=err_obs,
            cases_results=combined,
            out_png=out_dir / "platon_overlay_all.png",
            no_errorbars=args.no_errorbars,
        )

    print("Done.")
    print(f"  Read spectrum: {spec_path}")
    print(f"  Saved plots in: {out_dir}")
    print(f"  Elemental cases: {len(elemental_cases)} | Metallicity cases: {len(metallicity_cases)}")
    print("  Tip: edit elemental_cases / metallicity_cases in main() to match the paper exactly.")


if __name__ == "__main__":
    main()
