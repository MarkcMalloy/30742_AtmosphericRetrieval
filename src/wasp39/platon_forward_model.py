#!/usr/bin/env python3
"""
PLATON forward models overlay (patched):

Implements:
  1) Residual panel in plots (data - model)
  2) Bin-integrated forward models (evaluate PLATON on sub-samples within each bin, then average)
  3) Plotting in ppm (transit depth * 1e6)

Reads:  output/transmission_spectrum_unbinned.txt
Writes: output/platon_model_<family>_<name>.txt  (unitless transit depth)
Plots:  output/platon_overlay_<family>.png       (ppm + residuals panel)

Notes:
- This script expects a local module `platon_model.py` providing:
    PlatonPlanetStar, load_binned_spectrum, compute_platon_transit_depths, platon_list_opacity_names
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Abundance hygiene
# -----------------------------
def sanitize_abundance_overrides(
    overrides: Dict[str, float],
    *,
    he_vmr: float = 0.06,
    vmr_floor: float = 1e-12,
    force_h2: bool = True,
) -> Dict[str, float]:
    """
    Ensures:
      - all VMRs are finite and >= vmr_floor
      - total VMR <= 1 (leaves remainder to H2 by default)
      - mixture is renormalized to sum to 1
    """
    clean: Dict[str, float] = {}
    for k, v in overrides.items():
        if v is None:
            continue
        v = float(v)
        if not np.isfinite(v) or v <= 0:
            v = vmr_floor
        clean[k] = max(v, vmr_floor)

    # Enforce He
    clean["He"] = max(float(clean.get("He", he_vmr)), vmr_floor)

    total = float(sum(clean.values()))
    if force_h2:
        rem = 1.0 - total
        if rem < vmr_floor:
            # Scale everything except He down to make room for H2
            he = clean["He"]
            others = {k: v for k, v in clean.items() if k != "He"}
            others_sum = float(sum(others.values()))
            target_others_sum = max(1.0 - he - vmr_floor, vmr_floor)

            if others_sum > 0:
                scale = target_others_sum / others_sum
                for k in others:
                    clean[k] = max(others[k] * scale, vmr_floor)

            clean["He"] = he
            clean["H2"] = vmr_floor
        else:
            clean["H2"] = max(clean.get("H2", 0.0) + rem, vmr_floor)

    # Renormalize
    total = float(sum(clean.values()))
    for k in list(clean.keys()):
        clean[k] = clean[k] / total

    return clean


# -----------------------------
# Model specs
# -----------------------------
@dataclass(frozen=True)
class ModelCase:
    name: str
    cfg_overrides: Dict[str, float]
    abundance_overrides_vmr: Optional[Dict[str, float]] = None
    zero_opacities: Optional[List[str]] = None
    baseline_match: bool = True


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def _parse_zero_opacities(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts: List[str] = []
    for tok in s.replace(" ", ",").split(","):
        tok = tok.strip().strip("'").strip('"')
        if tok:
            parts.append(tok)
    return parts or None


# -----------------------------
# Bin handling + integrated model
# -----------------------------
def _bin_edges_from_centers(wl_um: np.ndarray) -> np.ndarray:
    wl_um = np.asarray(wl_um, float)
    if wl_um.size < 2:
        raise ValueError("Need at least 2 wavelength centers to build bin edges.")
    mids = 0.5 * (wl_um[1:] + wl_um[:-1])
    edges = np.concatenate(([wl_um[0] - (mids[0] - wl_um[0])], mids, [wl_um[-1] + (wl_um[-1] - mids[-1])]))
    # Fallback if something went weird
    if not np.all(np.diff(edges) > 0):
        d = np.diff(wl_um)
        edges = np.concatenate(([wl_um[0] - d[0] / 2], 0.5 * (wl_um[1:] + wl_um[:-1]), [wl_um[-1] + d[-1] / 2]))
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Non-monotonic bin edges; check wavelength centers.")
    return edges


def _integrated_model_on_bins(
    *,
    cfg,
    wl_centers_um: np.ndarray,
    compute_platon_transit_depths,
    abundance_overrides_vmr: Optional[Dict[str, float]],
    zero_opacities: Optional[List[str]],
    n_sub: int = 30,
) -> np.ndarray:
    """
    Evaluate PLATON on sub-samples within each bin and average to match binned data.
    Returns depth per bin (same length as wl_centers_um).
    """
    wl_centers_um = np.asarray(wl_centers_um, float)
    edges = _bin_edges_from_centers(wl_centers_um)

    # Midpoint samples within each bin (avoid exact edges)
    all_wl = []
    for i in range(len(wl_centers_um)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        step = (hi - lo) / n_sub
        pts = lo + (np.arange(n_sub) + 0.5) * step
        all_wl.append(pts)
    fine_wl = np.concatenate(all_wl, axis=0)

    _, fine_depth = compute_platon_transit_depths(
        cfg=cfg,
        wavelengths_um=fine_wl,
        abundance_overrides_vmr=abundance_overrides_vmr,
        zero_opacities=zero_opacities,
        out_txt=None,
        out_npz=None,
        out_png=None,
        debug=False,
    )

    if not np.isfinite(fine_depth).all():
        bad = np.count_nonzero(~np.isfinite(fine_depth))
        raise RuntimeError(f"Non-finite model depths produced: {bad} values. Likely PLATON data/abundance grid issue.")

    fine_depth = np.asarray(fine_depth, float).reshape(len(wl_centers_um), n_sub)
    return np.nanmean(fine_depth, axis=1)


# -----------------------------
# Forward model computation
# -----------------------------
def compute_case(
    *,
    case: ModelCase,
    base_cfg_kwargs: Dict[str, float],
    wl_um: np.ndarray,
    depth_obs: np.ndarray,
    compute_platon_transit_depths,
    PlatonPlanetStar,
    integrate_bins: bool,
    n_sub: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns: (wl_um, depth_model_for_plot, applied_offset)
    depth values are unitless transit depth (Rp/Rs)^2
    """
    cfg_kwargs = dict(base_cfg_kwargs)
    cfg_kwargs.update(case.cfg_overrides)
    cfg = PlatonPlanetStar(**cfg_kwargs)

    abund = case.abundance_overrides_vmr
    if abund is not None:
        abund = sanitize_abundance_overrides(abund, he_vmr=0.06, vmr_floor=1e-12, force_h2=True)

    if integrate_bins:
        depth_model = _integrated_model_on_bins(
            cfg=cfg,
            wl_centers_um=wl_um,
            compute_platon_transit_depths=compute_platon_transit_depths,
            abundance_overrides_vmr=abund,
            zero_opacities=case.zero_opacities,
            n_sub=n_sub,
        )

    else:
        depth_model = _integrated_model_on_bins(
            cfg=cfg,
            wl_centers_um=wl_um,
            compute_platon_transit_depths=compute_platon_transit_depths,
            abundance_overrides_vmr=abund,
            zero_opacities=case.zero_opacities,
            n_sub=n_sub,
        )

    offset = 0.0
    depth_plot = np.asarray(depth_model, float).copy()
    if case.baseline_match:
        offset = float(np.nanmedian(depth_plot - depth_obs))
        depth_plot = depth_plot - offset

    return wl_um, depth_plot, offset


# -----------------------------
# Plotting (ppm + residual panel)
# -----------------------------
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
    ppm: bool,
) -> None:
    wl_um = np.asarray(wl_um, float)
    depth_obs = np.asarray(depth_obs, float)
    err_obs = np.asarray(err_obs, float)

    scale = 1e6 if ppm else 1.0
    ylab = "Transit depth [ppm]" if ppm else "Transit depth $(R_p/R_*)^2$"
    rlab = "Data − Model [ppm]" if ppm else "Data − Model"

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    have_err = (not no_errorbars) and np.isfinite(err_obs).any()

    ax.errorbar(
        wl_um,
        depth_obs * scale,
        yerr=(err_obs * scale) if have_err else None,
        fmt="o",
        markersize=4,
        capsize=2,
        label=data_label,
    )

    # Plot each model + its residuals
    for case, depth_plot, offset in cases_results:
        ax.plot(wl_um, depth_plot * scale, lw=2, label=case.name)
        resid = (depth_obs - depth_plot) * scale
        axr.errorbar(
            wl_um,
            resid,
            yerr=(err_obs * scale) if have_err else None,
            fmt="o",
            markersize=3,
            capsize=2,
            alpha=0.9,
        )

    ax.set_ylabel(ylab)
    ax.set_title(f"PLATON forward models — {family_name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    axr.axhline(0.0, lw=1)
    axr.set_xlabel("Wavelength [µm]")
    axr.set_ylabel(rlab)
    axr.grid(alpha=0.3)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, default="output/transmission_spectrum_unbinned.txt")
    ap.add_argument("--outdir", type=str, default="output")

    # Geometry/stellar/planet defaults
    ap.add_argument("--rp-over-rs", type=float, default=0.15)
    ap.add_argument("--rstar-rsun", type=float, default=0.895)
    ap.add_argument("--mplanet-mjup", type=float, default=0.281)
    ap.add_argument("--T", type=float, default=1150.0)
    ap.add_argument("--cloudtop-pa", type=float, default=1e2)

    # Opacities
    ap.add_argument("--global-zero-opacities", type=str, default=None)

    # Plot options
    ap.add_argument("--no-errorbars", action="store_true")
    ap.add_argument("--no-combined-plot", action="store_true")
    ap.add_argument("--units", choices=["ppm", "depth"], default="ppm", help="Plot y-axis units.")
    ap.add_argument("--integrate-bins", action="store_true", help="Integrate model over bins (recommended).")
    ap.add_argument("--n-sub", type=int, default=30, help="Sub-samples per bin when integrating.")

    # Debug
    ap.add_argument("--list-opacities", action="store_true")

    # Data hygiene
    ap.add_argument("--high-err-sigma", type=float, default=3.0, help="Drop if err > this×median(err).")

    args = ap.parse_args()
    args.integrate_bins = True
    spec_path = Path(args.spec)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import helper module
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

    # Sort by wavelength
    srt = np.argsort(wl_um)
    wl_um, depth_obs, err_obs = wl_um[srt], depth_obs[srt], err_obs[srt]

    # Basic finite + positive errors
    finite = np.isfinite(wl_um) & np.isfinite(depth_obs) & np.isfinite(err_obs) & (err_obs > 0)
    wl_um, depth_obs, err_obs = wl_um[finite], depth_obs[finite], err_obs[finite]

    # Drop high-uncertainty points (always)
    med_err = float(np.nanmedian(err_obs))
    if not np.isfinite(med_err) or med_err <= 0:
        raise SystemExit(f"Median error is not valid: {med_err}")
    thresh = float(args.high_err_sigma) * med_err
    keep = err_obs <= thresh
    dropped = int(np.count_nonzero(~keep))
    if dropped > 0:
        print(f"[Data] Dropped {dropped} high-uncertainty point(s) (err > {args.high_err_sigma}×median = {thresh:.3g}).")
    wl_um, depth_obs, err_obs = wl_um[keep], depth_obs[keep], err_obs[keep]

    global_zero = _parse_zero_opacities(args.global_zero_opacities)

    base_cfg = dict(
        rp_over_rs=float(args.rp_over_rs),
        rstar_rsun=float(args.rstar_rsun),
        mplanet_mjup=float(args.mplanet_mjup),
        temperature_k=float(args.T),
        cloudtop_pressure_pa=float(args.cloudtop_pa),
        logZ=0.2,
        CO_ratio=0.55,
    )

    # ------------------------------------------------------------
    # DEFINE YOUR MODEL FAMILIES HERE
    # ------------------------------------------------------------

    elemental_cases: List[ModelCase] = [
        ModelCase(
            name="Hot Jupiter: CO moderate, CH4 suppressed",
            cfg_overrides={},
            abundance_overrides_vmr={
                "He": 0.09,
                "H2O": 1e-4,
                "CO": 1e-6,
                "CO2": 1e-5,
                "CH4": 1e-5,
                "SO2": 1e-9,
            },
            baseline_match=True,
        ),
    ]

    metallicity_cases: List[ModelCase] = [
        ModelCase(
            name="Metal 1× solar",
            cfg_overrides={"logZ": 0.0, "CO_ratio": 0.55},
            abundance_overrides_vmr=None,
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

    # Apply global opacity toggle if provided (case-specific still wins)
    if global_zero:
        elemental_cases = [
            ModelCase(**{**case.__dict__, "zero_opacities": case.zero_opacities or global_zero})
            for case in elemental_cases
        ]
        metallicity_cases = [
            ModelCase(**{**case.__dict__, "zero_opacities": case.zero_opacities or global_zero})
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
                integrate_bins=bool(args.integrate_bins),
                n_sub=int(args.n_sub),
            )

            results.append((case, depth_plot, offset))

            # Save model (unitless)
            out_txt = out_dir / f"platon_model_{family_key}_{_slug(case.name)}.txt"
            np.savetxt(
                out_txt,
                np.c_[wl_um, depth_plot],
                header="wavelength_um transit_depth_model_unitless",
                comments="",
            )
        return results

    elemental_results = run_family("elemental", elemental_cases)
    metallicity_results = run_family("metallicity", metallicity_cases)

    ppm = (args.units == "ppm")

    plot_family(
        family_name="Elemental composition-defined",
        data_label="Data",
        wl_um=wl_um,
        depth_obs=depth_obs,
        err_obs=err_obs,
        cases_results=elemental_results,
        out_png=out_dir / "platon_overlay_elemental.png",
        no_errorbars=args.no_errorbars,
        ppm=ppm,
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
        ppm=ppm,
    )

    if not args.no_combined_plot:
        combined: List[Tuple[ModelCase, np.ndarray, float]] = []
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
            ppm=ppm,
        )

    print("Done.")
    print(f"  Read spectrum: {spec_path}")
    print(f"  Saved plots in: {out_dir}")
    print(f"  Used bin-integrated model: {bool(args.integrate_bins)} (n_sub={int(args.n_sub)})")
    print(f"  Units: {args.units}")


if __name__ == "__main__":
    main()
