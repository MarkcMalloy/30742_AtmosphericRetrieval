from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .io import (
    load_prepared_lightcurve,
    load_transmission_spectrum_txt,
    save_transmission_spectrum_txt,   # keep if you have it; remove if not in your repo
)
from .normalize import normalize_white_light
from .binning import bin_time_series
from .lightcurve import TransitConfig, batman_model
from .mcmc import fit_white_light_mcmc
from .plotting import (
    save_white_light_plot,
    save_data_plus_model,
    save_corner,
    save_bestfit_and_residuals,
    save_transmission_spectrum_plot,
)
from .spectrum import construct_transmission_spectrum
from .platon_model import (
    wasp39b_defaults_si,
    PlatonAtmosphereParams,
    platon_transit_depths_at_wavelengths,
    make_atm_params
)

def main():
    parser = argparse.ArgumentParser(description="WASP-39b transmission spectroscopy pipeline")
    parser.add_argument(
        "--skip-spectrum-mcmc",
        action="store_true",
        help="Skip per-bin transmission spectrum MCMC and load existing output file instead",
    )
    args = parser.parse_args()
    # ======================================================
    # STEP 0 — Paths + load the prepared light curve file
    # ======================================================
    print("\nSTEP 0 — Load prepared JWST light curve (.h5)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(base_dir, "WASP-39b_JWST_PRISM_2022-07-10_prepared_light_curve.h5")

    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    data = load_prepared_lightcurve(h5_path)
    bjd = data["bjd"]
    wavelength = data["wavelength"]
    flux = data["flux"]
    flux_err = data["flux_err"]
    oot_idx = data.get("oot_index_non_binned", None)

    # ======================================================
    # STEP 1 — White light curve construction + normalization
    # (matches Proj_WASP.py: sum bins 83:339, normalize by OOT)
    # ======================================================
    print("STEP 1 — Construct + normalize white-light curve")
    white_norm, white_err_norm = normalize_white_light(
        flux, flux_err,
        oot_index=oot_idx,
        wl_bin_start=83,
        wl_bin_end=339,
    )
    save_white_light_plot(
        os.path.join(out_dir, "01_white_lightcurve.png"),
        bjd, white_norm, white_err_norm,
        "WASP-39b White light curve (normalized)"
    )

    # ======================================================
    # STEP 2 — Time binning (same as your original: factor=10)
    # ======================================================
    print("STEP 2 — Time bin the white-light curve (factor=10)")
    binning_factor = 10
    bjd_b, white_b, white_e = bin_time_series(bjd, white_norm, white_err_norm, binning_factor)
    save_white_light_plot(
        os.path.join(out_dir, "02_white_lightcurve_binned.png"),
        bjd_b, white_b, white_e,
        f"WASP-39b White light curve (binned x{binning_factor})"
    )

    # ======================================================
    # STEP 3 — Initial BATMAN model (sanity check)
    # ======================================================
    print("STEP 3 — Plot binned data + initial BATMAN model")
    t0_init = float(bjd_b[int(np.nanargmin(white_b))])

    cfg = TransitConfig(
        t0=t0_init,
        per=4.055294,
        a=11.39,
        inc=87.32,
        u=(0.3, 0.1),
        limb_dark="quadratic",
    )
    rp_init = 0.1457

    model_init = batman_model(bjd_b, cfg, rp_init)
    save_data_plus_model(
        os.path.join(out_dir, "03_white_binned_plus_model.png"),
        bjd_b, white_b, white_e, model_init,
        "White-light binned data + initial BATMAN model"
    )

    # ======================================================
    # STEP 4 — White-light MCMC fit (transit + linear baseline)
    # ======================================================
    print("STEP 4 — Run white-light MCMC (binned)")

    chain, labels, best_params, best_model = fit_white_light_mcmc(
        t=bjd_b,
        flux=white_b,
        flux_err=white_e,
        cfg_init=cfg,
        rp_init=rp_init,
        nwalkers=50,
        nsteps_burn=2000,
        nsteps_prod=2000,
        thin=15,
        progress=True,   # <-- shows emcee progress bars
    )

    np.savetxt(
        os.path.join(out_dir, "mcmc_white_light_binned.txt"),
        chain,
        header=" ".join(labels)
    )
    save_corner(os.path.join(out_dir, "04_corner_white_light_binned.png"), chain, labels)
    save_bestfit_and_residuals(
        os.path.join(out_dir, "05_bestfit_model_and_residuals_binned.png"),
        bjd_b, white_b, white_e, best_model,
        "WASP-39b White Light Curve — Best-fit (binned) + residuals"
    )

    # Update cfg with best-fit medians for downstream per-bin fits
    t0_m, per_m, a_m, inc_m, rp_m, u1_m, u2_m, c0_m, c1_m = best_params
    cfg = TransitConfig(
        t0=float(t0_m),
        per=float(per_m),
        a=float(a_m),
        inc=float(inc_m),
        u=(float(u1_m), float(u2_m)),
        limb_dark="quadratic",
    )
    rp_init = float(rp_m)

    # ======================================================
    # STEP 5 — Transmission spectrum construction
    # ======================================================
    out_txt = os.path.join(out_dir, "transmission_spectrum.txt")
    out_png = os.path.join(out_dir, "transmission_spectrum.png")

    if args.skip_spectrum_mcmc and os.path.exists(out_txt):
        print("STEP 5 — Skipping transmission spectrum MCMC (loading existing result)")
        wl_c, depth, elo, ehi = load_transmission_spectrum_txt(out_txt)

    else:
        print("STEP 5 — Build transmission spectrum (wavelength bins + MCMC per bin)")
        wl_c, depth, elo, ehi = construct_transmission_spectrum(
            bjd=bjd,
            wavelength_um=wavelength,
            flux_2d=flux,
            fluxerr_2d=flux_err,
            cfg=cfg,
            n_wavelength_bins=30,
            oot_index=oot_idx,
            rp_init=rp_init,
            progress=True,
            verbose=True,
            time_bin_factor=10,
        )

        save_transmission_spectrum_txt(out_txt, wl_c, depth, elo, ehi)
        save_transmission_spectrum_plot(
            out_png,
            wl_c, depth, elo, ehi,
            "WASP-39b Transmission Spectrum",
        )

        print("Saved:", out_txt)
        print("Saved:", out_png)

    # ======================================================
    # STEP 6 — PLATON forward models (overlay on observed spectrum)
    # ======================================================
    print("STEP 6 — PLATON forward models (overlay on observed spectrum)")

    planet = wasp39b_defaults_si()

    models = [
        ("solar (logZ=0, C/O=0.53)", make_atm_params(logZ=0.0, CO_ratio=0.53)),
        ("metal-rich (logZ=1)", make_atm_params(logZ=1.0, CO_ratio=0.53)),
        ("high C/O (C/O=1.0)", make_atm_params(logZ=0.0, CO_ratio=1.0)),
    ]

    plt.figure(figsize=(11, 6))
    plt.errorbar(wl_c, depth, yerr=[elo, ehi], fmt="o", capsize=3, label="Observed (MCMC)")

    for label, atm in models:
        depth_model = platon_transit_depths_at_wavelengths(wl_c, planet, atm)
        plt.plot(wl_c, depth_model, label=f"PLATON: {label}")

    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Transit depth (Rp/R*)^2")
    plt.title("WASP-39b: Observed Transmission Spectrum vs PLATON Forward Models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(out_dir, "transmission_spectrum_with_platon.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
