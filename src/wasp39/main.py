from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from .io import load_prepared_lightcurve, load_transmission_spectrum_txt, save_transmission_spectrum_txt
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
#from .spectrum import construct_transmission_spectrum
from .spectrum_multicore import construct_transmission_spectrum
from .platon_model import (
    wasp39b_defaults_si,
    make_atm_params,
    platon_transit_depths_at_wavelengths,
    PlatonAtmosphereParams
)



def parse_steps(s: str) -> list[int]:
    s = s.strip().lower()
    return [0, 1, 2, 3, 4, 5, 6] if s in ("all", "*") else [int(x) for x in s.split(",") if x.strip()]


def Step0(ctx: dict) -> None:
    print("\nSTEP 0 — Load prepared JWST light curve (.h5)")
    base = os.path.dirname(os.path.abspath(__file__))
    ctx["base"] = base
    ctx["out"] = os.path.join(base, "output")  # assume exists
    h5 = os.path.join(base, "WASP-39b_JWST_PRISM_2022-07-10_prepared_light_curve.h5")

    d = load_prepared_lightcurve(h5)
    ctx.update(
        bjd=d["bjd"],
        wavelength=d["wavelength"],
        flux=d["flux"],
        flux_err=d["flux_err"],
        oot_idx=d.get("oot_index_non_binned", None),
    )


def Step1(ctx: dict) -> None:
    print("STEP 1 — Construct + normalize white-light curve")
    white, white_e = normalize_white_light(
        ctx["flux"], ctx["flux_err"], oot_index=ctx["oot_idx"], wl_bin_start=83, wl_bin_end=339
    )
    ctx.update(white=white, white_e=white_e)
    save_white_light_plot(os.path.join(ctx["out"], "01_white_lightcurve.png"), ctx["bjd"], white, white_e, "White (norm)")


def Step2(ctx: dict, binning_factor: int) -> None:
    print(f"STEP 2 — Time bin the white-light curve (factor={binning_factor})")
    bjd_b, white_b, white_be = bin_time_series(ctx["bjd"], ctx["white"], ctx["white_e"], binning_factor)
    ctx.update(bjd_b=bjd_b, white_b=white_b, white_be=white_be)
    save_white_light_plot(
        os.path.join(ctx["out"], "02_white_lightcurve_binned.png"),
        bjd_b, white_b, white_be, f"White (binned x{binning_factor})"
    )


def Step3(ctx: dict) -> None:
    print("STEP 3 — Plot binned data + initial BATMAN model")
    t0_init = float(ctx["bjd_b"][int(np.nanargmin(ctx["white_b"]))])
    cfg = TransitConfig(t0=t0_init, per=4.055294, a=11.39, inc=87.32, u=(0.3, 0.1), limb_dark="quadratic")
    rp_init = 0.1457

    ctx.update(cfg_init=cfg, rp_init=rp_init)
    m = batman_model(ctx["bjd_b"], cfg, rp_init)
    save_data_plus_model(os.path.join(ctx["out"], "03_white_binned_plus_model.png"),
                         ctx["bjd_b"], ctx["white_b"], ctx["white_be"], m, "White + BATMAN init")

def run_white_light(ctx: dict, t, flux, flux_err, tag: str) -> None:
    chain, labels, best_params, best_model = fit_white_light_mcmc(
        t=t, flux=flux, flux_err=flux_err,
        cfg_init=ctx["cfg_init"], rp_init=ctx["rp_init"],
        nwalkers=50, nsteps_burn=2000, nsteps_prod=2000, thin=15, progress=True,
    )
    save_corner(os.path.join(ctx["out"], f"04_corner_white_light_{tag}.png"), chain, labels)
    save_bestfit_and_residuals(
        os.path.join(ctx["out"], f"05_bestfit_model_and_residuals_{tag}.png"),
        t, flux, flux_err, best_model, f"Best-fit + residuals ({tag})"
    )
    return best_params


def Step4(ctx: dict) -> None:
    print("STEP 4 — Run white-light MCMC (unbinned + binned, Gaussian LD priors)")

    u1_mu, u2_mu = 0.25, 0.30   # replace later with theory values
    u_sigma = 0.05              # conservative, but effective

    def run_white_light(t, flux, flux_err, tag):
        chain, labels, best_params, best_model = fit_white_light_mcmc(
            t=t,
            flux=flux,
            flux_err=flux_err,
            cfg_init=ctx["cfg_init"],
            rp_init=ctx["rp_init"],
            nwalkers=50,
            nsteps_burn=2000,
            nsteps_prod=2000,
            thin=15,
            progress=True,
            u_gauss_mu=(u1_mu, u2_mu),
            u_gauss_sigma=u_sigma,
        )

        save_corner(
            os.path.join(ctx["out"], f"04_corner_white_light_{tag}.png"),
            chain, labels
        )

        save_bestfit_and_residuals(
            os.path.join(ctx["out"], f"05_bestfit_model_and_residuals_{tag}.png"),
            t, flux, flux_err, best_model,
            f"Best-fit + residuals ({tag})"
        )


        # Save best-fit so Step5/6 can run without rerunning MCMC
        out_npz = os.path.join(ctx["out"], f"white_light_bestfit_{tag}.npz")
        np.savez(
            out_npz,
            labels=np.array(labels, dtype=object),
            best_params=np.array(best_params, dtype=float),
            u_gauss_mu=np.array([u1_mu, u2_mu], dtype=float),
            u_gauss_sigma=np.array([u_sigma], dtype=float),
        )
        print(f"Saved white-light best-fit ({tag}) to: {out_npz}")


        return best_params

    # --- unbinned ---
    best_unbinned = run_white_light(
        ctx["bjd"], ctx["white"], ctx["white_e"], "unbinned"
    )

    # --- binned ---
    best_binned = run_white_light(
        ctx["bjd_b"], ctx["white_b"], ctx["white_be"], "binned"
    )

    # Choose which solution to propagate (recommend: unbinned)
    best = best_unbinned

    t0, per, a, inc, rp, u1, u2, *_ = best
    ctx["cfg"] = TransitConfig(
        t0=float(t0),
        per=float(per),
        a=float(a),
        inc=float(inc),
        u=(float(u1), float(u2)),
        limb_dark="quadratic",
    )
    ctx["rp_fit"] = float(rp)


def Step5(ctx: dict, *, n_wl_bins: int, spec_time_bin: int, white_tag: str = "unbinned") -> None:
    print(f"STEP 5 — Build transmission spectrum (tag={white_tag})")

    # Load Step4 best-fit if missing OR if we're switching to a different tag
    if ("cfg" not in ctx) or ("rp_fit" not in ctx) or (ctx.get("_white_tag") != white_tag):
        npz_path = os.path.join(ctx["out"], f"white_light_bestfit_{white_tag}.npz")
        if not os.path.exists(npz_path):
            raise RuntimeError(
                f"Step5 needs white-light best-fit for '{white_tag}', but it wasn't found: {npz_path}. "
                "Run Step4 once to generate it."
            )

        dat = np.load(npz_path, allow_pickle=True)
        best_params = dat["best_params"]
        t0_m, per_m, a_m, inc_m, rp_m, u1_m, u2_m, *_ = best_params

        ctx["cfg"] = TransitConfig(
            t0=float(t0_m),
            per=float(per_m),
            a=float(a_m),
            inc=float(inc_m),
            u=(float(u1_m), float(u2_m)),
            limb_dark="quadratic",
        )
        ctx["rp_fit"] = float(rp_m)
        ctx["_white_tag"] = white_tag
        print(f"Loaded white-light best-fit ({white_tag}) from: {npz_path}")

    # ✅ ALWAYS build + save the spectrum
    out_txt = os.path.join(ctx["out"], f"transmission_spectrum_{white_tag}.txt")
    out_png = os.path.join(ctx["out"], f"transmission_spectrum_{white_tag}.png")

    wl_c, depth, elo, ehi = construct_transmission_spectrum(
        bjd=ctx["bjd"],
        wavelength_um=ctx["wavelength"],
        flux_2d=ctx["flux"],
        fluxerr_2d=ctx["flux_err"],
        cfg=ctx["cfg"],
        n_wavelength_bins=n_wl_bins,
        oot_index=ctx["oot_idx"],
        rp_init=ctx["rp_fit"],
        progress=True,
        verbose=True,
        time_bin_factor=spec_time_bin,
        n_jobs=15
    )

    save_transmission_spectrum_txt(out_txt, wl_c, depth, elo, ehi)
    save_transmission_spectrum_plot(out_png, wl_c, depth, elo, ehi, f"Transmission Spectrum ({white_tag})")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_png}")

    ctx.update(wl_c=wl_c, depth=depth, elo=elo, ehi=ehi)

def Step6(ctx: dict) -> None:
    print("STEP 6 — PLATON forward models (notebook-style overlay)")

    # (Optional but recommended) Make PLATON data download location stable:
    # Do this *before* importing PLATON in a fresh process.
    # os.environ.setdefault("PLATON_DATA_DIR", os.path.join(ctx["base"], "platon_data"))

    # Ensure we have a spectrum to overlay
    if "wl_c" not in ctx or "depth" not in ctx:
        spec_path = os.path.join(ctx["out"], "transmission_spectrum.txt")
        if os.path.exists(spec_path):
            wl_c, depth, elo, ehi = _load_transmission_spectrum_txt(spec_path)
            ctx.update(wl_c=wl_c, depth=depth, elo=elo, ehi=ehi)
            print(f"Loaded transmission spectrum from: {spec_path}")
        else:
            raise FileNotFoundError(
                "No in-memory spectrum and no output/transmission_spectrum.txt found. "
                "Run Step 5 first (or provide the txt file)."
            )

    wl_um = np.asarray(ctx["wl_c"], dtype=float)

    # Planet/star defaults (SI units)
    planet = wasp39b_defaults_si()

    # Notebook-style “try a few forward models”
    models = [
        ("Solar (logZ=0, C/O=0.53)", PlatonAtmosphereParams(logZ=0.0, CO_ratio=0.53)),
        ("Metal-rich (logZ=1, C/O=0.53)", PlatonAtmosphereParams(logZ=1.0, CO_ratio=0.53)),
        ("Cloud deck (10 mbar)", PlatonAtmosphereParams(logZ=0.0, CO_ratio=0.53, cloudtop_pressure=1e3)),
        ("High C/O (logZ=0, C/O=1.0)", PlatonAtmosphereParams(logZ=0.0, CO_ratio=1.0)),
    ]

    plt.figure(figsize=(11, 6))
    plt.errorbar(
        wl_um, ctx["depth"],
        yerr=[ctx["elo"], ctx["ehi"]],
        fmt="o", capsize=3, label="Observed"
    )

    for label, atm in models:
        dmod = platon_transit_depths_at_wavelengths(wl_um, planet, atm)
        plt.plot(wl_um, dmod, label=f"PLATON: {label}")

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Transit depth (Rp/R*)^2")
    plt.title("Transmission Spectrum with PLATON forward models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(ctx["out"], "06_transmission_spectrum_with_platon.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved: {out_png}")


def _load_transmission_spectrum_txt(path: str):
    # File format: header line, then 4 columns:
    # wl_center_um depth depth_err_lo depth_err_hi
    arr = np.loadtxt(path, comments="#")
    wl_c = arr[:, 0]
    depth = arr[:, 1]
    elo = arr[:, 2]
    ehi = arr[:, 3]
    return wl_c, depth, elo, ehi



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all", help="e.g. all or 0,1,2,3,4,5,6")
    ap.add_argument("--skip-spectrum-mcmc", action="store_true")
    ap.add_argument("--binning-factor", type=int, default=10)
    ap.add_argument("--n-wavelength-bins", type=int, default=30)
    ap.add_argument("--spec-time-bin-factor", type=int, default=10)
    args = ap.parse_args()



    ctx: dict = {}
    Step = {
        0: lambda: Step0(ctx),
        1: lambda: Step1(ctx),
        2: lambda: Step2(ctx, args.binning_factor),
        3: lambda: Step3(ctx),
        4: lambda: Step4(ctx),
        5: lambda: Step5(ctx, n_wl_bins=args.n_wavelength_bins, spec_time_bin=args.spec_time_bin_factor),
        6: lambda: Step6(ctx),
    }

    for s in sorted(parse_steps(args.steps)):
        Step[s]()


if __name__ == "__main__":
    main()
