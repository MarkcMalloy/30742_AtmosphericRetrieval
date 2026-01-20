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
from .platon_model import compute_platon_transit_depths, PlatonPlanetStar, platon_overlay_binned, platon_list_opacity_names

def parse_steps(s: str) -> list[int]:
    s = s.strip().lower()
    return [0, 1, 2, 3, 4, 5, 6, 7] if s in ("all", "*") else [int(x) for x in s.split(",") if x.strip()]


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
    mode = ctx.get("white_light_mode", "both")  # "unbinned", "binned", or "both"
    propagate = ctx.get("white_light_propagate", None)  # "unbinned" or "binned" (optional)

    print(f"STEP 4 — Run white-light MCMC (mode={mode})")

    u1_mu, u2_mu = 0.25, 0.30   # replace later with theory values
    u_sigma = 0.05              # conservative, but effective

    per_mu = float(ctx["cfg_init"].per)
    per_sigma = 0.001  # days (~2.9 minutes). adjust if needed

    a_mu = float(ctx["cfg_init"].a)
    a_sigma = 0.20  # tune if needed

    def derive_inc_deg(a: float, b: float) -> float:
        # b = a*cos(i)  -> i = arccos(b/a)
        cosi = b / a
        cosi = float(np.clip(cosi, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosi)))

    def print_best_params(tag: str, labels: list, best_params: np.ndarray, inc_deg: float) -> None:
        vals = dict(zip(labels, best_params))
        print(f"\n[white-light best-fit | {tag}]")
        print(f"  t0  = {vals['t0']:.8f}")
        print(f"  per = {vals['per']:.8f} d")
        print(f"  a   = {vals['a']:.6f} (a/R*)")
        print(f"  b   = {vals['b']:.6f}")
        print(f"  inc = {inc_deg:.6f} deg  (derived from b/a)")
        print(f"  rp  = {vals['rp']:.6f} (Rp/R*)")
        print(f"  u1  = {vals['u1']:.6f}")
        print(f"  u2  = {vals['u2']:.6f}")
        print(f"  c0  = {vals['c0']:.6f}")
        print(f"  c1  = {vals['c1']:.6e}")

    def run_white_light(t, flux, flux_err, tag):
        chain, labels, best_params, best_model = fit_white_light_mcmc(
            t=t,
            flux=flux,
            flux_err=flux_err,
            cfg_init=ctx["cfg_init"],
            rp_init=ctx["rp_init"],
            nwalkers=64,
            nsteps_burn=3000,
            nsteps_prod=8000,
            thin=15,
            progress=True,
            u_gauss_mu=(u1_mu, u2_mu),
            u_gauss_sigma=u_sigma,
            per_gauss_mu=per_mu,
            per_gauss_sigma=per_sigma,
            a_gauss_mu=a_mu,
            a_gauss_sigma=a_sigma,
        )

        # Expect: labels == ["t0","per","a","b","rp","u1","u2","c0","c1"]
        # Derive inclination from best-fit (a,b)
        a_best = float(best_params[2])
        b_best = float(best_params[3])
        inc_best = derive_inc_deg(a_best, b_best)

        # Print best-fit summary
        print_best_params(tag, labels, best_params, inc_best)

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

        # Save original best_params plus an "inc" appended version
        labels_with_inc = list(labels) + ["inc"]
        best_params_with_inc = np.concatenate([best_params.astype(float), np.array([inc_best], dtype=float)])

        np.savez(
            out_npz,
            labels=np.array(labels, dtype=object),
            best_params=np.array(best_params, dtype=float),

            labels_with_inc=np.array(labels_with_inc, dtype=object),
            best_params_with_inc=np.array(best_params_with_inc, dtype=float),

            inc_best=np.array([inc_best], dtype=float),

            u_gauss_mu=np.array([u1_mu, u2_mu], dtype=float),
            u_gauss_sigma=np.array([u_sigma], dtype=float),

            per_gauss_mu=np.array([per_mu], dtype=float),
            per_gauss_sigma=np.array([per_sigma], dtype=float),
            a_gauss_mu=np.array([a_mu], dtype=float),
            a_gauss_sigma=np.array([a_sigma], dtype=float),
        )
        print(f"Saved white-light best-fit ({tag}) to: {out_npz}\n")

        return best_params, inc_best

    results = {}

    if mode in ("unbinned", "both"):
        best_params, inc_best = run_white_light(
            ctx["bjd"], ctx["white"], ctx["white_e"], "unbinned"
        )
        results["unbinned"] = (best_params, inc_best)

    if mode in ("binned", "both"):
        best_params, inc_best = run_white_light(
            ctx["bjd_b"], ctx["white_b"], ctx["white_be"], "binned"
        )
        results["binned"] = (best_params, inc_best)

    if not results:
        raise ValueError(f"Invalid white_light_mode={mode!r}. Use 'unbinned', 'binned', or 'both'.")

    # Decide which solution to propagate
    if propagate is None:
        propagate = next(iter(results.keys())) if len(results) == 1 else "unbinned"

    if propagate not in results:
        raise ValueError(
            f"white_light_propagate={propagate!r} not available (ran: {list(results.keys())})."
        )

    best_params, inc_best = results[propagate]
    ctx["white_light_tag"] = propagate  # useful for later steps if you want

    # Unpack sampled params: t0, per, a, b, rp, u1, u2, c0, c1
    t0, per, a, b, rp, u1, u2, *_ = best_params

    ctx["cfg"] = TransitConfig(
        t0=float(t0),
        per=float(per),
        a=float(a),
        inc=float(inc_best),  # derived
        u=(float(u1), float(u2)),
        limb_dark="quadratic",
    )
    ctx["rp_fit"] = float(rp)

    print(f"Propagating white-light solution: {propagate}")
    print(f"Derived inclination used for cfg: inc={inc_best:.6f} deg (b={float(b):.6f}, a={float(a):.6f})")


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

def Step6(ctx: dict, *, white_tag: str = "binned") -> None:
    print(f"STEP 6 — PLATON forward models (overlay, tag={white_tag})")
    out_dir = ctx["out"]

    cfg = PlatonPlanetStar(
        rp_over_rs=0.15,
        rstar_rsun=0.895,
        mplanet_mjup=0.281,
        temperature_k=1150.0,
        logZ=0.2,
        CO_ratio=0.55,
        cloudtop_pressure_pa=1e2,  # higher clouds (more flattening)
    )

    abundance_overrides_vmr = {
        "He": 0.06,
        "H2O": 6e-5,  # keep moderate
        "CO2": 1e-7,  # down from 4e-7 to kill 4.3 µm
        "CO": 3e-4,
        "CH4": 1e-8,
    }
    platon_list_opacity_names()
    binned_txt = f"{out_dir}/transmission_spectrum_binned.txt"
    platon_overlay_binned(
        binned_txt=binned_txt,
        out_png=f"{out_dir}/06_platon_overlay_binned.png",
        cfg=cfg,
        abundance_overrides_vmr=abundance_overrides_vmr,
        zero_opacities=["CO,CO2"],
        plot_raw_platon=False
    )
    print("PLATON transit depth spectrum + plot saved.")



def Step7(ctx: dict, *, tag: str = "binned") -> None:
    print(f"STEP 7 — PLATON retrieval (emcee, tag={tag})")

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from platon.constants import R_sun, M_jup
    from platon.combined_retriever import CombinedRetriever

    out_dir = ctx["out"]
    spec_path = os.path.join(out_dir, f"transmission_spectrum_{tag}.txt")
    if not os.path.exists(spec_path):
        raise RuntimeError(f"Missing spectrum: {spec_path}")

    wl_um, depth, elo, ehi = _load_transmission_spectrum_txt(spec_path)

    wl_um = np.asarray(wl_um, float)
    depth = np.asarray(depth, float)
    elo = np.asarray(elo, float)
    ehi = np.asarray(ehi, float)

    err = 0.5 * (np.abs(elo) + np.abs(ehi))
    mask = np.isfinite(wl_um) & np.isfinite(depth) & np.isfinite(err) & (err > 0)
    wl_um, depth, err = wl_um[mask], depth[mask], err[mask]
    if len(depth) < 2:
        raise RuntimeError("Not enough valid wavelength bins for retrieval")

    # Build bin edges from centers
    mids = 0.5 * (wl_um[1:] + wl_um[:-1])
    edges = np.concatenate([
        [wl_um[0] - (mids[0] - wl_um[0])],
        mids,
        [wl_um[-1] + (wl_um[-1] - mids[-1])]
    ])
    bins_m = np.column_stack([edges[:-1], edges[1:]]) * 1e-6

    base = ctx.get("platon_cfg", None)

    rstar_rsun = 0.895
    rp_over_rs = 0.15
    mplanet_mjup = 0.281
    temperature_k = 1150.0
    cloudtop_pressure_pa = 1e5
    tstar_k = 5400.0

    if base is not None:
        rstar_rsun = float(getattr(base, "rstar_rsun", rstar_rsun))
        rp_over_rs = float(getattr(base, "rp_over_rs", rp_over_rs))
        mplanet_mjup = float(getattr(base, "mplanet_mjup", mplanet_mjup))
        temperature_k = float(getattr(base, "temperature_k", temperature_k))
        cloudtop_pressure_pa = float(getattr(base, "cloudtop_pressure_pa", cloudtop_pressure_pa))
        tstar_k = float(getattr(base, "tstar_k", tstar_k))

    Rs = rstar_rsun * float(R_sun)
    Rp = rp_over_rs * Rs
    Mp = mplanet_mjup * float(M_jup)
    log_cloudtop_P0 = float(np.log10(cloudtop_pressure_pa))

    retriever = CombinedRetriever()

    # CRITICAL: for free retrieval / VMR fitting, PLATON requires logZ=None and CO_ratio=None
    fit_info = retriever.get_default_fit_info(
        Rs=Rs,
        Mp=Mp,
        Rp=Rp,
        T=temperature_k,
        logZ=None,
        CO_ratio=None,
        log_cloudtop_P=log_cloudtop_P0,
        T_star=tstar_k,
        free_retrieval=True,
        fit_vmr=True,
    )

    # Core priors
    fit_info.add_uniform_fit_param("Rp", 0.97 * Rp, 1.03 * Rp)
    fit_info.add_uniform_fit_param("T", 0.92 * temperature_k, 1.07 * temperature_k)
    fit_info.add_uniform_fit_param("log_cloudtop_P", 2.2, 5.0)

    # ------------------------------------------------------------
    # Gas priors — match the JWST WASP-39b PRISM graphic:
    # H2O, CO2, CO, SO2
    #
    # IMPORTANT: This PLATON version will KeyError if we try to add a prior
    # for a parameter name that doesn't already exist in fit_info.all_params.
    # So we:
    #   1) ask PLATON to include gases via add_gases_vmr
    #   2) safely constrain whatever parameter names actually exist
    # ------------------------------------------------------------

    # 1) Ask PLATON to include these gases (some installs may ignore some species)
    fit_info.add_gases_vmr(["H2O", "CO"], 1e-6, 1e-2)
    fit_info.add_gases_vmr(["CO2"], 1e-8, 1e-3)
    fit_info.add_gases_vmr(["SO2"], 1e-10, 1e-3)

    # 2) Constrain only parameters that actually exist (prevents KeyError)
    all_params = getattr(fit_info, "all_params", {})
    existing_params = set(all_params.keys())
    fit_names = set(getattr(fit_info, "fit_param_names", []) or [])

    def _safe_uniform(name: str, lo: float, hi: float) -> None:
        if (name in existing_params) and (name not in fit_names):
            fit_info.add_uniform_fit_param(name, lo, hi)
            fit_names.add(name)

    # If PLATON exposes log_* params, constrain those; otherwise constrain linear names.
    # H2O / CO
    _safe_uniform("log_H2O", -6.0, -2.0)
    _safe_uniform("log_CO",  -6.0, -2.0)
    _safe_uniform("H2O", 1e-6, 1e-2)
    _safe_uniform("CO",  1e-6, 1e-2)

    # CO2
    _safe_uniform("log_CO2", -8.0, -3.0)
    _safe_uniform("CO2", 1e-8, 1e-3)

    # SO2
    _safe_uniform("log_SO2", -10.0, -3.0)
    _safe_uniform("SO2", 1e-10, 1e-3)

    try:
        keys_preview = list(existing_params)
        keys_preview.sort()
        print("FitInfo all_params keys (preview):", keys_preview[:60])
    except Exception:
        pass

    print("PLATON fit parameters:")
    print("  " + ", ".join(fit_info.fit_param_names))

    nwalkers = int(ctx.get("platon_nwalkers", 50))
    nsteps = int(ctx.get("platon_nsteps", 20))
    print(f"Running emcee (single-core): walkers={nwalkers}, steps={nsteps}, nbins={len(depth)}")

    # Transmission-only: pass None so EclipseDepthCalculator is never created
    result = retriever.run_emcee(
        bins_m, depth, err,
        None, None, None,
        fit_info,
        nwalkers=nwalkers,
        nsteps=nsteps,
        include_condensation=True,
    )

    bestfit_png = os.path.join(out_dir, f"07_platon_retrieval_bestfit_{tag}.png")
    if plt.gcf().axes:
        plt.tight_layout()
        plt.savefig(bestfit_png, dpi=200)
        print(f"Saved: {bestfit_png}")
    plt.close("all")

    chain = np.asarray(result.flatchain)
    lnprob = np.asarray(result.flatlnprobability)

    corner_png = os.path.join(out_dir, f"07_corner_platon_retrieval_{tag}.png")
    save_corner(corner_png, chain, fit_info.fit_param_names)
    print(f"Saved: {corner_png}")

    i_best = int(np.nanargmax(lnprob))
    p_best = chain[i_best]
    ctx[f"platon_bestfit_{tag}"] = dict(zip(fit_info.fit_param_names, map(float, p_best)))

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
    ap.add_argument("--steps", default="all", help="e.g. all or 0,1,2,3,4,5,6,7")
    ap.add_argument("--skip-spectrum-mcmc", action="store_true")
    ap.add_argument("--binning-factor", type=int, default=10)
    ap.add_argument("--n-wavelength-bins", type=int, default=30)
    ap.add_argument("--spec-time-bin-factor", type=int, default=10)
    args = ap.parse_args()



    ctx: dict = {}
    ctx["white_light_mode"] = "binned"
    Step = {
        0: lambda: Step0(ctx),
        1: lambda: Step1(ctx),
        2: lambda: Step2(ctx, args.binning_factor),
        3: lambda: Step3(ctx),
        4: lambda: Step4(ctx),
        5: lambda: (
            #Step5(ctx, n_wl_bins=args.n_wavelength_bins, spec_time_bin=args.spec_time_bin_factor, white_tag="unbinned"),
            Step5(ctx, n_wl_bins=args.n_wavelength_bins, spec_time_bin=args.spec_time_bin_factor, white_tag="binned"),
        ),
        6: lambda: Step6(ctx),
        7: lambda: Step7(ctx),
    }

    for s in sorted(parse_steps(args.steps)):
        Step[s]()


if __name__ == "__main__":
    main()
