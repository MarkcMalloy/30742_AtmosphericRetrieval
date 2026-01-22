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
from .spectrum import construct_transmission_spectrum
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
    mode = ctx.get("white_light_mode", "both")
    propagate = ctx.get("white_light_propagate", None)

    print(f"STEP 4 — Run white-light MCMC (mode={mode})")
    if "cfg_init" not in ctx or "rp_init" not in ctx:
        # Fallback: reproduce Step3 init so Step4 can be run standalone
        if "bjd_b" in ctx and "white_b" in ctx and "white_be" in ctx:
            t0_init = float(ctx["bjd_b"][int(np.nanargmin(ctx["white_b"]))])
            cfg = TransitConfig(
                t0=t0_init, per=4.055294, a=11.39, inc=87.32, u=(0.3, 0.1), limb_dark="quadratic"
            )
            rp_init = 0.1457
            ctx.update(cfg_init=cfg, rp_init=rp_init)
        else:
            raise RuntimeError(
                "Step4 needs cfg_init/rp_init. Run Step3 first (and Step2 if using binned mode)."
            )


    u1_mu, u2_mu = 0.25, 0.30
    u_sigma = 0.05

    # Fixed (NOT sampled)
    per_fixed = 4.1055294

    a_mu = float(ctx["cfg_init"].a)
    a_sigma = 0.20

    def derive_inc_deg(a: float, b: float) -> float:
        cosi = float(np.clip(b / a, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosi)))

    def print_best_params(tag: str, labels: list, best_params: np.ndarray, inc_deg: float) -> None:
        vals = dict(zip(labels, best_params))
        print(f"\n[white-light best-fit | {tag}]")
        print(f"  t0  = {vals['t0']:.8f}")
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
            cfg_init=ctx["cfg_init"],  # still contains fixed per
            rp_init=ctx["rp_init"],
            nwalkers=64,
            nsteps_burn=3000,
            nsteps_prod=8000,
            thin=15,
            progress=True,
            u_gauss_mu=(u1_mu, u2_mu),
            u_gauss_sigma=u_sigma,
            a_gauss_mu=a_mu,
            a_gauss_sigma=a_sigma,
        )

        # labels == ["t0","a","b","rp","u1","u2","c0","c1"]
        a_best = float(best_params[1])
        b_best = float(best_params[2])
        inc_best = derive_inc_deg(a_best, b_best)

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

        out_npz = os.path.join(ctx["out"], f"white_light_bestfit_{tag}.npz")

        labels_with_inc = list(labels) + ["inc"]
        best_params_with_inc = np.concatenate(
            [best_params.astype(float), np.array([inc_best], dtype=float)]
        )

        np.savez(
            out_npz,
            labels=np.array(labels, dtype=object),
            best_params=np.array(best_params, dtype=float),

            labels_with_inc=np.array(labels_with_inc, dtype=object),
            best_params_with_inc=np.array(best_params_with_inc, dtype=float),

            inc_best=np.array([inc_best], dtype=float),

            u_gauss_mu=np.array([u1_mu, u2_mu], dtype=float),
            u_gauss_sigma=np.array([u_sigma], dtype=float),

            a_gauss_mu=np.array([a_mu], dtype=float),
            a_gauss_sigma=np.array([a_sigma], dtype=float),
        )

        print(f"Saved white-light best-fit ({tag}) to: {out_npz}\n")
        return best_params, inc_best

    results = {}

    if mode in ("unbinned", "both"):
        results["unbinned"] = run_white_light(
            ctx["bjd"], ctx["white"], ctx["white_e"], "unbinned"
        )

    if mode in ("binned", "both"):
        results["binned"] = run_white_light(
            ctx["bjd_b"], ctx["white_b"], ctx["white_be"], "binned"
        )

    if not results:
        raise ValueError(f"Invalid white_light_mode={mode!r}")

    if propagate is None:
        propagate = next(iter(results.keys())) if len(results) == 1 else "unbinned"

    if propagate not in results:
        raise ValueError(
            f"white_light_propagate={propagate!r} not available (ran: {list(results.keys())})."
        )

    best_params, inc_best = results[propagate]
    ctx["white_light_tag"] = propagate

    # Unpack: t0, a, b, rp, u1, u2, ...
    t0, a, b, rp, u1, u2, *_ = best_params

    ctx["cfg"] = TransitConfig(
        t0=float(t0),
        per=per_fixed,          # FIXED, not sampled
        a=float(a),
        inc=float(inc_best),
        u=(float(u1), float(u2)),
        limb_dark="quadratic",
    )

    ctx["rp_fit"] = float(rp)

    print(f"Propagating white-light solution: {propagate}")
    print(
        f"Using fixed period per={per_fixed:.8f} d | "
        f"Derived inc={inc_best:.6f} deg (b={float(b):.6f}, a={float(a):.6f})"
    )


def Step5(ctx: dict, *, n_wl_bins: int, spec_time_bin: int, white_tag: str = "unbinned") -> None:
    print(f"STEP 5 — Build transmission spectrum (common-mode corrected, tag={white_tag})")

    import os
    import numpy as np

    # ----------------------------
    # 0) Load white-light best fit
    # ----------------------------
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

    out_txt = os.path.join(ctx["out"], f"transmission_spectrum_{white_tag}.txt")
    out_png = os.path.join(ctx["out"], f"transmission_spectrum_{white_tag}.png")

    # Time binning factor
    time_bin_factor = int(spec_time_bin) if (spec_time_bin is not None) else int(ctx.get("binning_factor", 1))
    if time_bin_factor < 1:
        time_bin_factor = 1

    # ----------------------------
    # 1) Build common-mode series
    # ----------------------------
    bjd = np.asarray(ctx["bjd"])
    wl = np.asarray(ctx["wavelength"])
    flux_2d = np.asarray(ctx["flux"])
    fluxerr_2d = np.asarray(ctx["flux_err"])
    oot = np.asarray(ctx["oot_idx"], dtype=bool)

    if flux_2d.ndim != 2:
        raise RuntimeError(f"Expected flux_2d to be 2D (ntime, nwave); got shape {flux_2d.shape}")
    if flux_2d.shape != fluxerr_2d.shape:
        raise RuntimeError("flux_2d and fluxerr_2d must have the same shape")

    ntime, nwave = flux_2d.shape
    if wl.shape[0] != nwave:
        # If your data is transposed, fix it here
        if wl.shape[0] == ntime and flux_2d.shape[1] != ntime:
            raise RuntimeError("wavelength length matches ntime unexpectedly; please check array shapes.")
        raise RuntimeError(f"wavelength_um length ({wl.shape[0]}) must match flux_2d second dim ({nwave})")

    # Weighted white flux (reduces white noise; robust to bad channels)
    w = 1.0 / np.maximum(fluxerr_2d, 1e-12) ** 2
    wsum = np.sum(w, axis=1)
    white_flux = np.sum(w * flux_2d, axis=1) / np.maximum(wsum, 1e-30)

    # Normalize white flux using out-of-transit
    if np.any(oot):
        white_norm = np.nanmedian(white_flux[oot])
    else:
        white_norm = np.nanmedian(white_flux)
    white_flux_n = white_flux / white_norm

    # Try to compute a transit model for the white curve (preferred)
    white_model = None
    try:
        # Adjust this import/name if your batman wrapper lives elsewhere
        from wasp39.transit import batman_model  # noqa
        # Typical signature in your project is batman_model(t, cfg, rp)
        white_model = batman_model(bjd, ctx["cfg"], float(ctx["rp_fit"]))
    except Exception:
        white_model = None

    if white_model is None:
        # Fallback: smooth the white flux itself to get a common-mode trend
        # (still useful for removing drifts; not as clean as transit-model residuals)
        try:
            from scipy.signal import savgol_filter
            # window length must be odd and < ntime
            win = min(101, ntime - (1 - ntime % 2))
            if win < 11:
                win = max(5, ntime - (1 - ntime % 2))
            if win % 2 == 0:
                win += 1
            white_trend = savgol_filter(white_flux_n, window_length=win, polyorder=2, mode="interp")
        except Exception:
            # last-resort: median filter-ish smoothing
            k = min(21, ntime)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            x = np.pad(white_flux_n, (pad, pad), mode="edge")
            white_trend = np.array([np.median(x[i:i + k]) for i in range(ntime)], float)

        common_mode = np.clip(white_flux_n / np.maximum(white_trend, 1e-30), 0.2, 5.0)
        print("[Step5] Common-mode: using smoothed white flux fallback (no transit model available).")
    else:
        # Residual common-mode = data/model (captures achromatic systematics)
        white_model = np.asarray(white_model, float)
        # Ensure white_model is normalized similarly (OOT ~ 1)
        if np.any(oot):
            mnorm = np.nanmedian(white_model[oot])
        else:
            mnorm = np.nanmedian(white_model)
        white_model_n = white_model / np.maximum(mnorm, 1e-30)

        common_mode = np.clip(white_flux_n / np.maximum(white_model_n, 1e-30), 0.2, 5.0)
        print("[Step5] Common-mode: using (white flux)/(white transit model) residuals.")

    # Apply correction to all channels
    flux_corr = flux_2d / common_mode[:, None]
    fluxerr_corr = fluxerr_2d / common_mode[:, None]  # approx propagation

    # Optional: light clipping of extreme corrected points (helps hot pixels/cosmic rays)
    # (Keep it conservative; channel fits should handle robustly too.)
    if ctx.get("spec_sigma_clip", False):
        med = np.nanmedian(flux_corr, axis=0, keepdims=True)
        mad = np.nanmedian(np.abs(flux_corr - med), axis=0, keepdims=True)
        sig = 1.4826 * mad + 1e-12
        bad = np.abs(flux_corr - med) > (5.0 * sig)
        flux_corr = np.where(bad, np.nan, flux_corr)
        fluxerr_corr = np.where(bad, np.nan, fluxerr_corr)

    # ----------------------------
    # 2) Build transmission spectrum
    # ----------------------------
    wl_c, depth, elo, ehi = construct_transmission_spectrum(
        bjd=bjd,
        wavelength_um=wl,
        flux_2d=flux_corr,
        fluxerr_2d=fluxerr_corr,
        cfg=ctx["cfg"],
        n_wavelength_bins=int(n_wl_bins),
        oot_index=oot,
        rp_init=float(ctx["rp_fit"]),
        progress=True,
        verbose=True,
        time_bin_factor=time_bin_factor,
    )

    save_transmission_spectrum_txt(out_txt, wl_c, depth, elo, ehi)
    save_transmission_spectrum_plot(out_png, wl_c, depth, elo, ehi, f"Transmission Spectrum ({white_tag}, common-mode)")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_png}")

    ctx.update(wl_c=wl_c, depth=depth, elo=elo, ehi=ehi)

def Step6(ctx: dict, *, white_tag: str = "binned") -> None:
    print(f"STEP 6 — PLATON forward models (overlay, tag={white_tag})")
    out_dir = ctx["out"]

    cfg = PlatonPlanetStar(
        rp_over_rs=0.10,
        rstar_rsun=0.895,
        mplanet_mjup=0.281,
        temperature_k=1175.0,
        logZ=0.2,
        CO_ratio=0.3,
        cloudtop_pressure_pa=1e6,  # higher clouds (more flattening)
    )

    #abundance_overrides_vmr = {
     #   "He": 0.09,
      #  "H2O": 4e-3,
       # "CO": 9e-5,
        #"CO2": 2e-5,
        #"CH4": 8e-6,
        #"SO2": 1e-5,
        #"K": 8e-4,
    #}
    abundance_overrides_vmr = {
        "He": 0.09,
        "H2O": 0.1,
        "CO2": 0.1,
    }
    platon_list_opacity_names()
    binned_txt = f"{out_dir}/transmission_spectrum_binned.txt"
    platon_overlay_binned(
        binned_txt=binned_txt,
        out_png=f"{out_dir}/final/06_platon_overlay_binned.png",
        cfg=cfg,
        abundance_overrides_vmr=abundance_overrides_vmr,
        zero_opacities=["CO,CO2"],
        plot_raw_platon=False
    )
    print("PLATON transit depth spectrum + plot saved.")


def Step7(ctx: dict, *, tag: str = "binned") -> None:
    print(f"STEP 7 — PLATON equilibrium retrieval (WASP-39b priors, emcee, tag={tag})")

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings

    from platon.constants import R_sun, M_jup
    from platon.combined_retriever import CombinedRetriever

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    def _set_fitinfo_key(fit_info, key: str, value) -> bool:
        """
        Best-effort setter for different PLATON FitInfo versions.
        Returns True if we found a dict-like storage and set the key.
        """
        for attr in ("default_params", "params", "fit_param_defaults", "_param_defaults"):
            d = getattr(fit_info, attr, None)
            if isinstance(d, dict):
                d[key] = value
                return True
        # Also try attribute set (some versions expose as attributes)
        if hasattr(fit_info, key):
            try:
                setattr(fit_info, key, value)
                return True
            except Exception:
                pass
        return False

    def _get_fitinfo_key(fit_info, key: str):
        for attr in ("default_params", "params", "fit_param_defaults", "_param_defaults"):
            d = getattr(fit_info, attr, None)
            if isinstance(d, dict) and key in d:
                return d[key]
        if hasattr(fit_info, key):
            try:
                return getattr(fit_info, key)
            except Exception:
                return None
        return None

    out_dir = ctx["out"]
    spec_path = os.path.join(out_dir, f"transmission_spectrum_{tag}.txt")
    if not os.path.exists(spec_path):
        raise RuntimeError(f"Missing spectrum: {spec_path}")

    wl_um, depth, elo, ehi = _load_transmission_spectrum_txt(spec_path)
    wl_um = np.asarray(wl_um, float)
    depth = np.asarray(depth, float)
    elo = np.asarray(elo, float)
    ehi = np.asarray(ehi, float)

    # Symmetrized errors
    err = 0.5 * (np.abs(elo) + np.abs(ehi))

    # ----------------------------
    # A) Robust data hygiene
    # ----------------------------
    n_in = len(wl_um)

    base = np.isfinite(wl_um) & np.isfinite(depth) & np.isfinite(err) & (err > 0)
    wl_um, depth, err = wl_um[base], depth[base], err[base]

    phys = (depth > 0.0) & (depth < 0.1)
    wl_um, depth, err = wl_um[phys], depth[phys], err[phys]

    if len(depth) < 3:
        raise RuntimeError("Not enough valid wavelength bins for retrieval after filtering")

    med = float(np.nanmedian(err))
    if not np.isfinite(med) or med <= 0:
        raise RuntimeError("Median uncertainty is non-finite or non-positive; check spectrum errors.")

    keep = err <= 1.0 * med
    dropped = int(np.count_nonzero(~keep))
    wl_um, depth, err = wl_um[keep], depth[keep], err[keep]
    if dropped > 0:
        print(f"[Step7] Dropped {dropped} high-uncertainty bin(s) (>{3.0:.1f}× median err).")

    if len(depth) < 3:
        raise RuntimeError("Not enough valid wavelength bins after outlier rejection")

    srt = np.argsort(wl_um)
    wl_um, depth, err = wl_um[srt], depth[srt], err[srt]

    # ----------------------------
    # B) Error floor (stability)
    # ----------------------------
    err_floor = float(ctx.get("platon_err_floor", 0.0))
    if err_floor <= 0:
        err_floor = 0.01 * float(np.nanmedian(err))
    err_floor = max(err_floor, 1e-5)
    err = np.maximum(err, err_floor)

    # ----------------------------
    # C) Build bin edges (meters)
    # ----------------------------
    mids = 0.5 * (wl_um[1:] + wl_um[:-1])
    edges = np.concatenate([
        [wl_um[0] - (mids[0] - wl_um[0])],
        mids,
        [wl_um[-1] + (wl_um[-1] - mids[-1])]
    ])
    if not np.all(np.diff(edges) > 0):
        d = np.diff(wl_um)
        edges = np.concatenate([[wl_um[0] - d[0] / 2], 0.5 * (wl_um[1:] + wl_um[:-1]), [wl_um[-1] + d[-1] / 2]])

    bins_m = np.column_stack([edges[:-1], edges[1:]]) * 1e-6
    if not np.all(bins_m[:, 1] > bins_m[:, 0]):
        raise RuntimeError("Non-positive bin widths encountered; check wavelength centers input.")

    # ----------------------------
    # D) Planet/star config
    # ----------------------------
    base_cfg = ctx.get("platon_cfg", None)

    rstar_rsun = 0.895
    rp_over_rs = 0.15
    mplanet_mjup = 0.281
    temperature_k = 1150.0
    cloudtop_pressure_pa = 1e5
    tstar_k = 5400.0

    if base_cfg is not None:
        rstar_rsun = float(getattr(base_cfg, "rstar_rsun", rstar_rsun))
        rp_over_rs = float(getattr(base_cfg, "rp_over_rs", rp_over_rs))
        mplanet_mjup = float(getattr(base_cfg, "mplanet_mjup", mplanet_mjup))
        temperature_k = float(getattr(base_cfg, "temperature_k", temperature_k))
        cloudtop_pressure_pa = float(getattr(base_cfg, "cloudtop_pressure_pa", cloudtop_pressure_pa))
        tstar_k = float(getattr(base_cfg, "tstar_k", tstar_k))

    Rs = rstar_rsun * float(R_sun)
    Rp0 = rp_over_rs * Rs
    Mp = mplanet_mjup * float(M_jup)
    log_cloudtop_P0 = float(np.log10(cloudtop_pressure_pa))

    # ----------------------------
    # E) Priors (WASP-39b-ish)
    # ----------------------------
    T_lo, T_hi = 950.0, 1250.0

    logZ_lo, logZ_hi = -0.5, 3.0
    logZ0 = float(ctx.get("platon_logZ0", 1.0))

    CO_lo, CO_hi = 0.05, 0.60
    CO_ratio0 = float(ctx.get("platon_CO_ratio0", 0.3))

    logPc_lo, logPc_hi = 3.3, 5.3
    Rp_frac = 0.022

    # haze / scattering
    log_scatt_lo, log_scatt_hi = -6.0, 1.0
    scatt_slope_lo, scatt_slope_hi = -8.0, 0.0

    # error inflation
    err_mult_lo, err_mult_hi = 0.5, 30.0

    # Initial values (validated BEFORE sampling)
    log_scatt0 = float(ctx.get("platon_log_scatt0", -3.0))
    scatt_slope0 = float(ctx.get("platon_scatt_slope0", -4.0))
    err_mult0 = float(ctx.get("platon_err_mult0", 10.0))

    log_scatt0 = float(np.clip(log_scatt0, log_scatt_lo + 1e-6, log_scatt_hi - 1e-6))
    scatt_slope0 = float(np.clip(scatt_slope0, scatt_slope_lo + 1e-6, scatt_slope_hi - 1e-6))
    err_mult0 = float(np.clip(err_mult0, err_mult_lo + 1e-6, err_mult_hi - 1e-6))

    nwalkers = int(ctx.get("platon_nwalkers", 80))
    nsteps = int(ctx.get("platon_nsteps", 2000))
    burn_frac = float(ctx.get("platon_burn_frac", 0.3))
    include_condensation = bool(ctx.get("platon_include_condensation", True))

    retriever = CombinedRetriever()

    fit_info = retriever.get_default_fit_info(
        Rs=Rs,
        Mp=Mp,
        Rp=Rp0,
        T=temperature_k,
        logZ=logZ0,
        CO_ratio=CO_ratio0,
        log_cloudtop_P=log_cloudtop_P0,
        T_star=tstar_k,
        free_retrieval=False,
        fit_vmr=False,
    )

    # ---- Force power-law scattering if possible (prevents Mie restriction) ----
    # Some PLATON versions support 'profile_type' to select scattering model.
    desired_profile = str(ctx.get("platon_profile_type", "power_law"))
    _set_fitinfo_key(fit_info, "profile_type", desired_profile)

    # Always set safe initial defaults inside bounds
    _set_fitinfo_key(fit_info, "scatt_slope", scatt_slope0)
    _set_fitinfo_key(fit_info, "error_multiple", err_mult0)

    # If we are in Mie mode, PLATON requires log_scatt_factor == 0.
    # We don't know for sure which mode your PLATON version is in, so we:
    #  - attempt power-law (above)
    #  - then enforce log_scatt_factor=0 if validation would otherwise fail
    log_scatt0_eff = 0.0
    profile_now = _get_fitinfo_key(fit_info, "profile_type")
    is_mie = isinstance(profile_now, str) and ("mie" in profile_now.lower())

    if is_mie:
        log_scatt0_eff = 0.0
        log_scatt_fit = False
        print("[Step7] Detected Mie scattering profile_type; fixing log_scatt_factor=0 (PLATON requirement).")
    else:
        log_scatt0_eff = log_scatt0
        log_scatt_fit = True

    _set_fitinfo_key(fit_info, "log_scatt_factor", float(log_scatt0_eff))

    # Fit params + priors
    fit_info.add_uniform_fit_param("Rp", (1.0 - Rp_frac) * Rp0, (1.0 + Rp_frac) * Rp0)
    fit_info.add_uniform_fit_param("T", T_lo, T_hi)
    fit_info.add_uniform_fit_param("log_cloudtop_P", logPc_lo, logPc_hi)
    fit_info.add_uniform_fit_param("logZ", logZ_lo, logZ_hi)
    fit_info.add_uniform_fit_param("CO_ratio", CO_lo, CO_hi)

    # Scattering params:
    # - if Mie, we MUST NOT fit log_scatt_factor (it must be exactly 0)
    if log_scatt_fit:
        fit_info.add_uniform_fit_param("log_scatt_factor", log_scatt_lo, log_scatt_hi)
    # scatt_slope: keep fitting (if your PLATON version rejects this in Mie too, set scatt_slope_lo/hi to a single value)
    fit_info.add_uniform_fit_param("scatt_slope", scatt_slope_lo, scatt_slope_hi)

    # Error multiple
    fit_info.add_uniform_fit_param("error_multiple", err_mult_lo, err_mult_hi)

    print("PLATON fit parameters:")
    print("  " + ", ".join(fit_info.fit_param_names))

    ndim = len(fit_info.fit_param_names)
    if nwalkers < 2 * ndim:
        nwalkers = 2 * ndim
        print(f"[Step7] Increased nwalkers to {nwalkers} (need >= 2*ndim, ndim={ndim})")

    print(f"Running emcee: walkers={nwalkers}, steps={nsteps}, nbins={len(depth)}")
    print(
        f"  Priors: T=[{T_lo},{T_hi}] K, logZ=[{logZ_lo},{logZ_hi}], CO=[{CO_lo},{CO_hi}], "
        f"logPc=[{logPc_lo},{logPc_hi}], Rp±{Rp_frac*100:.1f}%"
    )
    print(f"  err_floor={err_floor:g}, burn_frac={burn_frac}, include_condensation={include_condensation}")
    if log_scatt_fit:
        print(f"  Scattering: log_scatt_factor=[{log_scatt_lo},{log_scatt_hi}] (init {log_scatt0_eff:.2f}), scatt_slope=[{scatt_slope_lo},{scatt_slope_hi}] (init {scatt_slope0:.2f})")
    else:
        print(f"  Scattering: log_scatt_factor fixed to 0 (Mie). scatt_slope=[{scatt_slope_lo},{scatt_slope_hi}] (init {scatt_slope0:.2f})")
    print(f"  Error inflation: error_multiple=[{err_mult_lo},{err_mult_hi}] (init {err_mult0:.2f})")

    # ----------------------------
    # F) Run emcee
    # ----------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
        print(f"[Step7] Starting PLATON MCMC (~{nwalkers * nsteps} likelihood evals)")

        if tqdm is not None:
            with tqdm(total=nsteps, desc="PLATON emcee sampling", unit="step", leave=True) as pbar:
                result = retriever.run_emcee(
                    bins_m, depth, err,
                    None, None, None,
                    fit_info,
                    nwalkers=nwalkers,
                    nsteps=nsteps,
                    include_condensation=include_condensation,
                )
                pbar.n = nsteps
                pbar.refresh()
        else:
            result = retriever.run_emcee(
                bins_m, depth, err,
                None, None, None,
                fit_info,
                nwalkers=nwalkers,
                nsteps=nsteps,
                include_condensation=include_condensation,
            )

    # ----------------------------
    # G) Save any figure PLATON left open
    # ----------------------------
    bestfit_png = os.path.join(out_dir, f"07_platon_equilibrium_bestfit_{tag}.png")
    if plt.gcf().axes:
        plt.tight_layout()
        plt.savefig(bestfit_png, dpi=200)
        print(f"Saved: {bestfit_png}")
    plt.close("all")

    # ----------------------------
    # H) Extract posterior (step-wise burn if available)
    # ----------------------------
    if hasattr(result, "chain") and hasattr(result, "lnprobability"):
        c = np.asarray(result.chain)
        lp = np.asarray(result.lnprobability)
        burn_steps = int(burn_frac * c.shape[1])
        c = c[:, burn_steps:, :]
        lp = lp[:, burn_steps:]
        chain_post = c.reshape(-1, c.shape[-1])
        lnprob_post = lp.reshape(-1)
    else:
        chain = np.asarray(result.flatchain)
        lnprob = np.asarray(result.flatlnprobability)
        good = np.isfinite(lnprob) & np.all(np.isfinite(chain), axis=1)
        chain = chain[good]
        lnprob = lnprob[good]
        burn_n = int(burn_frac * chain.shape[0])
        burn_n = min(burn_n, max(0, chain.shape[0] - 10))
        chain_post = chain[burn_n:]
        lnprob_post = lnprob[burn_n:]

    good = np.isfinite(lnprob_post) & np.all(np.isfinite(chain_post), axis=1)
    chain_post = chain_post[good]
    lnprob_post = lnprob_post[good]
    if chain_post.shape[0] < 50:
        raise RuntimeError("Too few finite samples; try increasing steps or tightening priors slightly.")
    print(f"[Step7] Posterior samples kept: {chain_post.shape[0]}")

    # ----------------------------
    # I) Corner + save posterior
    # ----------------------------
    corner_png = os.path.join(out_dir, f"07_corner_platon_equilibrium_{tag}.png")
    save_corner(corner_png, chain_post, fit_info.fit_param_names)
    print(f"Saved: {corner_png}")

    i_best = int(np.argmax(lnprob_post))
    p_best = chain_post[i_best]
    best = dict(zip(fit_info.fit_param_names, map(float, p_best)))
    ctx[f"platon_bestfit_{tag}"] = best

    npz_path = os.path.join(out_dir, f"07_platon_equilibrium_posterior_{tag}.npz")
    np.savez(
        npz_path,
        fit_param_names=np.array(fit_info.fit_param_names, dtype=object),
        chain=chain_post,
        lnprob=lnprob_post,
        meta=dict(
            target="WASP-39b",
            tag=tag,
            err_floor=float(err_floor),
            burn_frac=float(burn_frac),
            nwalkers=int(nwalkers),
            nsteps=int(nsteps),
            profile_type=str(_get_fitinfo_key(fit_info, "profile_type")),
            priors=dict(
                T=[T_lo, T_hi],
                logZ=[logZ_lo, logZ_hi],
                CO_ratio=[CO_lo, CO_hi],
                log_cloudtop_P=[logPc_lo, logPc_hi],
                log_scatt_factor=([log_scatt_lo, log_scatt_hi] if log_scatt_fit else [0.0, 0.0]),
                scatt_slope=[scatt_slope_lo, scatt_slope_hi],
                error_multiple=[err_mult_lo, err_mult_hi],
                Rp_frac=Rp_frac,
            ),
            input_bins=int(n_in),
            base_valid=int(np.count_nonzero(base)),
            phys_valid=int(np.count_nonzero(phys)),
            higherr_dropped=int(dropped),
            final_bins=int(len(depth)),
        ),
    )
    print(f"Saved posterior: {npz_path}")


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
    ap.add_argument("--binning-factor", type=int, default=5)
    ap.add_argument("--n-wavelength-bins", type=int, default=75)
    ap.add_argument("--spec-time-bin-factor", type=int, default=10)
    ap.add_argument(
        "--drop-high-err",
        action="store_true",
        help="Drop wavelength points with err > 3×median(err).",
    )
    ap.add_argument(
        "--high-err-sigma",
        type=float,
        default=3.0,
        help="Threshold multiplier: drop if err > (this)×median(err). Used with --drop-high-err.",
    )

    args = ap.parse_args()



    ctx: dict = {}
    ctx["white_light_mode"] = "binned"
    ctx["platon_profile_type"] = "power_law"
    Step = {
        0: lambda: Step0(ctx),
        1: lambda: Step1(ctx),
        2: lambda: Step2(ctx, args.binning_factor),
        3: lambda: Step3(ctx),
        4: lambda: Step4(ctx),
        5: lambda: Step5(
            ctx,
            spec_time_bin=args.spec_time_bin_factor,
            n_wl_bins=args.n_wavelength_bins,
            white_tag="unbinned",
        ),
        6: lambda: Step6(ctx),
        7: lambda: Step7(ctx, tag="unbinned"),
    }

    for s in sorted(parse_steps(args.steps)):
        Step[s]()


if __name__ == "__main__":
    main()
