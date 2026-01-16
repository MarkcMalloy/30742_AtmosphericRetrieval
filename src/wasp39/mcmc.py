from __future__ import annotations

import numpy as np
import emcee
from typing import Callable, Tuple, Optional, Union
from dataclasses import replace

from .lightcurve import TransitConfig, batman_model


def log_likelihood(theta: np.ndarray, t: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                   model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    model = model_fn(t, theta)
    inv_sigma2 = 1.0 / (yerr ** 2)
    return -0.5 * np.sum((y - model) ** 2 * inv_sigma2 + np.log(2.0 * np.pi / inv_sigma2))



def run_emcee(
    logprob_fn: Callable[[np.ndarray], float],
    ndim: int,
    nwalkers: int,
    p0: np.ndarray,
    nsteps_burn: int,
    nsteps_prod: int,
    progress: bool = True,
) -> emcee.EnsembleSampler:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn)
    sampler.run_mcmc(p0, nsteps_burn, progress=progress)
    state = sampler.get_last_sample()
    sampler.run_mcmc(state, nsteps_prod, progress=progress)
    return sampler


# --- Bin-depth (Rp/R*) MCMC helpers ---

def logprior_rp(rp: float, rp_bounds: Tuple[float, float]) -> float:
    lo, hi = rp_bounds
    return 0.0 if (lo < rp < hi) else -np.inf


def fit_one_bin_mcmc(
    t: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    cfg: TransitConfig,
    rp_init: float,
    rp_bounds: Tuple[float, float] = (0.05, 0.3),
    nwalkers: int = 32,
    nsteps_burn: int = 2000,
    nsteps_prod: int = 3000,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Rp/R* for a single wavelength bin using emcee.
    Numerically stable time: uses t_rel = t - median(t) internally for BATMAN.
    Returns:
        samples_rp: (n_samples,)
        model_median: (len(t),)
    """
    # --- ensure float arrays ---
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    # --- numerically stable time ---
    t_ref = float(np.median(t))
    t_rel = t - t_ref

    # shift t0 into the same relative frame for BATMAN
    cfg_rel = TransitConfig(
        t0=float(cfg.t0) - t_ref,
        per=float(cfg.per),
        a=float(cfg.a),
        inc=float(cfg.inc),
        u=(float(cfg.u[0]), float(cfg.u[1])),
        limb_dark=cfg.limb_dark,
    )

    def logprob(theta: np.ndarray) -> float:
        rp = float(theta[0])
        lp = logprior_rp(rp, rp_bounds)
        if not np.isfinite(lp):
            return -np.inf

        model = batman_model(t_rel, cfg_rel, rp)
        inv_sigma2 = 1.0 / (flux_err ** 2)
        ll = -0.5 * np.sum((flux - model) ** 2 * inv_sigma2 + np.log(2.0 * np.pi / inv_sigma2))
        return lp + ll

    ndim = 1
    p0 = rp_init + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = run_emcee(logprob, ndim, nwalkers, p0, nsteps_burn, nsteps_prod, progress=progress)

    flat = sampler.get_chain(discard=0, flat=True)
    samples = flat[:, 0]
    rp_med = np.median(samples)

    # evaluate model at median (still returned on original time grid length)
    model_med = batman_model(t_rel, cfg_rel, rp_med)
    return samples, model_med


def fit_white_light_mcmc(
    t: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    cfg_init: TransitConfig,
    rp_init: float,
    *,
    nwalkers: int = 50,
    nsteps_burn: int = 2000,
    nsteps_prod: int = 2000,
    thin: int = 15,
    progress: bool = True,
    # Priors (loosely following your Proj_WASP.py)
    t0_width: float = 0.1,
    per_bounds: Tuple[float, float] = (4.045, 4.065),
    a_bounds: Tuple[float, float] = (10.5, 12.0),
    inc_bounds: Tuple[float, float] = (86.5, 88.5),
    rp_bounds: Tuple[float, float] = (0.13, 0.16),
    u_bounds: Tuple[float, float] = (-1.0, 1.0),
    c0_bounds: Tuple[float, float] = (0.9, 1.1),
    c1_bounds: Tuple[float, float] = (-0.01, 0.01),
    # Optional Gaussian priors on limb darkening (u1,u2)
    u_gauss_mu: Optional[Tuple[float, float]] = None,
    u_gauss_sigma: Union[float, Tuple[float, float]] = 0.05,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """
    White-light MCMC fit: transit (BATMAN) * linear baseline.

    Parameters sampled (absolute t0):
        [t0, per, a, inc, rp, u1, u2, c0, c1]

    Numerically stable time:
        Uses t_rel = t - median(t) internally for BATMAN and baseline.

    Returns:
        chain_flat: (Nsamples, 9)
        labels: list[str]
        best_params: (9,) median of posterior (t0 remains absolute BJD)
        best_model: (len(t),) model evaluated at best_params
    """
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    # --- numerically stable time ---
    t_ref = float(np.median(t))
    t_rel = t - t_ref

    labels = ["t0", "per", "a", "inc", "rp", "u1", "u2", "c0", "c1"]

    # --- initial parameter vector (from cfg_init + rp_init) ---
    p_init = np.array([
        float(cfg_init.t0),   # absolute
        float(cfg_init.per),
        float(cfg_init.a),
        float(cfg_init.inc),
        float(rp_init),
        float(cfg_init.u[0]),
        float(cfg_init.u[1]),
        1.0,   # c0
        0.0,   # c1
    ], dtype=float)

    # --- log prior (flat bounds) ---
    def log_prior(p: np.ndarray) -> float:
        t0, per, a, inc, rp, u1, u2, c0, c1 = p

        if not (p_init[0] - t0_width < t0 < p_init[0] + t0_width): return -np.inf
        if not (per_bounds[0] < per < per_bounds[1]):              return -np.inf
        if not (a_bounds[0] < a < a_bounds[1]):                    return -np.inf
        if not (inc_bounds[0] < inc < inc_bounds[1]):              return -np.inf
        if not (rp_bounds[0] < rp < rp_bounds[1]):                 return -np.inf
        if not (u_bounds[0] < u1 < u_bounds[1]):                   return -np.inf
        if not (u_bounds[0] < u2 < u_bounds[1]):                   return -np.inf
        if not (c0_bounds[0] < c0 < c0_bounds[1]):                 return -np.inf
        if not (c1_bounds[0] < c1 < c1_bounds[1]):                 return -np.inf

        lp = 0.0

        # Optional Gaussian priors on limb darkening
        if u_gauss_mu is not None:
            mu1, mu2 = u_gauss_mu
            if isinstance(u_gauss_sigma, tuple):
                sig1, sig2 = u_gauss_sigma
            else:
                sig1 = float(u_gauss_sigma)
                sig2 = float(u_gauss_sigma)
            if sig1 <= 0 or sig2 <= 0:
                return -np.inf
            lp += -0.5 * ((u1 - mu1) / sig1) ** 2 - np.log(sig1 * np.sqrt(2.0 * np.pi))
            lp += -0.5 * ((u2 - mu2) / sig2) ** 2 - np.log(sig2 * np.sqrt(2.0 * np.pi))

        return lp

    # --- model ---
    def model_from_params(p: np.ndarray) -> np.ndarray:
        # NOTE: t0 stays absolute in p, but BATMAN uses relative time,
        # so we shift t0 into the same frame: t0_rel = t0 - t_ref
        t0, per, a, inc, rp, u1, u2, c0, c1 = p
        cfg = TransitConfig(
            t0=float(t0) - t_ref,  # shifted for BATMAN
            per=float(per),
            a=float(a),
            inc=float(inc),
            u=(float(u1), float(u2)),
            limb_dark=cfg_init.limb_dark,
        )
        transit = batman_model(t_rel, cfg, float(rp))

        # baseline in the same relative frame (more stable than (t - mean(t)))
        baseline = float(c0) + float(c1) * t_rel
        return transit * baseline

    def log_likelihood_full(p: np.ndarray) -> float:
        m = model_from_params(p)
        inv_sigma2 = 1.0 / (flux_err ** 2)
        return -0.5 * np.sum((flux - m) ** 2 * inv_sigma2 + np.log(2.0 * np.pi / inv_sigma2))

    def logprob(p: np.ndarray) -> float:
        lp = log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_full(p)

    ndim = len(p_init)
    pos0 = p_init + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
    sampler.run_mcmc(pos0, nsteps_burn, progress=progress)
    state = sampler.get_last_sample()
    sampler.run_mcmc(state, nsteps_prod, progress=progress)

    chain_flat = sampler.get_chain(discard=max(1, nsteps_burn // 2), thin=thin, flat=True)

    best_params = np.median(chain_flat, axis=0)     # t0 remains absolute
    best_model = model_from_params(best_params)

    return chain_flat, labels, best_params, best_model
