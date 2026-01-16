
from __future__ import annotations
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import emcee
from typing import Callable, Tuple, Optional, Union
from dataclasses import replace

from .lightcurve import TransitConfig, batman_model

N_CORES = 8


@dataclass
class WhiteLightLogProb:
    t_rel: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    t_ref: float
    limb_dark: str

    t0_init: float
    t0_width: float

    per_bounds: Tuple[float, float]
    a_bounds: Tuple[float, float]
    b_bounds: Tuple[float, float]
    rp_bounds: Tuple[float, float]
    u_bounds: Tuple[float, float]
    c0_bounds: Tuple[float, float]
    c1_bounds: Tuple[float, float]

    u_gauss_mu: Optional[Tuple[float, float]] = None
    u_gauss_sigma: Union[float, Tuple[float, float]] = 0.05
    per_gauss_mu: Optional[float] = None
    per_gauss_sigma: Optional[float] = None
    a_gauss_mu: Optional[float] = None
    a_gauss_sigma: Optional[float] = None

    @staticmethod
    def gauss_lnprior(x: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return -np.inf
        return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))

    def log_prior(self, p: np.ndarray) -> float:
        t0, per, a, b, rp, u1, u2, c0, c1 = p

        if not (self.t0_init - self.t0_width < t0 < self.t0_init + self.t0_width):
            return -np.inf
        if not (self.per_bounds[0] < per < self.per_bounds[1]):
            return -np.inf
        if not (self.a_bounds[0] < a < self.a_bounds[1]):
            return -np.inf
        if not (self.b_bounds[0] < b < self.b_bounds[1]):
            return -np.inf
        if b >= a:
            return -np.inf
        if not (self.rp_bounds[0] < rp < self.rp_bounds[1]):
            return -np.inf
        if not (self.u_bounds[0] < u1 < self.u_bounds[1]):
            return -np.inf
        if not (self.u_bounds[0] < u2 < self.u_bounds[1]):
            return -np.inf
        if not (self.c0_bounds[0] < c0 < self.c0_bounds[1]):
            return -np.inf
        if not (self.c1_bounds[0] < c1 < self.c1_bounds[1]):
            return -np.inf

        lp = 0.0

        if self.per_gauss_mu is not None:
            lp += self.gauss_lnprior(per, self.per_gauss_mu, self.per_gauss_sigma)
        if self.a_gauss_mu is not None:
            lp += self.gauss_lnprior(a, self.a_gauss_mu, self.a_gauss_sigma)

        if self.u_gauss_mu is not None:
            mu1, mu2 = self.u_gauss_mu
            sig1 = sig2 = self.u_gauss_sigma if not isinstance(self.u_gauss_sigma, tuple) else self.u_gauss_sigma
            lp += self.gauss_lnprior(u1, mu1, sig1)
            lp += self.gauss_lnprior(u2, mu2, sig2)

        return lp

    def model_from_params(self, p: np.ndarray) -> np.ndarray:
        t0, per, a, b, rp, u1, u2, c0, c1 = p
        inc = np.degrees(np.arccos(np.clip(b / a, -1.0, 1.0)))

        cfg = TransitConfig(
            t0=t0 - self.t_ref,
            per=per,
            a=a,
            inc=inc,
            u=(u1, u2),
            limb_dark=self.limb_dark,
        )

        transit = batman_model(self.t_rel, cfg, rp)
        baseline = c0 + c1 * self.t_rel
        return transit * baseline

    def __call__(self, p: np.ndarray) -> float:
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf

        model = self.model_from_params(p)
        inv_sigma2 = 1.0 / (self.flux_err ** 2)
        return -0.5 * np.sum((self.flux - model) ** 2 * inv_sigma2 + np.log(2 * np.pi / inv_sigma2)) + lp


def fit_white_light_mcmc(
    t, flux, flux_err, cfg_init, rp_init,
    nwalkers=64, nsteps_burn=3000, nsteps_prod=8000, thin=1, progress=True,
    t0_width=0.1,
    per_bounds=(3.5, 4.5),
    a_bounds=(10.0, 13.0),
    b_bounds=(0.0, 1.2),
    rp_bounds=(0.1, 0.2),
    u_bounds=(-1.0, 1.0),
    c0_bounds=(0.9, 1.1),
    c1_bounds=(-0.01, 0.01),
    u_gauss_mu=None,
    u_gauss_sigma=0.05,
    per_gauss_mu=None,
    per_gauss_sigma=None,
    a_gauss_mu=None,
    a_gauss_sigma=None,
):
    t = np.asarray(t, float)
    flux = np.asarray(flux, float)
    flux_err = np.asarray(flux_err, float)

    t_ref = np.median(t)
    t_rel = t - t_ref

    b_init = cfg_init.a * np.cos(np.deg2rad(cfg_init.inc))

    p_init = np.array([
        cfg_init.t0,
        cfg_init.per,
        cfg_init.a,
        b_init,
        rp_init,
        cfg_init.u[0],
        cfg_init.u[1],
        1.0,
        0.0,
    ])

    ndim = len(p_init)
    scales = np.array([0.005, 0.02*p_init[1], 0.05, 0.02, 0.002, 0.02, 0.02, 0.01, 1e-4])
    pos0 = p_init + scales * np.random.randn(nwalkers, ndim)

    logprob = WhiteLightLogProb(
        t_rel=t_rel,
        flux=flux,
        flux_err=flux_err,
        t_ref=t_ref,
        limb_dark=cfg_init.limb_dark,
        t0_init=p_init[0],
        t0_width=t0_width,
        per_bounds=per_bounds,
        a_bounds=a_bounds,
        b_bounds=b_bounds,
        rp_bounds=rp_bounds,
        u_bounds=u_bounds,
        c0_bounds=c0_bounds,
        c1_bounds=c1_bounds,
        u_gauss_mu=u_gauss_mu,
        u_gauss_sigma=u_gauss_sigma,
        per_gauss_mu=per_gauss_mu,
        per_gauss_sigma=per_gauss_sigma,
        a_gauss_mu=a_gauss_mu,
        a_gauss_sigma=a_gauss_sigma,
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_CORES) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, pool=pool)
        sampler.run_mcmc(pos0, nsteps_burn, progress=progress)
        state = sampler.get_last_sample()
        sampler.run_mcmc(state, nsteps_prod, progress=progress)

    chain = sampler.get_chain(discard=nsteps_burn//2, thin=thin, flat=True)
    best_params = np.median(chain, axis=0)
    best_model = logprob.model_from_params(best_params)

    labels = ["t0","per","a","b","rp","u1","u2","c0","c1"]
    return chain, labels, best_params, best_model
