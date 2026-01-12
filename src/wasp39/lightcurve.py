from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import batman


@dataclass
class TransitConfig:
    t0: float
    per: float
    a: float          # a/R*
    inc: float        # degrees
    ecc: float = 0.0
    w: float = 90.0   # degrees
    u: tuple = (0.3, 0.1)
    limb_dark: str = "quadratic"


def batman_model(t: np.ndarray, cfg: TransitConfig, rp: float) -> np.ndarray:
    params = batman.TransitParams()
    params.t0 = cfg.t0
    params.per = cfg.per
    params.a = cfg.a
    params.inc = cfg.inc
    params.ecc = cfg.ecc
    params.w = cfg.w
    params.rp = rp
    params.u = list(cfg.u)
    params.limb_dark = cfg.limb_dark
    m = batman.TransitModel(params, t)
    return m.light_curve(params)

def batman_model_with_linear_baseline(t: np.ndarray, cfg: TransitConfig, rp: float, c0: float, c1: float, t_ref: float) -> np.ndarray:
    transit = batman_model(t, cfg, rp)
    baseline = c0 + c1 * (t - t_ref)
    return baseline * transit



def batman_model_fixed_except_rp(t: np.ndarray, cfg: TransitConfig, rp: float) -> np.ndarray:
    # alias kept for compatibility with your existing naming
    return batman_model(t, cfg, rp)
