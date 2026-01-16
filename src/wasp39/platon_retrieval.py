# src/wasp39/platon_retrieval.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

# NOTE:
# Do NOT import platon.combined_retriever / FitInfo / Plotter here.
# Those pull in extra deps (astropy/pandas) and should live in Step7.

def run_platon_retrieval_emcee(
    wl_um: np.ndarray,
    depth_obs: np.ndarray,
    elo: np.ndarray,
    ehi: np.ndarray,
    planet_params: Dict[str, float],
    out_dir: str,
    tag: str = "binned",
    wl_min_um: float = 1.0,
    wl_max_um: float = 5.0,
    nwalkers: int = 24,
    nsteps_burn: int = 50,
    nsteps_prod: int = 150,
    ncores: int = 1,
    max_points: Optional[int] = 50,
    downsample_method: str = "uniform",
    seed: int = 0,
    make_plots: bool = True,
    n_sub_bin: int = 5,
    fit_mode: str = "3param",
    fixed_custom_abundances: Optional[Dict[str, float]] = None,
    fixed_atm_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Placeholder 'clean' retrieval entrypoint so Step6 imports stop crashing.

    Replace this body with your full emcee implementation (the one that imports emcee,
    uses your platon_model wrapper, etc.), but keep CombinedRetriever imports OUT of this file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Minimal no-op result so pipeline continues.
    # (This prevents Step6 from failing while you keep Step7 for the simple retrieval.)
    result: Dict[str, Any] = {
        "theta_names": [],
        "q16": np.array([]),
        "q50": np.array([]),
        "q84": np.array([]),
        "note": "run_platon_retrieval_emcee is currently a stub. Use Step7 for simple PLATON retrieval.",
    }


    # Save something so you can see it ran.
    np.savez(os.path.join(out_dir, f"07_platon_retrieval_{tag}.npz"), **result)
    return result
