from __future__ import annotations

import h5py
import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
from typing import Tuple

def load_prepared_lightcurve(h5_path: str) -> Dict[str, Any]:
    """
    Load the prepared JWST light curve from the course .h5 files.

    Expected datasets:
      - bjd: (n_int,)
      - wavelength: (n_wl,)
      - flux: (n_int, n_wl)
      - flux_err: (n_int, n_wl)
      - optionally it_index_non_binned, oot_index_non_binned
    """
    with h5py.File(h5_path, "r") as f:
        data = {
            "bjd": f["bjd"][:],
            "wavelength": f["wavelength"][:],
            "flux": f["flux"][:],
            "flux_err": f["flux_err"][:],
            "it_index_non_binned": f["it_index_non_binned"][:] if "it_index_non_binned" in f else None,
            "oot_index_non_binned": f["oot_index_non_binned"][:] if "oot_index_non_binned" in f else None,
        }
    return data


def save_transmission_spectrum_txt(
    out_txt: str,
    wl_centers_um: np.ndarray,
    depth: np.ndarray,
    depth_err_lo: np.ndarray,
    depth_err_hi: np.ndarray,
) -> None:
    header = "wl_center_um depth depth_err_lo depth_err_hi"
    arr = np.column_stack([wl_centers_um, depth, depth_err_lo, depth_err_hi])
    np.savetxt(out_txt, arr, header=header)

def load_transmission_spectrum_txt(path):
    """
    Load transmission spectrum saved by save_transmission_spectrum_txt.
    Returns wl, depth, elo, ehi
    """
    data = np.loadtxt(path)
    wl, depth, elo, ehi = data.T
    return wl, depth, elo, ehi


def load_binned_spectrum(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load binned transmission spectrum.

    Accepts:
      - 4 cols: wavelength_um, depth, err_lo, err_hi
      - 3 cols: wavelength_um, depth, err
      - 2 cols: wavelength_um, depth (errors set to nan)

    Returns:
      wl_um, depth, err (1-sigma symmetric; for 4-col uses mean of lo/hi)
    """
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected spectrum format in {path}: shape={arr.shape}")

    wl_um = arr[:, 0].astype(float)
    depth = arr[:, 1].astype(float)

    if arr.shape[1] >= 4:
        err = 0.5 * (np.abs(arr[:, 2]) + np.abs(arr[:, 3])).astype(float)
    elif arr.shape[1] == 3:
        err = np.abs(arr[:, 2]).astype(float)
    else:
        err = np.full_like(depth, np.nan, dtype=float)

    return wl_um, depth, err


