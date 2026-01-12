from __future__ import annotations

import h5py
import numpy as np
from typing import Dict, Optional, Any


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

