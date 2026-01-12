from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from typing import Optional


def save_white_light_plot(
    out_png: str,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    title: str,
    *,
    max_points_for_full_errorbars: int = 2000,
    errbar_stride: int = 50,
) -> None:
    """
    Plot white light curve without turning into an errorbar carpet.

    Strategy:
      - If N is small: standard errorbar plot.
      - If N is large: plot points without yerr + (optional) sparse errorbars.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    yerr = np.asarray(yerr)

    fig = plt.figure(figsize=(10, 6))

    n = len(t)
    if n <= max_points_for_full_errorbars:
        plt.errorbar(t, y, yerr=yerr, fmt=".", capsize=2, alpha=0.9, markersize=4)
    else:
        # main: fast, readable point cloud (no yerr)
        plt.plot(t, y, ".", markersize=1.5, alpha=0.35, rasterized=True, label="Data")

        # optional: sparse errorbars so you still "see" typical uncertainties
        if errbar_stride is not None and errbar_stride > 0:
            idx = np.arange(0, n, errbar_stride)
            plt.errorbar(
                t[idx], y[idx], yerr=yerr[idx],
                fmt="none", ecolor="black", elinewidth=0.6, capsize=0, alpha=0.25
            )

    plt.xlabel("Time (BJD)")
    plt.ylabel("Normalized flux")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_data_plus_model(
    out_png: str,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    model: np.ndarray,
    title: str,
    *,
    max_points_for_full_errorbars: int = 2000,
    errbar_stride: int = 50,
) -> None:
    """
    Plot data + model without an errorbar carpet for large-N.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    model = np.asarray(model)

    fig = plt.figure(figsize=(12, 6))

    # model first (clean line)
    plt.plot(t, model, lw=2.5, label="Model")

    n = len(t)
    if n <= max_points_for_full_errorbars:
        plt.errorbar(t, y, yerr=yerr, fmt=".", capsize=2, label="Data", alpha=0.85, markersize=4)
    else:
        plt.plot(t, y, ".", markersize=1.5, alpha=0.35, rasterized=True, label="Data")

        if errbar_stride is not None and errbar_stride > 0:
            idx = np.arange(0, n, errbar_stride)
            plt.errorbar(
                t[idx], y[idx], yerr=yerr[idx],
                fmt="none", ecolor="black", elinewidth=0.6, capsize=0, alpha=0.25
            )

    plt.xlabel("Time (BJD)")
    plt.ylabel("Normalized flux")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_bestfit_and_residuals(
    out_png,
    t,
    y,
    yerr,
    model,
    title,
    *,
    max_points_for_full_errorbars: int = 2000,
    errbar_stride: int = 50,
):
    """
    Best-fit plot + residuals, avoiding errorbar carpets for large-N.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    model = np.asarray(model)

    residuals = y - model
    chi2 = np.nansum((residuals / yerr) ** 2)
    ndof = max(1, len(y) - 9)  # 9 params in white-light fit
    red_chi2 = chi2 / ndof

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    n = len(t)

    # --- top panel: data + model ---
    ax1.plot(t, model, lw=2.5, label="Best-fit model")

    if n <= max_points_for_full_errorbars:
        ax1.errorbar(t, y, yerr=yerr, fmt="o", markersize=4, capsize=2, alpha=0.85, label="Data")
    else:
        ax1.plot(t, y, ".", markersize=1.5, alpha=0.35, rasterized=True, label="Data")

        if errbar_stride is not None and errbar_stride > 0:
            idx = np.arange(0, n, errbar_stride)
            ax1.errorbar(
                t[idx], y[idx], yerr=yerr[idx],
                fmt="none", ecolor="black", elinewidth=0.6, capsize=0, alpha=0.25
            )

    ax1.set_ylabel("Normalized flux")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.text(
        0.02, 0.98, fr"Reduced $\chi^2 = {red_chi2:.2f}$",
        transform=ax1.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # --- bottom panel: residuals ---
    if n <= max_points_for_full_errorbars:
        ax2.errorbar(t, residuals, yerr=yerr, fmt="o", markersize=4, capsize=2, alpha=0.85)
    else:
        ax2.plot(t, residuals, ".", markersize=1.5, alpha=0.35, rasterized=True)

        if errbar_stride is not None and errbar_stride > 0:
            idx = np.arange(0, n, errbar_stride)
            ax2.errorbar(
                t[idx], residuals[idx], yerr=yerr[idx],
                fmt="none", ecolor="black", elinewidth=0.6, capsize=0, alpha=0.25
            )

    ax2.axhline(0.0, lw=1.5, linestyle="--")
    ax2.set_xlabel("Time (BJD)")
    ax2.set_ylabel("Residuals")
    ax2.grid(True, alpha=0.3)

    max_abs = np.nanmax(np.abs(residuals))
    if np.isfinite(max_abs):
        ax2.set_ylim(-1.2 * max_abs, 1.2 * max_abs)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)



def save_corner(out_png: str, samples: np.ndarray, labels: list[str]) -> None:
    fig = corner.corner(samples, labels=labels, show_titles=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_transmission_spectrum_plot(out_png: str,
                                   wl_um: np.ndarray,
                                   depth: np.ndarray,
                                   err_lo: np.ndarray,
                                   err_hi: np.ndarray,
                                   title: str = "Transmission spectrum") -> None:
    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(wl_um, depth, yerr=[err_lo, err_hi], fmt="o", capsize=3, markersize=5)
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Transit depth (Rp/R*)^2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)