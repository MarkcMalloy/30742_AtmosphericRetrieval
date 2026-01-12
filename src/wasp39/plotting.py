from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from typing import Optional


def save_white_light_plot(out_png: str, t: np.ndarray, y: np.ndarray, yerr: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(t, y, yerr=yerr, fmt=".", capsize=2)
    plt.xlabel("Time (BJD)")
    plt.ylabel("Normalized flux")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_data_plus_model(out_png: str, t: np.ndarray, y: np.ndarray, yerr: np.ndarray, model: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(t, model, lw=2.5, label="Model")
    plt.errorbar(t, y, yerr=yerr, fmt=".", capsize=2, label="Data", alpha=0.8)
    plt.xlabel("Time (BJD)")
    plt.ylabel("Normalized flux")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
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

def save_bestfit_and_residuals(out_png, t, y, yerr, model, title):
    residuals = y - model
    chi2 = np.nansum((residuals / yerr) ** 2)
    ndof = max(1, len(y) - 9)  # 9 params in white-light fit
    red_chi2 = chi2 / ndof

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    ax1.errorbar(t, y, yerr=yerr, fmt="o", markersize=4, capsize=2, alpha=0.85, label="Data")
    ax1.plot(t, model, lw=2.5, label="Best-fit model")
    ax1.set_ylabel("Normalized flux")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.text(0.02, 0.98, fr"Reduced $\chi^2 = {red_chi2:.2f}$",
             transform=ax1.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax2.errorbar(t, residuals, yerr=yerr, fmt="o", markersize=4, capsize=2, alpha=0.85)
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
