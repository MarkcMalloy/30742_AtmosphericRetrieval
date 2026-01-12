import numpy as np
import h5py
import matplotlib.pyplot as plt
import batman
import emcee
import os
import corner
from tqdm.auto import tqdm  # Better for notebooks/Jupyter

# Clear terminal (optional, works in terminals)
os.system('clear')

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
print("base_dir: ", base_dir)
h5_path = os.path.join(base_dir, 'WASP-39b_JWST_PRISM_2022-07-10_prepared_light_curve.h5')
print("h5_path: ", h5_path)
out_dir = os.path.join(base_dir, 'outfiles_project')

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Load data
def load_prepared_lightcurve(h5_path: str):
    """
    Load the prepared JWST light curve from the course .h5 files.
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

# Load the data
data = load_prepared_lightcurve(h5_path)

# Show the keys
print('Keys in the data dictionary:')
print(data.keys())

# Number of observations (integrations)
n_integrations = data['bjd'].shape[0]
print(f"Number of integrations: {n_integrations}")

# Number of wavelength bins
n_wl = data['wavelength'].shape[0]
print(f"Number of wavelength bins: {n_wl}")

# Shape of flux
print("Flux.shape =", data['flux'].shape)

# Extract variables
bjd = data['bjd']
wavelength = data['wavelength']
flux = data['flux']
flux_err = data['flux_err']
oot_index_non_binned = data['oot_index_non_binned']
# it_index_non_binned not used in white light curve

# Create white light curve (PRISM-specific bins)
white_lc_startwlbin = 83
white_lc_endwlbin = 339
white = np.sum(flux[:, white_lc_startwlbin:white_lc_endwlbin], axis=1)
white_err = np.sqrt(np.sum(flux_err[:, white_lc_startwlbin:white_lc_endwlbin]**2, axis=1))

# Normalization function
def normalize_data(flux, flux_err, oot_index):
    median_oot = np.median(flux[oot_index])
    return flux / median_oot, flux_err / median_oot

# Normalize white light curve
white_norm, white_err_norm = normalize_data(white, white_err, oot_index_non_binned)

# Plot 1: Normalized white light curve (raw)
fig1 = plt.figure(figsize=(10, 6))
plt.errorbar(bjd, white_norm, yerr=white_err_norm, label='Normalized light curve',
             fmt='.', capsize=2, markersize=2, alpha=0.6)
plt.xlabel('Time (BJD)')
plt.ylabel('Normalized flux')
plt.title('Normalized white light curve of WASP-39b')
plt.legend()
plt.tight_layout()
fig1.savefig(os.path.join(out_dir, '01_normalized_white_lightcurve.png'), dpi=200)
plt.close(fig1)

# Updated bin_data (loop-based, handles any length including incomplete final bin)
def bin_data(bjd, flux, flux_err, binning_factor):
    """
    Bins time-series data, including any incomplete final bin.
    """
    n_integrations = bjd.shape[0]
    n_integrations_binned = int(np.ceil(n_integrations / binning_factor))
    
    bjd_binned = np.empty(n_integrations_binned)
    flux_binned = np.empty(n_integrations_binned)
    flux_err_binned = np.empty(n_integrations_binned)
    
    for i in range(n_integrations_binned):
        start = i * binning_factor
        end = min(start + binning_factor, n_integrations)
        n_points = end - start
        
        bjd_binned[i] = np.mean(bjd[start:end])
        flux_binned[i] = np.mean(flux[start:end])
        flux_err_binned[i] = np.sqrt(np.sum(flux_err[start:end]**2)) / n_points
    
    return bjd_binned, flux_binned, flux_err_binned

# Bin the normalized white light curve
binning_factor = 10
bjd_binned, white_binned, white_err_binned = bin_data(bjd, white_norm, white_err_norm, binning_factor)

# Plot 2: Binned white light curve
fig2 = plt.figure(figsize=(10, 6))
plt.errorbar(bjd_binned, white_binned, yerr=white_err_binned,
             label='Binned light curve (bin factor = 10)', fmt='o', capsize=4, markersize=6)
plt.xlabel('Time (BJD)')
plt.ylabel('Normalized flux')
plt.title('Binned white light curve of WASP-39b')
plt.legend()
plt.tight_layout()
fig2.savefig(os.path.join(out_dir, '02_binned_white_lightcurve.png'), dpi=200)
plt.close(fig2)

# White light curve fitting setup
light_curve_params_to_fit = ['t0', 'per', 'a', 'inc', 'rp', 'u1', 'u2']  # Transit parameters to fit

def calc_light_curve_model(bjd, vars):
    """
    Calculate the light curve model using the batman package.
    Parameters:
    bjd (array-like): Array of Barycentric Julian Dates (BJD) at which to calculate the light curve.
    vars (array-like): Array of parameters for the transit model in the order of light_curve_params_to_fit.
    
    Returns:
    lc_flux (array-like): The calculated light curve flux values.
    """
    # Initialize the batman parameters
    batmanparams = batman.TransitParams()
    
    # Extract parameters using the order defined in light_curve_params_to_fit
    batmanparams.t0 = vars[light_curve_params_to_fit.index('t0')]      # mid-transit time
    batmanparams.per = vars[light_curve_params_to_fit.index('per')]     # orbital period
    batmanparams.a = vars[light_curve_params_to_fit.index('a')]         # a/R*
    batmanparams.inc = vars[light_curve_params_to_fit.index('inc')]     # inclination
    batmanparams.rp = vars[light_curve_params_to_fit.index('rp')]       # Rp/R*
    
    # Fixed parameters (circular orbit)
    batmanparams.ecc = 0.0      # eccentricity
    batmanparams.w = 90.0       # longitude of periastron (degrees)
    
    # Limb darkening (quadratic)
    u1 = vars[light_curve_params_to_fit.index('u1')]
    u2 = vars[light_curve_params_to_fit.index('u2')]
    batmanparams.u = [u1, u2]
    batmanparams.limb_dark = "quadratic"
    
    # Initialize the transit model
    m = batman.TransitModel(batmanparams, np.asarray(bjd))
    
    # Calculate and return the model flux
    lc_flux = m.light_curve(batmanparams)
    
    return lc_flux

# Plotting a light curve model with reasonable initial parameters for WASP-39b
# Initial guess for t0: time of minimum flux in binned data
t0_initial = bjd_binned[np.argmin(white_binned)]

# Literature values (from discovery paper and approximations)
per_initial = 4.055294    # days
a_initial = 11.39         # a/R* (approximate)
inc_initial = 87.32       # degrees (approximate, near grazing but not)
rp_initial = 0.1457        # Rp/R* (for Saturn-like density and inflated radius)
u1_initial = 0.3          # Approximate for G-star in NIR
u2_initial = 0.1

vars_initial = [t0_initial, per_initial, a_initial, inc_initial, rp_initial, u1_initial, u2_initial]

# Generate model on binned times
lc_model = calc_light_curve_model(bjd_binned, vars_initial)

# Plot 3: Binned data + initial model
fig3 = plt.figure(figsize=(12, 6))
# Draw model on top of markers: larger linewidth, higher zorder, and disable clipping
plt.plot(bjd_binned, lc_model, color='red', lw=3, label='Initial Batman transit model', zorder=10, clip_on=False)
plt.errorbar(bjd_binned, white_binned, yerr=white_err_binned,
             fmt='o', color='black', ecolor='gray', elinewidth=1, capsize=2,
             markersize=6, markeredgecolor='black', markeredgewidth=0.5,
             alpha=0.9, label='Binned white-light data', zorder=2)
plt.xlabel('Time (BJD)')
plt.ylabel('Normalized Flux')
plt.title('WASP-39b White Light Curve – Binned Data + Initial Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(os.path.join(out_dir, '03_binned_data_with_initial_model.png'), dpi=200)
plt.close(fig3)

# ====================
# White light curve MCMC fitting setup
# ====================

# Transit parameters to fit (we add systematics here)
light_curve_params_to_fit = ['t0', 'per', 'a', 'inc', 'rp', 'u1', 'u2']  # core transit params

# Literature-informed initial values for WASP-39b (updated from discovery + JWST ERS)
# Orbital period (very precise, we'll fit loosely)
per_initial = 4.0552941      # days (from JWST ERS and recent refs)
per_lower   = per_initial - 0.01
per_upper   = per_initial + 0.01

# Scaled semi-major axis a/R*
a_initial = 11.39
a_lower   = a_initial - 0.4
a_upper   = a_initial + 0.21

# Orbital inclination (deg)
inc_initial = 87.32
inc_lower   = inc_initial - 0.17
inc_upper   = 88.0

# Planet-to-star radius ratio Rp/R*
rp_initial = 0.1457
rp_lower   = rp_initial - 0.0015
rp_upper   = rp_initial + 0.0016

# Limb darkening (quadratic, wide priors for NIR)
u1_initial = 0.3
u1_lower   = -1.0
u1_upper   =  1.0

u2_initial = 0.1
u2_lower   = -1.0
u2_upper   =  1.0

# Mid-transit time: use approximate minimum of binned data
t0_initial = bjd_binned[np.argmin(white_binned)]
t0_lower   = t0_initial - 0.1
t0_upper   = t0_initial + 0.1

# Time-mean for detrending (helps numerical stability)
t_mean = np.mean(bjd_binned)

# Initial systematics coefficients (constant + linear slope)
c0_initial = 1.0
c1_initial = 0.0

# Full initial parameter vector (transit + systematics)
initial_params = [t0_initial, per_initial, a_initial, inc_initial, rp_initial, u1_initial, u2_initial,
                  c0_initial, c1_initial]

# --------------------
# Batman transit model function (updated to accept full params)
# --------------------
def calc_light_curve_model(bjd, params):
    """
    Returns batman transit flux * linear baseline.
    params = [t0, per, a, inc, rp, u1, u2, c0, c1]
    """
    batmanparams = batman.TransitParams()
    batmanparams.t0   = params[0]
    batmanparams.per  = params[1]
    batmanparams.a    = params[2]
    batmanparams.inc  = params[3]
    batmanparams.rp   = params[4]
    batmanparams.u    = [params[5], params[6]]  # u1, u2
    batmanparams.ecc  = 0.0
    batmanparams.w    = 90.0
    batmanparams.limb_dark = "quadratic"

    m = batman.TransitModel(batmanparams, bjd)
    transit_flux = m.light_curve(batmanparams)

    # Linear systematics
    baseline = params[7] + params[8] * (bjd - t_mean)

    return transit_flux * baseline

# --------------------
# Likelihood & prior functions
# --------------------
def log_likelihood(params, times, flux, flux_err):
    model = calc_light_curve_model(times, params)
    return -0.5 * np.sum(((flux - model) / flux_err)**2)

def log_prior(params):
    t0, per, a, inc, rp, u1, u2, c0, c1 = params

    if not (t0_lower   < t0  < t0_upper):   return -np.inf
    if not (per_lower  < per < per_upper):  return -np.inf
    if not (a_lower    < a   < a_upper):    return -np.inf
    if not (inc_lower  < inc < inc_upper):  return -np.inf
    if not (rp_lower   < rp  < rp_upper):   return -np.inf
    if not (u1_lower   < u1  < u1_upper):   return -np.inf
    if not (u2_lower   < u2  < u2_upper):   return -np.inf
    if not (0.9 < c0 < 1.1):                return -np.inf  # baseline ~1
    if not (-0.01 < c1 < 0.01):             return -np.inf  # small slope

    return 0.0

def log_probability(params, times, flux, flux_err):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, times, flux, flux_err)

# --------------------
# Run emcee
# --------------------

ndim = len(initial_params)
nwalkers = 50  # more walkers than dimensions for stability

# Initial walker positions: tiny scatter around best guess
pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability,
    args=(bjd_binned, white_binned, white_err_binned)
)

# Burn-in + production run
nsteps_burn = 10000
nsteps_prod = 10000

print("Running burn-in...")
sampler.run_mcmc(pos, nsteps_burn, progress=True)

print("Running production...")
state = sampler.get_last_sample()
sampler.run_mcmc(state, nsteps_prod, progress=True)

# --------------------
# Basic results (after burn-in discard)
# --------------------
flat_samples = sampler.get_chain(discard=nsteps_burn//2, thin=15, flat=True)

# Median and 16/84 percentiles
medians = np.median(flat_samples, axis=0)
lower   = np.percentile(flat_samples, 16, axis=0)
upper   = np.percentile(flat_samples, 84, axis=0)

param_names = ['t0', 'per', 'a/R*', 'inc', 'Rp/R*', 'u1', 'u2', 'c0', 'c1']

print("\nMCMC Results (median ± (84-16)/2):")
for name, med, lo, hi in zip(param_names, medians, medians-lower, upper-medians):
    print(f"{name:8s} = {med:.6f} +{hi:.6f} -{lo:.6f}")

# Save results to file
results_file = os.path.join(out_dir, 'mcmc_results_binned.txt')
with open(results_file, 'w') as f:
    f.write("MCMC Results (median ± (84-16)/2):\n")
    for name, med, lo, hi in zip(param_names, medians, medians-lower, upper-medians):
        f.write(f"{name:8s} = {med:.6f} +{hi:.6f} -{lo:.6f}\n")
print(f"\nResults saved to {results_file}")

# Optional: Save corner plot
fig_corner = corner.corner(flat_samples, labels=param_names, show_titles=True)
fig_corner.savefig(os.path.join(out_dir, '04_corner_white_light_10k10k.png'), dpi=200)
plt.close(fig_corner)

# ====================
# Best-fit model + residuals plot
# ====================

# Extract best-fit parameters (median of the posterior)
best_fit_params = np.median(flat_samples, axis=0)

# Calculate the best-fit model (transit * baseline)
best_fit_model = calc_light_curve_model(bjd_binned, best_fit_params)

# Calculate residuals (data - model)
residuals = white_binned - best_fit_model

# Calculate reduced chi-squared for goodness-of-fit
chi2 = np.sum(residuals**2 / white_err_binned**2)
ndof = len(white_binned) - len(best_fit_params)  # degrees of freedom
reduced_chi2 = chi2 / ndof if ndof > 0 else np.nan

# Create figure with two subplots (light curve on top, residuals below)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                               gridspec_kw={'height_ratios': [3, 1]},
                               sharex=True)

# Top panel: Data + best-fit model
ax1.errorbar(bjd_binned, white_binned, yerr=white_err_binned,
             fmt='o', color='orange', ecolor='gray', alpha=0.8,
             markersize=5, markeredgecolor='black', markeredgewidth=0.5,
             label='Binned white-light data')
ax1.plot(bjd_binned, best_fit_model, color='blue', lw=3,
         label='Best-fit model (transit + linear baseline)', zorder=10, clip_on=False)
ax1.set_ylabel('Normalized Flux')
ax1.set_title('WASP-39b White Light Curve – Best-Fit Model')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add reduced chi-squared annotation
ax1.text(0.02, 0.98, fr'Reduced $\chi^2 = {reduced_chi2:.2f}$',
         transform=ax1.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Bottom panel: Residuals
ax2.errorbar(bjd_binned, residuals, yerr=white_err_binned,
             fmt='o', color='orange', ecolor='gray', markersize=5,
             markeredgecolor='black', markeredgewidth=0.5)
ax2.axhline(0, color='blue', lw=1.5, linestyle='--')
ax2.set_xlabel('Time (BJD)')
ax2.set_ylabel('Residuals')
ax2.grid(True, alpha=0.3)

# Optional: set consistent y-limits on residuals (symmetric around zero)
max_res = np.max(np.abs(residuals)) * 1.2
ax2.set_ylim(-max_res, max_res)

plt.tight_layout()

# Save the figure
fig.savefig(os.path.join(out_dir, '05_bestfit_model_and_residuals.png'), dpi=200)
plt.close(fig)

# ====================
# === UNBINNED ANALYSIS (full resolution) ===
# ====================

print("\n=== Starting unbinned (full-resolution) white light curve analysis ===\n")

# Use the already normalized unbinned data
bjd_unbinned = bjd
flux_unbinned = white_norm
flux_err_unbinned = white_err_norm

# We can keep the same initial parameters / priors as for binned
# (they are literature-based and independent of binning)
# But we recalculate t0_initial using unbinned data for better centering
t0_initial_unbinned = bjd_unbinned[np.argmin(flux_unbinned)]
t_mean_unbinned = np.mean(bjd_unbinned)

# Full initial parameter vector (same as before, just update t0 & t_mean)
initial_params_unbinned = [t0_initial_unbinned, per_initial, a_initial, inc_initial, rp_initial,
                           u1_initial, u2_initial, c0_initial, c1_initial]

# Update t0 bounds slightly around the new estimate
t0_lower = t0_initial_unbinned - 0.1
t0_upper = t0_initial_unbinned + 0.1

# ====================
# Optional: Plot initial model on unbinned data (for visual check)
# ====================
lc_model_unbinned = calc_light_curve_model(bjd_unbinned, initial_params_unbinned)

fig_init_un = plt.figure(figsize=(12, 6))
plt.plot(bjd_unbinned, lc_model_unbinned, color='red', lw=2.5,
         label='Initial model (transit + linear baseline)', zorder=10, alpha=0.9)
plt.errorbar(bjd_unbinned, flux_unbinned, yerr=flux_err_unbinned,
             fmt='.', color='darkblue', alpha=0.4, markersize=3,
             label='Unbinned normalized data')
plt.xlabel('Time (BJD)')
plt.ylabel('Normalized Flux')
plt.title('WASP-39b White Light Curve – Unbinned Data + Initial Model')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
fig_init_un.savefig(os.path.join(out_dir, '06_unbinned_initial_model.png'), dpi=200)
plt.close(fig_init_un)

# ====================
# MCMC on unbinned data
# ====================
print("Running burn-in (unbinned)...")
sampler_unb = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability,
    args=(bjd_unbinned, flux_unbinned, flux_err_unbinned)
)
sampler_unb.run_mcmc(initial_params_unbinned + 1e-4 * np.random.randn(nwalkers, ndim),
                     nsteps_burn, progress=True)

print("Running production (unbinned)...")
state_unb = sampler_unb.get_last_sample()
sampler_unb.run_mcmc(state_unb, nsteps_prod, progress=True)

# Results
flat_samples_unb = sampler_unb.get_chain(discard=nsteps_burn//2, thin=15, flat=True)

medians_unb = np.median(flat_samples_unb, axis=0)
lower_unb   = np.percentile(flat_samples_unb, 16, axis=0)
upper_unb   = np.percentile(flat_samples_unb, 84, axis=0)

print("\nMCMC Results – UNBINNED data:")
for name, med, lo, hi in zip(param_names, medians_unb, medians_unb-lower_unb, upper_unb-medians_unb):
    print(f"{name:8s} = {med:.6f} +{hi:.6f} -{lo:.6f}")

# Save results
results_file_unb = os.path.join(out_dir, 'mcmc_results_unbinned.txt')
with open(results_file_unb, 'w') as f:
    f.write("MCMC Results – UNBINNED data (median ± (84-16)/2):\n")
    for name, med, lo, hi in zip(param_names, medians_unb, medians_unb-lower_unb, upper_unb-medians_unb):
        f.write(f"{name:8s} = {med:.6f} +{hi:.6f} -{lo:.6f}\n")

# Corner plot unbinned
fig_corner_unb = corner.corner(flat_samples_unb, labels=param_names, show_titles=True)
fig_corner_unb.savefig(os.path.join(out_dir, '07_corner_white_light_unbinned.png'), dpi=200)
plt.close(fig_corner_unb)

# ====================
# Best-fit model + residuals – UNBINNED
# ====================
best_fit_params_unb = np.median(flat_samples_unb, axis=0)
best_fit_model_unb = calc_light_curve_model(bjd_unbinned, best_fit_params_unb)

residuals_unb = flux_unbinned - best_fit_model_unb

chi2_unb = np.sum(residuals_unb**2 / flux_err_unbinned**2)
ndof_unb = len(flux_unbinned) - len(best_fit_params_unb)
reduced_chi2_unb = chi2_unb / ndof_unb if ndof_unb > 0 else np.nan

fig_unb, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

# Top: data + model
ax1.errorbar(bjd_unbinned, flux_unbinned, yerr=flux_err_unbinned,
             fmt='.', color='darkblue', alpha=0.5, markersize=2,
             label='Unbinned data')
ax1.plot(bjd_unbinned, best_fit_model_unb, color='blue', lw=2.5,
         label='Best-fit model', zorder=10)
ax1.set_ylabel('Normalized Flux')
ax1.set_title('WASP-39b White Light Curve – Unbinned Best-Fit Model')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)

ax1.text(0.02, 0.98, fr'Reduced $\chi^2 = {reduced_chi2_unb:.2f}$',
         transform=ax1.transAxes, fontsize=12, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Bottom: residuals
ax2.errorbar(bjd_unbinned, residuals_unb, yerr=flux_err_unbinned,
             fmt='.', color='darkblue', alpha=0.5, markersize=2)
ax2.axhline(0, color='blue', lw=1.5, ls='--')
ax2.set_xlabel('Time (BJD)')
ax2.set_ylabel('Residuals')
ax2.grid(True, alpha=0.2)

max_res_unb = np.max(np.abs(residuals_unb)) * 1.3
ax2.set_ylim(-max_res_unb, max_res_unb)

plt.tight_layout()
fig_unb.savefig(os.path.join(out_dir, '08_bestfit_model_and_residuals_unbinned.png'), dpi=200)
plt.close(fig_unb)

print("Unbinned analysis complete. All figures and results saved in:", out_dir)

# ==========================================================
# TRANSMISSION SPECTRUM CONSTRUCTION
#  1) wavelength binning
#  2) MCMC per bin to get Rp/R*
#  3) build + save transmission spectrum PNG
# ==========================================================

def make_wavelength_bins(wavelength_um, nbins=20, wl_min=None, wl_max=None):
    """
    Create equal-width wavelength bins over [wl_min, wl_max].
    Returns a list of (wl_lo, wl_hi, idx) where idx selects channels in that bin.
    """
    wl = np.asarray(wavelength_um)

    if wl_min is None:
        wl_min = np.nanmin(wl)
    if wl_max is None:
        wl_max = np.nanmax(wl)

    edges = np.linspace(wl_min, wl_max, nbins + 1)
    bins = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        # include left edge, exclude right edge except final
        if i < nbins - 1:
            idx = np.where((wl >= lo) & (wl < hi))[0]
        else:
            idx = np.where((wl >= lo) & (wl <= hi))[0]

        if idx.size > 0:
            bins.append((lo, hi, idx))

    return bins


def make_binned_lightcurve_for_wlbin(flux_2d, err_2d, chan_idx):
    """
    Sum flux across wavelength channels in a bin.
    Propagate uncertainties in quadrature.
    """
    lc = np.sum(flux_2d[:, chan_idx], axis=1)
    lc_err = np.sqrt(np.sum(err_2d[:, chan_idx] ** 2, axis=1))
    return lc, lc_err


def normalize_by_oot(flux_1d, err_1d, oot_index):
    """
    Normalize a light curve using the median out-of-transit flux.
    """
    med = np.median(flux_1d[oot_index])
    return flux_1d / med, err_1d / med


def batman_model_fixed_except_rp(times, t0, per, a, inc, u1, u2, rp):
    """
    Batman transit model with fixed geometry + LD, only rp varies.
    """
    p = batman.TransitParams()
    p.t0 = t0
    p.per = per
    p.a = a
    p.inc = inc
    p.rp = rp
    p.ecc = 0.0
    p.w = 90.0
    p.limb_dark = "quadratic"
    p.u = [u1, u2]

    m = batman.TransitModel(p, np.asarray(times))
    return m.light_curve(p)


def calc_bin_model(times, fixed, params_free):
    """
    Model = transit(rp) * (c0 + c1*(t - t_mean))
    fixed = (t0, per, a, inc, u1, u2)
    params_free = (rp, c0, c1)
    """
    t0, per, a, inc, u1, u2 = fixed
    rp, c0, c1 = params_free

    transit = batman_model_fixed_except_rp(times, t0, per, a, inc, u1, u2, rp)
    t_mean = np.mean(times)
    baseline = c0 + c1 * (times - t_mean)
    return transit * baseline


def loglike_bin(theta, times, flux, err, fixed):
    model = calc_bin_model(times, fixed, theta)
    return -0.5 * np.sum(((flux - model) / err) ** 2)


def logprior_bin(theta, rp0, rp_width=0.02):
    """
    Flat priors:
      rp within [rp0 - rp_width, rp0 + rp_width]
      c0 within [0.9, 1.1]
      c1 within [-0.01, 0.01]
    """
    rp, c0, c1 = theta
    if not (rp0 - rp_width < rp < rp0 + rp_width):
        return -np.inf
    if not (0.9 < c0 < 1.1):
        return -np.inf
    if not (-0.01 < c1 < 0.01):
        return -np.inf
    return 0.0


def logprob_bin(theta, times, flux, err, fixed, rp0, rp_width):
    lp = logprior_bin(theta, rp0=rp0, rp_width=rp_width)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike_bin(theta, times, flux, err, fixed)


def fit_one_bin_mcmc(times, flux, err, fixed, rp0,
                     rp_width=0.02, nwalkers=32, burn=800, prod=1200, thin=10):
    """
    Fit a single wavelength-bin light curve with emcee.
    Returns posterior samples for (rp, c0, c1).
    """
    ndim = 3
    # initial guess near rp0, baseline ~1, slope ~0
    p0 = np.array([rp0, 1.0, 0.0])
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, logprob_bin,
        args=(times, flux, err, fixed, rp0, rp_width)
    )

    sampler.run_mcmc(pos, burn, progress=False)
    state = sampler.get_last_sample()
    sampler.run_mcmc(state, prod, progress=False)

    chain = sampler.get_chain(discard=int(burn * 0.5), thin=thin, flat=True)
    return chain


def construct_transmission_spectrum(
    bjd, wavelength, flux, flux_err, oot_index,
    fixed_params, rp0,
    nbins=20,
    wl_min=0.5, wl_max=5.0,
    out_png=os.path.join(out_dir, "08_transmission_spectrum.png"),
    out_txt=os.path.join(out_dir, "08_transmission_spectrum.txt"),
    mcmc_nwalkers=32, mcmc_burn=800, mcmc_prod=1200, mcmc_thin=10,
    rp_width=0.02
):
    """
    Full pipeline:
      1) bin in wavelength
      2) MCMC each binned light curve to get rp posterior
      3) build transmission spectrum depth=(rp)^2 and save PNG
    """
    # 1) wavelength bins
    bins = make_wavelength_bins(wavelength, nbins=nbins, wl_min=wl_min, wl_max=wl_max)

    wl_centers = []
    depths = []
    depth_err_lo = []
    depth_err_hi = []

    # loop bins
    for (wl_lo, wl_hi, idx) in tqdm(bins, desc="Fitting wavelength bins"):
        # build LC for this wavelength bin
        lc, lc_err = make_binned_lightcurve_for_wlbin(flux, flux_err, idx)
        lc_n, lcerr_n = normalize_by_oot(lc, lc_err, oot_index)

        # 2) MCMC to get Rp/R*
        samples = fit_one_bin_mcmc(
            bjd, lc_n, lcerr_n,
            fixed=fixed_params,
            rp0=rp0,
            rp_width=rp_width,
            nwalkers=mcmc_nwalkers,
            burn=mcmc_burn,
            prod=mcmc_prod,
            thin=mcmc_thin
        )

        rp_samp = samples[:, 0]
        depth_samp = rp_samp ** 2

        # summarize depth posterior
        d_med = np.median(depth_samp)
        d_p16 = np.percentile(depth_samp, 16)
        d_p84 = np.percentile(depth_samp, 84)

        wl_center = 0.5 * (wl_lo + wl_hi)

        wl_centers.append(wl_center)
        depths.append(d_med)
        depth_err_lo.append(d_med - d_p16)
        depth_err_hi.append(d_p84 - d_med)

    wl_centers = np.array(wl_centers)
    depths = np.array(depths)
    depth_err_lo = np.array(depth_err_lo)
    depth_err_hi = np.array(depth_err_hi)

    # 3) save spectrum table
    with open(out_txt, "w") as f:
        f.write("# wl_center_um  depth  depth_err_lo  depth_err_hi\n")
        for w, d, elo, ehi in zip(wl_centers, depths, depth_err_lo, depth_err_hi):
            f.write(f"{w:.6f} {d:.8e} {elo:.8e} {ehi:.8e}\n")

    # 3) plot + save PNG
    fig = plt.figure(figsize=(11, 6))
    plt.errorbar(
        wl_centers, depths,
        yerr=[depth_err_lo, depth_err_hi],
        fmt="o", capsize=3, markersize=5
    )
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Transit depth (Rp/R*)^2")
    plt.title("WASP-39b Transmission Spectrum (binned + MCMC depth)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"\nSaved transmission spectrum PNG: {out_png}")
    print(f"Saved transmission spectrum table: {out_txt}")

    return wl_centers, depths, depth_err_lo, depth_err_hi


# ----------------------------------------------------------
# Run transmission spectrum construction
# ----------------------------------------------------------
# Choose which white-light fit to anchor the fixed parameters from:
# - best_fit_params_unb (unbinned fit) if available
# - otherwise best_fit_params (binned fit)
try:
    anchor = best_fit_params_unb
except NameError:
    anchor = best_fit_params

# anchor indices: [t0, per, a, inc, rp, u1, u2, c0, c1]
t0_fix, per_fix, a_fix, inc_fix = anchor[0], anchor[1], anchor[2], anchor[3]
rp0 = anchor[4]
u1_fix, u2_fix = anchor[5], anchor[6]

fixed = (t0_fix, per_fix, a_fix, inc_fix, u1_fix, u2_fix)

# You can tune nbins + MCMC settings depending on runtime
wl_centers, depths, elo, ehi = construct_transmission_spectrum(
    bjd=bjd,
    wavelength=wavelength,
    flux=flux,
    flux_err=flux_err,
    oot_index=oot_index_non_binned,
    fixed_params=fixed,
    rp0=rp0,
    nbins=20,          # try 15–30 to start
    wl_min=0.5,
    wl_max=5.0,
    mcmc_nwalkers=32,
    mcmc_burn=800,
    mcmc_prod=1200,
    mcmc_thin=10,
    rp_width=0.02
)
