## Extremely basic test file, to understand where this code isn't working.

#Imports
import numpy as np
import scipy.integrate as integrate
import scipy.stats as spstats
import matplotlib
import matplotlib.pyplot as plt
from skmdn import MixtureDensityEstimator

# note densities are all conditional on x unless specified

# Package settings
matplotlib.use('TkAgg')
np.random.seed(42)

# params
n_samples_max = 100000
n_samples_min = 10
n_sizes = 5
data_sizes = np.logspace(np.log10(n_samples_min), np.log10(n_samples_max), num=n_sizes, dtype=int, base=10)

min_x = -20
max_x = 20
min_y = -10
max_y = 10

y_grid_size = 1000

n_repeats = 5 #repeats per size.

mu_1 = -5
mu_2 = 5
sigma_1 = 2
sigma_2 = sigma_1 # Homoscedastic
pi_1 = 0.7
pi_2 = 0.3

#test_x_index = 500 if manually setting, otherwise chosen randomly.

# Data generation
xs = np.linspace(min_x, max_x, n_samples_max)
y_grid = np.linspace(min_y, max_y, y_grid_size) # must be linspace as MDN outputs a linear grid of y values for density estimation


# release numpy seed for sampling, to allow for different samples across runs.
# np.random.seed(None)

moaes = []

for i in range(n_sizes):
    # data to be consitent across repeats
    print(f"Subsampling data for size: {data_sizes[i]}")
    xs_i = np.random.choice(xs, size=data_sizes[i], replace=False)
    xs_i = np.expand_dims(xs_i, axis=1)
    moae_is = []
    for j in range(n_repeats):
        print(f"Size: {data_sizes[i]}, Repeat: {j+1}")
        y_samples = []
        y_densities = []

        for x in xs_i:
            y_dens_raw = spstats.norm.pdf(y_grid, loc=mu_1, scale=sigma_1) * pi_1 + spstats.norm.pdf(y_grid, loc=mu_2, scale=sigma_2) * pi_2
            # print(y_dens_raw, sum(y_dens_raw), integrate.trapezoid(y_dens_raw, y_grid))
            y_probs = y_dens_raw / np.sum(y_dens_raw)  # Normalize to get probabilities
            y_dens = y_dens_raw / integrate.trapezoid(y_dens_raw, y_grid)  # Normalize to get a proper density
            y_densities.append(y_dens)
            # print(y_dens, sum(y_dens), integrate.trapezoid(y_dens, y_grid))
            # quit()
            ## Draws from y grid according to the density
            y_samples.append(np.random.choice(y_grid, size=1, p=y_probs))

        # Model learning
        model = MixtureDensityEstimator(n_gaussians=6, weight_decay=1e-4, lr=1e-4, hidden_dim=50, epochs=500)
        # model = MixtureDensityEstimator(n_gaussians=6, weight_decay=0, hidden_dim=50, epochs=250)

        # Fitting on all data for now to test.
        y_samples = np.array(y_samples)

        model.fit(xs_i, y_samples)

        yhat_densities, _ = model.pdf(xs_i, resolution=len(y_grid), y_min=min_y, y_max=max_y)
        # print(np.shape(yhat_cond_dens))
        # print(np.shape(yhat_cond_dens[test_x_index]))
        # print("quitting")
        # quit()
        # Plotting

        # Model performance
        map_hats = y_grid[np.argmax(yhat_densities,axis=1) ]
        map_true = y_grid[np.argmax(y_densities, axis=1)]

        moae_is.append(np.mean(np.abs(map_hats - map_true)))
    moaes.append(moae_is)


# print(moaes)

# PLOTTING

# Data
plt.plot(xs, y_samples, 'o', alpha=0.5, label='Samples')
plt.show()


# Model performance
# Plot MOAE (mean + spread) across data sizes
try:
    moaes_arr = np.asarray(moaes, dtype=float)
    moae_mean = np.nanmean(moaes_arr, axis=1)
    moae_std = np.nanstd(moaes_arr, axis=1)

    fig2, ax2 = plt.subplots()
    ax2.plot(data_sizes, moae_mean, marker='o', linestyle='-', color='C0', label='Mean MOAE')
    ax2.fill_between(data_sizes, moae_mean - moae_std, moae_mean + moae_std, color='C0', alpha=0.25, label='±1 std')
    ax2.set_xscale('log')
    ax2.set_xlabel('Training size')
    ax2.set_ylabel('Mode Absolute Error (MOAE)')
    ax2.set_title('MOAE vs Training Size')
    ax2.grid(True, which='both', ls='--', lw=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.show()
except Exception:
    # If anything goes wrong plotting MOAEs, continue without crashing
    import traceback
    traceback.print_exc()

# Conditional density
test_x_index = np.random.randint(0, n_samples_max) # Choose a random index for testing

# # Checks
# print(sum(yhat_densities[test_x_index]), integrate.trapezoid(yhat_densities[test_x_index], y_grid))
# print(sum(y_densities[test_x_index]), integrate.trapezoid(y_densities[test_x_index], y_grid))

# Produces plots for last model
fig, ax = plt.subplots()
ax.plot(y_grid, y_densities[test_x_index], label='True Conditional Density', color='blue')
ax.plot(y_grid, yhat_densities[test_x_index], label='MDN Predicted Density', color='red')
plt.show()
