## Extremely basic test file, to understand where this code isn't working.
# derived from why-isnt-anything-working.py, but with a DPG where mixture variance is constant.

#Imports
import numpy as np
import scipy.integrate as integrate
import scipy.stats as spstats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skmdn import MixtureDensityEstimator

# note densities are all conditional on x unless specified

# Package settings
matplotlib.use('TkAgg')
np.random.seed(42)

# params
n_samples_max = 1000
n_samples_min = 1000
n_sizes = int(np.log10(n_samples_max/n_samples_min) + 1)
data_sizes = np.logspace(np.log10(n_samples_min), np.log10(n_samples_max), num=n_sizes, dtype=int, base=10)

min_x = -10
max_x = 10
min_y = -10
max_y = 10

y_grid_size = 1000

n_repeats = 1 #repeats per size.

# mu_1 = -5 mus depend on x.
# mu_2 = 5
# sigma_1 = 2 sigma depends on x.
# sigma_2 = sigma_1 # Homoscedastic

pi_1 = 0.6
pi_2 = 0.4

# set mixture sigma, must be larger than maximum values of sigma_2
# For this setup I reckon it needs to be bigger than sigma^2 = 50
sigma =  0.5

# mu = mu_1 * pi_1 + mu_2 * pi_2

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
    # print(xs_i)
    xs_i = np.sort(xs_i, axis=0)
    # print(xs_i)
    xs_i = np.expand_dims(xs_i, axis=1)
    moae_is = []
    for j in range(n_repeats):
        print(f"Size: {data_sizes[i]}, Repeat: {j+1}")
        y_samples = []
        y_densities = []


        # low component sigma region
        sigma_c = 0.1
        sigma_1_l = np.ones_like(xs_i) * sigma_c
        sigma_2_l = sigma_1_l # homoscedastic
        mu_1_l = np.ones_like(xs_i)
        mu_2_l = np.ones_like(xs_i) * 0

        # override sigma
        mu = mu_1_l * pi_1 + mu_2_l * pi_2
        sigma = np.sqrt(sigma_c**2 + pi_1*(mu_1_l-mu)**2 + pi_2*(mu_2_l-mu)**2)

        # high component sigma region
        mu_1_h = np.ones_like(xs_i) * 0.8
        mu_2_h = np.ones_like(xs_i) * 0.3
        mu_h = mu_1_h * pi_1 + mu_2_h * pi_2

        sigma_c = np.sqrt(sigma**2-pi_1*(mu_1_h-mu_h)**2-pi_2*(mu_2_h-mu_h)**2)
        sigma_1_h = np.ones_like(xs_i) * sigma_c
        sigma_2_h = sigma_1_h # homoscedastic
 
        # combine regions
        # simple split into 2 regions.
        # mu_1 = np.where(xs_i < min_x+1/3*(max_x-min_x) or xs_i > min_x+2/3*(max_x-min_x), mu_1_l, mu_1_h)
        # mu_1 = np.where(xs_i < 0, mu_1_l, mu_1_h) # simple example to start
        # mu_2 = np.where(xs_i < 0, mu_2_l, mu_2_h)
        # sigma_1 = np.where(xs_i < 0, sigma_1_l, sigma_1_h)
        # sigma_2 = np.where(xs_i < 0, sigma_2_l, sigma_2_h)
        # # print(len(mu_1), len(xs_i))
        # print(xs_i, mu_1, mu_2, sigma_1, sigma_2)
        # quit()

        # split x into thirds: first & last third -> low sigma, middle third -> high sigma
        x_th1 = min_x + (max_x - min_x) / 3.0
        x_th2 = min_x + 2.0 * (max_x - min_x) / 3.0

        low_mask = np.logical_or(xs_i < x_th1, xs_i > x_th2)

        mu_1  = np.where(low_mask, mu_1_l,  mu_1_h)
        mu_2  = np.where(low_mask, mu_2_l,  mu_2_h)
        sigma_1 = np.where(low_mask, sigma_1_l, sigma_1_h)
        sigma_2 = np.where(low_mask, sigma_2_l, sigma_2_h)

        # Set mus for each subsample.
        # mu_1 = -4.5/400*xs_i**2 + 2.5
        # mu_2 = -1*mu_1
        # mu = mu_1 * pi_1 + mu_2 * pi_2

        # Set sigmas for each subsample.
        # sigma_2 = 2*abs(xs_i - (min_x+max_x)/2) /abs(max_x -(min_x+max_x)/2) +0.1 # Avoid zero, and ensure positive. [0.1, 2.1]
        # sigma_2 = -1*sigma_2+2.2 # flipping to get near 0 at ends, and 2.1 in middle.
        # sigma_2 = np.ones_like(xs_i) # fix second component sigma to be constant.

        # sigma_1 = np.sqrt((-sigma_2**2*pi_2-pi_1*(mu_1-mu)**2-pi_2*(mu_2-mu)**2+sigma**2)/pi_1) # Set sigma_1 to ensure constant variance across x, given sigma_2 and the mixture weights and means.

        # plt.plot(xs_i, sigma_2, 'o', label='sigma_2')  
        # plt.show()  
        # quit()
        for x in xs_i:
            # print(sigma_1)
            idx = np.where(xs_i==(x))[0][0]
            # print(idx, x, mu_1[idx], mu_2[idx], sigma_1[idx], sigma_2[idx])
            y_dens_raw = spstats.norm.pdf(y_grid, loc=mu_1[idx], scale=sigma_1[idx]) * pi_1 + spstats.norm.pdf(y_grid, loc=mu_2[idx], scale=sigma_2[idx]) * pi_2
            # if (x>0):
            #     plt.plot(y_grid, y_dens_raw, label=f"x: {x[0]}, sigma_1: {sigma_1[idx]}, sigma_2: {sigma_2[idx]}")
            #     plt.show()
            #     quit()
            # # print(y_dens_raw, sum(y_dens_raw), integrate.trapezoid(y_dens_raw, y_grid))
            y_probs = y_dens_raw / np.sum(y_dens_raw)  # Normalize to get probabilities
            y_dens = y_dens_raw / integrate.trapezoid(y_dens_raw, y_grid)  # Normalize to get a proper density
            y_densities.append(y_dens)

            # if (x>0):
            #     plt.plot(y_grid, y_probs, label=f"x: {x[0]}, sigma_1: {sigma_1[idx]}, sigma_2: {sigma_2[idx]}")
            #     plt.show()
            #     plt.plot(y_grid, y_dens, label=f"x: {x[0]}, sigma_1: {sigma_1[idx]}, sigma_2: {sigma_2[idx]}")
            #     plt.show() 
            #     quit()

            # print(y_dens, sum(y_dens), integrate.trapezoid(y_dens, y_grid))
            # quit()
            ## Draws from y grid according to the density
            # if (x<0):
                # testing only
            y_samples.append(np.random.choice(y_grid, size=1, p=y_probs))
            # if (x>0):
            #     print(y_samples[-1])
            # print(f"x: {x[0]}, sigma_1: {sigma_1[idx]}, sigma_2: {sigma_2[idx]}, std of y_samples: {np.std(y_samples)}")
            # print(sigma_1[idx],sigma_2[idx],np.std(y_samples), )
        # Model learning
        model = MixtureDensityEstimator(n_gaussians=2, weight_decay=1e-4, lr=1e-5, hidden_dim=50, epochs=10000)
        # model = MixtureDensityEstimator(n_gaussians=6, weight_decay=0, hidden_dim=50, epochs=250)

        # Fitting on all data for now to test.
        y_samples = np.array(y_samples)
        # plt.hist(y_samples, bins=50, density=True, alpha=0.5, label='Sampled y distribution')
        # plt.show()
        # quit()
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
fig1, ax1 = plt.subplots()
ax1.plot(xs, y_samples, 'o', alpha=0.5, label='Samples')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Generated data')
plt.show()


# Model performance
# Plot MOAE (mean + spread) across data sizes
try:
    moaes_arr = np.asarray(moaes, dtype=float)
    moae_mean = np.nanmean(moaes_arr, axis=1)
    moae_stderr = np.nanstd(moaes_arr, axis=1) / np.sqrt(n_repeats)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(data_sizes, moae_mean, marker='o', linestyle='-', color='C0', label='Mean MOAE')
    ax2.fill_between(data_sizes, moae_mean - moae_stderr, moae_mean + moae_stderr, color='C0', alpha=0.25, label='±1 standard error')
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

# Conditional density — interactive slider to choose x for plotting
# Use the last fitted model's results: `xs_i`, `y_densities`, `yhat_densities`.
try:
    xs_plot = np.asarray(xs_i).flatten()
    # stack true densities (list of arrays) into shape (n_points, y_grid_size)
    ytrue_arr = np.vstack(y_densities)
    yhat_arr = np.asarray(yhat_densities)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    # initial index 0
    idx0 = 0
    true_line, = ax.plot(y_grid, ytrue_arr[idx0], label='True Conditional Density', color='blue')
    pred_line, = ax.plot(y_grid, yhat_arr[idx0], label='MDN Predicted Density', color='red')
    ax.set_xlabel('y')
    ax.set_ylabel('Density')
    ax.set_title(f'Conditional Density at x = {xs_plot[idx0]:.3f}')
    ax.legend()

    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)
    sld = Slider(ax_x, 'x', float(xs_plot.min()), float(xs_plot.max()), valinit=float(xs_plot[idx0]))

    def update(val):
        xv = val
        # find nearest x in the subsample
        idx = int(np.argmin(np.abs(xs_plot - xv)))
        true_line.set_ydata(ytrue_arr[idx])
        pred_line.set_ydata(yhat_arr[idx])
        ax.set_title(f'Conditional Density at x = {xs_plot[idx]:.3f}')
        fig.canvas.draw_idle()

    sld.on_changed(update)
    plt.show()
except Exception:
    import traceback
    traceback.print_exc()
