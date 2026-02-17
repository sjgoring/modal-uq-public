
import numpy as np

def nll_from_density_at_truth(density, y_grid, y_true):
    idx = np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)
    p = density[np.arange(len(y_true)), idx] + 1e-12
    return -np.log(p)

def crps_from_cdf_grid(cdf, y_grid, y_true):
    idx = np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)
    crps = []
    for i, k in enumerate(idx):
        left = np.trapz((cdf[i,:k] - 0.0)**2, y_grid[:k])
        right= np.trapz((cdf[i,k:] - 1.0)**2, y_grid[k:])
        crps.append(left+right)
    return np.array(crps)
