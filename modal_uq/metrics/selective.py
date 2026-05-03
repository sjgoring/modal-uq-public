
import numpy as np
import scipy.integrate as integrate


def _slice_density_by_keep(dens, keep):
    if dens is None:
        return None
    dens = np.asarray(dens)
    if dens.ndim == 2:
        return dens[keep]
    if dens.ndim == 3:
        return dens[:, keep]
    raise ValueError(f"Unsupported density shape {dens.shape}; expected [N, G] or [M, N, G].")

def risk_coverage(y_true, y_pred, uncertainty, risk_fn, risk_fn_kwargs, steps=20, true_dens=None, est_dens=None, y_grid=None):
    risk_fn_kwargs.update({'true_dens': true_dens, 'est_dens': est_dens, 'y_grid': y_grid})
    # Pass args not needed by all risk functions, but which may be needed by some, via risk_fn_kwargs dict, which is unpacked in the risk function call. This allows us to use a common risk_coverage function for all risk functions, even those that require additional arguments.
    order = np.argsort(-uncertainty)
    risks, coverages = [], []
    n = len(y_true)
    for k in range(steps+1):
        keep = order[k*n//(steps+1):]
        if len(keep) == 0:
            risks.append(np.nan); coverages.append(0.0)
            # update true_dens, and est_dens in risk_fn_kwargs to be NaN or empty arrays to avoid potential issues in risk function calculations when keep is empty
            risk_fn_kwargs.update({'true_dens': np.array([]), 'est_dens': np.array([])})
            continue
        # update true dens and est_dens in risk_fn_kwargs to only include the samples in keep, to avoid potential issues in risk function calculations when keep is a subset of the data
        if true_dens is not None:
            risk_fn_kwargs.update({'true_dens': _slice_density_by_keep(true_dens, keep)})
        if est_dens is not None:
            risk_fn_kwargs.update({'est_dens': _slice_density_by_keep(est_dens, keep)})
        risks.append(float(risk_fn(y_true[keep], y_pred[keep], risk_fn_kwargs)))
        coverages.append(len(keep)/n)
    return np.array(coverages), np.array(risks)

def aurc(coverages, risks):
    # Order coverage, risk pairs by increasing coverage
    order = np.argsort(coverages)
    coverages = coverages[order]
    risks = risks[order]
    return float(integrate.trapezoid(risks, coverages))
