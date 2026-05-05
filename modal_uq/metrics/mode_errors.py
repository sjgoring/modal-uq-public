
import numpy as np
import scipy.integrate as integrate


def _normalize_density(dens, y_grid):
    area = integrate.trapezoid(dens, y_grid, axis=1)
    return dens / (area[:, None] + 1e-12)


def _as_member_stack(dens):
    dens = np.asarray(dens)
    if dens.ndim == 2:
        return dens[None, ...]
    if dens.ndim == 3:
        return dens
    raise ValueError(f"Unsupported density shape {dens.shape}; expected [N, G] or [M, N, G].")


def _collapse_est_dens_for_lr(est_dens):
    """Collapse a BMA/ensemble density stack [M, N, G] to the canonical ensemble mean [N, G].

    LR expects y_mode_pred to correspond to the *aggregate* predictive density (as used by
    selective.py). To preserve the strict LR contract while avoiding member-wise mismatch,
    we canonicalize est_dens to the ensemble mean before any grid/argmax validation.
    """
    est = np.asarray(est_dens)
    if est.ndim == 3:
        return est.mean(axis=0)
    return est


def _score_likelihood_ratio_single(y_true, y_mode_pred, true_dens, est_dens, y_grid, reference_dist):
    # true_dens = _normalize_density(true_dens, y_grid)
    # est_dens = _normalize_density(est_dens, y_grid)

    #check, are y_true and y_mode_pred actually on the grid? If not, we are doing a nearest neighbor lookup, which is fine, but we should be aware of it. If they are not on the grid, print a warning.
    if not np.all(np.isin(y_true, y_grid)):
        raise ValueError("Warning: y_true contains values that are not on the y_grid. Nearest neighbor lookup will be used for likelihood ratio measure.")
    if not np.all(np.isin(y_mode_pred, y_grid)):
        raise ValueError("Warning: y_mode_pred contains values that are not on the y_grid. Nearest neighbor lookup will be used for likelihood ratio measure.")

    #check, are y_true and y_mode_pred actually the argmax of true_dens and est_dens respectively? If not, print a warning, because this measure is really designed for the case where y_true is the true mode and y_mode_pred is the predicted mode, so we expect them to be the argmax of their respective densities. If they are not, we can still compute the measure, but it may be less interpretable.
    if not np.all(y_true == y_grid[true_dens.argmax(axis=1)]):
        raise ValueError("Warning: y_true is not the argmax of true_dens. This may make the likelihood ratio measure less interpretable.")
    if not np.all(y_mode_pred == y_grid[est_dens.argmax(axis=1)]):
        raise ValueError("Warning: y_mode_pred is not the argmax of est_dens. This may make the likelihood ratio measure less interpretable.")

    if reference_dist == "true":
        p_ystar = true_dens[np.arange(len(y_true)), np.abs(y_grid[None, :] - y_true[:, None]).argmin(axis=1)]
        p_yhat = true_dens[np.arange(len(y_true)), np.abs(y_grid[None, :] - y_mode_pred[:, None]).argmin(axis=1)]
        return float(np.mean(p_ystar / (p_yhat + 1e-12))) # ensures measure  bounded below by one.
    elif reference_dist == "est":
        p_ystar = est_dens[np.arange(len(y_true)), np.abs(y_grid[None, :] - y_true[:, None]).argmin(axis=1)]
        p_yhat = est_dens[np.arange(len(y_true)), np.abs(y_grid[None, :] - y_mode_pred[:, None]).argmin(axis=1)]
        return float(np.mean(p_yhat / (p_ystar + 1e-12))) # ensures measure  bounded below by one.
    else:    
        # print(reference_dist)
        raise ValueError("reference_dist must be either 'true' or 'est'")

def _score_modal_coverage_single(y_true, y_mode_pred, true_dens, est_dens, y_grid, reference_dist):
    true_dens = _normalize_density(true_dens, y_grid)
    est_dens = _normalize_density(est_dens, y_grid)

    if reference_dist == "est":
        indicator = (est_dens > est_dens[np.arange(len(y_true)), np.abs(y_grid[None, :] - y_true[:, None]).argmin(axis=1)][:, None]).astype(float)
        coverage = integrate.trapezoid(est_dens * indicator, y_grid, axis=1)
    elif reference_dist == "true":
        indicator = (true_dens > true_dens[np.arange(len(y_true)), np.abs(y_grid[None, :] - y_mode_pred[:, None]).argmin(axis=1)][:, None]).astype(float)
        coverage = integrate.trapezoid(true_dens * indicator, y_grid, axis=1)
    else:
        raise ValueError("reference_dist must be either 'true' or 'est'")
    return float(np.mean(coverage))

def modal_absolute_error(y_true, y_mode_pred, kwargs_dict):
    return float(np.mean(np.abs(y_true - y_mode_pred)))

def modal_squared_error(y_true, y_mode_pred, kwargs_dict):
    return float(np.mean((y_true - y_mode_pred)**2))

def likelihood_ratio_measure(y_true, y_mode_pred, kwargs_dict):
    # Shapes: samples are over x, grid points are over y
    # y_true: (n_samples,) y*
    # y_mode_pred: (n_samples,) y^
    # true_dens: (n_samples, n_grid_points) p*(y)
    # est_dens: (n_samples, n_grid_points) p^(y)
    # y_grid: (n_grid_points,)

    # print("Debug - likelihood_ratio_measure:")
    # print("kwargs_dict", kwargs_dict)
    # extract from kwargs_dict
    true_dens = kwargs_dict.get('true_dens')
    est_dens = kwargs_dict.get('est_dens')
    y_grid =  kwargs_dict.get('y_grid')
    reference_dist= kwargs_dict.get('reference_dist')

    # LR-only: collapse BMA/member stack estimated densities [M, N, G] to canonical ensemble mean [N, G]
    # before any validation checks to align with ensemble-level y_mode_pred.
    est_dens = _collapse_est_dens_for_lr(est_dens)
    # check for missing kwargs
    if true_dens is None or est_dens is None or y_grid is None:
        raise ValueError("true_dens, est_dens, and y_grid must be provided in kwargs_dict")

    # Notice we assum 1D x, so:
    if y_mode_pred.ndim > 1:
        raise ValueError("y_mode_pred suggests x is not 1D")
    true_stack = _as_member_stack(true_dens)
    est_stack = _as_member_stack(est_dens)
    n_members = max(true_stack.shape[0], est_stack.shape[0])

    if true_stack.shape[0] not in {1, n_members}:
        raise ValueError("true_dens member axis must be 1 or match est_dens.")
    if est_stack.shape[0] not in {1, n_members}:
        raise ValueError("est_dens member axis must be 1 or match true_dens.")

    if true_stack.shape[0] == 1 and n_members > 1:
        true_stack = np.repeat(true_stack, n_members, axis=0)
    if est_stack.shape[0] == 1 and n_members > 1:
        est_stack = np.repeat(est_stack, n_members, axis=0)

    scores = [
        _score_likelihood_ratio_single(y_true, y_mode_pred, true_stack[m], est_stack[m], y_grid, reference_dist)
        for m in range(n_members)
    ]
    return float(np.mean(scores))

def modal_coverage_measure(y_true, y_mode_pred, kwargs_dict):
    # Shapes: samples are over x, grid points are over y
    # y_true: (n_samples,) y*
    # y_mode_pred: (n_samples,) y^
    # true_dens: (n_samples, n_grid_points) p*(y)
    # est_dens: (n_samples, n_grid_points) p^(y)
    # y_grid: (n_grid_points,)

    # extract from kwargs_dict
    true_dens = kwargs_dict.get('true_dens')
    est_dens = kwargs_dict.get('est_dens')
    y_grid =  kwargs_dict.get('y_grid')
    reference_dist= kwargs_dict.get('reference_dist', 'true')
    # check for missing kwargs
    if true_dens is None or est_dens is None or y_grid is None:
        raise ValueError("true_dens, est_dens, and y_grid must be provided in kwargs_dict")

    # Notice we assum 1D x, so:
    if y_mode_pred.ndim > 1:
        raise ValueError("y_mode_pred suggests x is not 1D")
    true_stack = _as_member_stack(true_dens)
    est_stack = _as_member_stack(est_dens)
    n_members = max(true_stack.shape[0], est_stack.shape[0])

    if true_stack.shape[0] not in {1, n_members}:
        raise ValueError("true_dens member axis must be 1 or match est_dens.")
    if est_stack.shape[0] not in {1, n_members}:
        raise ValueError("est_dens member axis must be 1 or match true_dens.")

    if true_stack.shape[0] == 1 and n_members > 1:
        true_stack = np.repeat(true_stack, n_members, axis=0)
    if est_stack.shape[0] == 1 and n_members > 1:
        est_stack = np.repeat(est_stack, n_members, axis=0)

    scores = [
        _score_modal_coverage_single(y_true, y_mode_pred, true_stack[m], est_stack[m], y_grid, reference_dist)
        for m in range(n_members)
    ]
    return float(np.mean(scores))




if __name__ == "__main__":
    # Example usage
    y_true = np.array([3.0, 2.0, 2.0]) 
    y_mode_pred = np.array([3.0, 2.0, 1.0])
    true_dens = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])
    est_dens = np.array([[0.2, 0.3, 0.5], [0.3, 0.4, 0.3], [0.45, 0.35, 0.2]])
    y_grid = np.array([1.0, 2.0, 3.0])

    #ad extra dim for y_true and y_mode_pred to simulate multiple members (e.g. from an ensemble)
    np.expand_dims(y_true, axis=0)
    np.expand_dims(y_mode_pred, axis=0) 

    # LR True - should be p*(y*) / p*(y^) = [0.7/,0.7,0.5/0.5,0.4/0.3] = [1.0, 1.0, 1.333]

    print("Check - y_true is argmax of true_dens:", np.all(y_true == y_grid[true_dens.argmax(axis=1)]))
    print("Check - y_mode_pred is argmax of est_dens:", np.all(y_mode_pred == y_grid[est_dens.argmax(axis=1)]))

     # shape should be (n_members,) and values should be around [1.0, 1.0, 1.333]
     # returned average should be around 1.11

    lr_true = likelihood_ratio_measure(y_true, y_mode_pred, kwargs_dict={'true_dens': true_dens,'est_dens': est_dens,'y_grid': y_grid, 'reference_dist':"true"})
    # print(lr_true)
    # quit()
    
    if not np.isclose(lr_true, 1.11): # allow for some numerical imprecision, and the fact that we are averaging over three samples, two of which have LR of 1.0 and one has LR of 1.333, so the average should be around 1.11 but could be higher if the 1.333 sample has more influence due to numerical imprecision.
        raise ValueError(f"LR True should be around 1.11, got {lr_true}")

    print("Modal Absolute Error:", modal_absolute_error(y_true, y_mode_pred, kwargs_dict={}))
    print("Modal Squared Error:", modal_squared_error(y_true, y_mode_pred, kwargs_dict={}))
    print("Likelihood Ratio Measure (true):", likelihood_ratio_measure(y_true, y_mode_pred, kwargs_dict={'true_dens': true_dens,'est_dens': est_dens,'y_grid': y_grid, 'reference_dist':"true"}))
    print("Likelihood Ratio Measure (est):", likelihood_ratio_measure(y_true, y_mode_pred, kwargs_dict={'true_dens': true_dens,'est_dens': est_dens,'y_grid': y_grid, 'reference_dist':"est"}))
    print("Modal Coverage Measure (true):", modal_coverage_measure(y_true, y_mode_pred, kwargs_dict={'true_dens': true_dens,'est_dens': est_dens,'y_grid': y_grid, 'reference_dist':"true"}))
    print("Modal Coverage Measure (est):", modal_coverage_measure(y_true, y_mode_pred, kwargs_dict={'true_dens': true_dens,'est_dens': est_dens,'y_grid': y_grid, 'reference_dist':"est"}))