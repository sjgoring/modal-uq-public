
import numpy as np
import scipy.integrate as integrate

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
    # check for missing kwargs
    if true_dens is None or est_dens is None or y_grid is None:
        raise ValueError("true_dens, est_dens, and y_grid must be provided in kwargs_dict")

    # Notice we assum 1D x, so:
    if y_mode_pred.ndim > 1:
        raise ValueError("y_mode_pred suggests x is not 1D")

    print("Debug - likelihood_ratio_measure input shapes:")
    print("y_true", y_true.shape, "y_mode_pred", y_mode_pred.shape, "true_dens", true_dens.shape, "est_dens", est_dens.shape, "y_grid", y_grid.shape)

    # Debug: Checking y_true and y_mode_pred against argmax of true_dens and est_dens respectively.
    print("Debug - Checking y_true and y_mode_pred against argmax of true_dens and est_dens respectively.")
    print("true_dens correct", np.equal(y_true, y_grid[true_dens.argmax(axis=1)]))
    print("est_dens correct", np.equal(y_mode_pred, y_grid[est_dens.argmax(axis=1)]))
    print("Todo: Complete this debugging.")
    # quit()
    # normalise incoming densities as empirical probabilities
    true_dens = true_dens / (true_dens.sum(axis=1)[:, None] + 1e-12)
    est_dens = est_dens / (est_dens.sum(axis=1)[:, None] + 1e-12)

    if reference_dist == "true":
        # relative likelihood of y^ against y* under p*. That is, p*(y^) / p*(y*).
        p_ystar = true_dens[np.arange(len(y_true)), np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)]
        p_yhat = true_dens[np.arange(len(y_true)), np.abs(y_grid[None,:] - y_mode_pred[:,None]).argmin(axis=1)]
        print("Debug - likelihood_ratio_measure p_ystar < p_hat?", np.mean(p_ystar < p_yhat))
        return float(np.mean(p_yhat / (p_ystar + 1e-12)))
    elif reference_dist == "est":
        # relative likelihood of y* against y^ under p^. That is, p^(y*) / p^(y^).
        p_ystar = est_dens[np.arange(len(y_true)), np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)]
        p_yhat = est_dens[np.arange(len(y_true)), np.abs(y_grid[None,:] - y_mode_pred[:,None]).argmin(axis=1)]
        return float(np.mean(p_ystar / (p_yhat + 1e-12)))
    else:
        raise ValueError("reference_dist must be either 'true' or 'est'")

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

    print("Debug - coverage_measure input shapes:")
    print("y_true", y_true.shape, "y_mode_pred", y_mode_pred.shape, "true_dens", true_dens.shape, "est_dens", est_dens.shape, "y_grid", y_grid.shape)

    # normalise incoming densities as empirical densities (not probabilities), via trapezoidal rule
    true_dens = true_dens / (true_dens.sum(axis=1)[:, None] + 1e-12)
    est_dens = est_dens / (est_dens.sum(axis=1)[:, None] + 1e-12)

    if reference_dist == "est":
        # take the integral of all density values where the estimated density of y is greater than the estimated density of y*.
        # first, define an indicator function for each sample and grid point, which is 1 if p^(y) > p^(y*), and 0 otherwise
        indicator = (est_dens > est_dens[np.arange(len(y_true)), np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)][:,None]).astype(float)
        # then, for each sample, take the integral of the estimated over the grid, weighted by the indicator function, to get the coverage measure
        coverage = integrate.trapezoid(est_dens * indicator, y_grid, axis=1)
    elif reference_dist == "true":
        # as above but with p*, and y^.
        indicator = (true_dens > true_dens[np.arange(len(y_true)), np.abs(y_grid[None,:] - y_mode_pred[:,None]).argmin(axis=1)][:,None]).astype(float)
        coverage = integrate.trapezoid(true_dens * indicator, y_grid, axis=1)
    else:
        raise ValueError("reference_dist must be either 'true' or 'est'")
    return float(np.mean(coverage))




if __name__ == "__main__":
    # Example usage
    y_true = np.array([3.0, 2.0, 2.0])
    y_mode_pred = np.array([3.0, 2.0, 1.0])
    true_dens = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])
    est_dens = np.array([[0.2, 0.3, 0.5], [0.3, 0.4, 0.3], [0.45, 0.35, 0.2]])
    y_grid = np.array([1.0, 2.0, 3.0])

    print("Check - y_true is argmax of true_dens:", np.all(y_true == y_grid[true_dens.argmax(axis=1)]))
    print("Check - y_mode_pred is argmax of est_dens:", np.all(y_mode_pred == y_grid[est_dens.argmax(axis=1)]))

    print("Modal Absolute Error:", modal_absolute_error(y_true, y_mode_pred))
    print("Modal Squared Error:", modal_squared_error(y_true, y_mode_pred))
    print("Likelihood Ratio Measure (true):", likelihood_ratio_measure(y_true, y_mode_pred, true_dens, est_dens, y_grid, reference_dist="true"))
    print("Likelihood Ratio Measure (est):", likelihood_ratio_measure(y_true, y_mode_pred, true_dens, est_dens, y_grid, reference_dist="est"))
    print("Modal Coverage Measure (true):", modal_coverage_measure(y_true, y_mode_pred, true_dens, est_dens, y_grid, reference_dist="true"))
    print("Modal Coverage Measure (est):", modal_coverage_measure(y_true, y_mode_pred, true_dens, est_dens, y_grid, reference_dist="est"))