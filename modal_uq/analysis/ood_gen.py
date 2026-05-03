import numpy as np

def _ensure_rng(rng):
    return np.random.default_rng(rng)

def shift_features(X, n_samples, shift_scale=3.0, axis=None, rng=None):
    """Generate OOD samples by shifting features away from training support.

    Parameters
    - X: array-like, shape (N, d)
    - n_samples: int, number of OOD samples to generate
    - shift_scale: float, multiply feature-wise std by this factor to shift
    - axis: None or int/sequence, which features to shift (None => all)
    - rng: seed or Generator

    Returns
    - X_ood: ndarray shape (n_samples, d)
    """
    rng = _ensure_rng(rng)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    N, d = X.shape
    idx = rng.integers(0, N, size=n_samples)
    base = X[idx].astype(float)

    std = X.std(axis=0)
    if axis is None:
        shift = std * float(shift_scale)
    else:
        shift = np.zeros(d)
        if np.isscalar(axis):
            shift[int(axis)] = std[int(axis)] * float(shift_scale)
        else:
            for a in axis:
                shift[int(a)] = std[int(a)] * float(shift_scale)

    X_ood = base + shift
    return X_ood


def add_noise(X, n_samples, noise_scale=1.0, rng=None):
    """Generate OOD samples by adding Gaussian noise scaled by training std.
    """
    rng = _ensure_rng(rng)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    N, d = X.shape
    idx = rng.integers(0, N, size=n_samples)
    base = X[idx].astype(float)
    std = X.std(axis=0)
    noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, d)) * (std * float(noise_scale))
    return base + noise


def extrapolate(X, n_samples, factor=1.5, rng=None):
    """Generate OOD by extrapolating beyond empirical min/max.

    X_ood = mean + factor * (sample - mean)
    """
    rng = _ensure_rng(rng)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    N, d = X.shape
    idx = rng.integers(0, N, size=n_samples)
    base = X[idx].astype(float)
    mu = X.mean(axis=0)
    return mu + float(factor) * (base - mu)


def mix_region(X, n_samples, fraction=0.2, rng=None):
    """Generate OOD by mixing features from two examples and extrapolating.

    Creates samples of the form a + alpha*(b-a) with alpha sampled >1 to go outside convex hull.
    """
    rng = _ensure_rng(rng)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    N, d = X.shape
    idx1 = rng.integers(0, N, size=n_samples)
    idx2 = rng.integers(0, N, size=n_samples)
    a = X[idx1].astype(float)
    b = X[idx2].astype(float)
    alpha = rng.uniform(1.1, 2.0, size=(n_samples, 1))
    return a + alpha * (b - a)
