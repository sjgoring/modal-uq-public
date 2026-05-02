import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from skmdn import MixtureDensityEstimator

# Generate dataset
n_samples = 1000
X_full, _ = make_moons(n_samples=n_samples, noise=0.1)
X = X_full[:, 0].reshape(-1, 1)  # Use only the first column as input
Y = X_full[:, 1].reshape(-1, 1)  # Predict the second column

# Add some noise to Y to make the problem more suitable for MDN
Y += 0.1 * np.random.randn(n_samples, 1)

# Fit the model
mdn = MixtureDensityEstimator()
mdn.fit(X, Y)

# Predict some quantiles on the train set 
means, quantiles = mdn.predict(X, quantiles=[0.01, 0.1, 0.9, 0.99], resolution=100000)
plt.scatter(X, Y)
plt.scatter(X, quantiles[:, 0], color='orange')
plt.scatter(X, quantiles[:, 1], color='green')
plt.scatter(X, quantiles[:, 2], color='green')
plt.scatter(X, quantiles[:, 3], color='orange')
plt.scatter(X, means, color='red')
plt.show()