import sys
import os
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from modal_uq.models.mdn import MixtureDensityModel
from modal_uq.datasets.synthetic_conditional import SyntheticMultiModalConditionalDataset
from modal_uq.datasets.moons_synthetic import MoonsSyntheticDataset
from modal_uq.datasets.synthetic_constant_var import SyntheticConstantVarDataset

# Generate synthetic data
# n_samples = int(1000/.8)
n_samples = 1000
n_modes = 2
n_features = 1
mode_locs = [[0,0], [5,5]]
mode_scales = [0.5, 0.5]

## Synthetic multi-modal conditional dataset with 2 modes, 1D input, and 1D output
# dataset = SyntheticMultiModalConditionalDataset(
#     n_samples=n_samples,
#     n_modes=n_modes,
#     n_features=n_features,
#     mode_locs=mode_locs,
#     mode_scales=mode_scales,
#     component_type='gaussian',
#     seed_master=42
# )

dataset= SyntheticConstantVarDataset(
    # Use defaults.
)


# X, y, global_mode, mode_ids = dataset.get_data()
X, y, _,_,_ = dataset.get_data() #for Synth_const.

## MDN santiy check: moons
# dataset = MoonsSyntheticDataset(n_samples=n_samples, noise=0.1, random_state=42, return_1d=True, target="y")
# X, y = dataset.sample()
# to revert to Sytbolic regression dataset, use: comment out the two lines above.

#X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

# For checking MDN fitting only
X_train, y_train = X, y
X_test, y_test = X, y

# Possible solution to tuple index issue in MDN fit.
# print(y_train.shape)
y_train = np.expand_dims(y_train, axis=1)
# print(y_train.shape)
# exit()

# Train GP model
# model = MixtureDensityModel(n_gaussians=2, weight_decay=0, hidden_dim=250, epochs=500)
# model = MixtureDensityModel(hidden_dim = 50,
#     n_gaussians = 20,
#     epochs = 250,
#     lr = 0.01,
#     weight_decay = 0)
# model = MixtureDensityModel(hidden_dim = 100,
#     n_gaussians = 6,
#     epochs = 5000,
#     lr = 0.001,
#     weight_decay = 0)

# Same params as in basic-test-constant-var-diff-conc.py for consistency.
model = MixtureDensityModel(hidden_dim = 50,
    n_gaussians = 2,
    epochs = 10000,
    lr = 1e-5,
    weight_decay = 1e-4)


t1 = os.times()
print("Beginning MDN fit at {} with {} samples".format(datetime.datetime.now().time(), X_train.shape[0]))
# print(X_train.shape, y_train.shape)
model.fit(X_train, y_train) 

print("Training complete, time taken: {} seconds".format(os.times()[0] - t1[0]))


# Predict density
y_grid = model.default_y_grid(X_test, grid_points=1000)
# print(y_grid)
# dens = model.predict_density(X, y_grid)
# print(dens)
# print(dens)

# Plot
feat_grid = dataset.get_feature_grid(X_test)
# print(feat_grid.shape)
feat_grid = np.expand_dims(feat_grid, axis=1)
# print(feat_grid.shape)

# print(feat_grid.shape, y_grid.shape)


pis, mus, sigmas = model.mdn.forward(X_test)
# print(pis[1:10,:], mus[1:10,:], sigmas[1:10,:])


dens=model.predict_density(feat_grid, y_grid)

# print(dens.shape)
#exit()

# print(model.mdn.n_gaussians)

# moons
# dataset.plot_conditional_y_given_x(X_test, y_test, predictive_density = dens)
# synth cond
# dataset.plot_conditional_y_given_x(X_test, y_test, predictive_density = dens, mu_fn=dataset.test_mu_fn, sigma_fn=dataset.test_sigma_fn, pi_fn=dataset.test_pi_fn)
# synth const var
dataset.plot_conditional_y_given_x(X_test, y_test, predictive_density = dens)


# plt.figure(figsize=(8, 5))
# plt.scatter(X[:, 0], y, c=mode_ids, cmap='tab10', label='Data', alpha=0.6)
# for i in range(len(X)):
#     plt.plot([X[i, 0]]*len(y_grid), y_grid, dens[i], color='gray', alpha=0.2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('GP Predicted Density and Data')
# plt.legend()
# plt.show()
