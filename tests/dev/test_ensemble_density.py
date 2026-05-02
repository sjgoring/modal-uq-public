import sys
import os
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from modal_uq.models.mdn import MixtureDensityModel
from modal_uq.models.ensemble import Ensemble
from modal_uq.datasets.synthetic_conditional import SyntheticMultiModalConditionalDataset

# Generate synthetic data
n_samples = 5000
n_modes = 2
n_features = 1
mode_locs = [[0,0], [5,5]]
mode_scales = [0.5, 0.5]
dataset = SyntheticMultiModalConditionalDataset(
    n_samples=n_samples,
    n_modes=n_modes,
    n_features=n_features,
    mode_locs=mode_locs,
    mode_scales=mode_scales,
    component_type='gaussian',
    seed_master=42
)

X, y, global_mode, mode_ids = dataset.get_data(pi_fn=dataset.test_pi_fn, mu_fn=dataset.test_mu_fn, sigma_fn=dataset.test_sigma_fn, noise_fn=dataset.test_no_fn)

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

# Possible solution to tuple index issue in MDN fit.
y_train = np.expand_dims(y_train, axis=1)


# Train Ensmble MDN model
# model = MixtureDensityModel(n_gaussians=2, weight_decay=1e-2, epochs=1000)
model = Ensemble(base_model='mdn', base_params={'n_gaussians': 2, 'weight_decay': 1e-2, 'epochs': 1000}, n_members=5,bootstrap=False, seed=42, inferential_choice={'predict':'bma_expected','approximate':'point_estimate', 'point_estimate_criterion':'mle'})
t1 = os.times()
print("Beginning Ensemble MDN fit at {} with {} samples".format(datetime.datetime.now().time(), X_train.shape[0]))
print(X_train.shape, y_train.shape)
model.fit(X_train, y_train)
print("Training complete, time taken: {} seconds".format(os.times()[0] - t1[0]))




# Predict density
y_grid = model.default_y_grid(X_test, grid_points=100)
# print(y_grid)
# dens = model.predict_density(X, y_grid)
# print(dens)
# print(dens)

# Plot
feat_grid = dataset.get_feature_grid(X_test)
print(feat_grid.shape)
feat_grid = np.expand_dims(feat_grid, axis=1)
print(feat_grid.shape)

# print(feat_grid.shape, y_grid.shape)


pis, mus, sigmas = model.mdn.forward(X_test)
print(pis[1:10,:], mus[1:10,:], sigmas[1:10,:])


dens=model.predict_density(feat_grid, y_grid)

# print(dens.shape)
#exit()

print(model.mdn.n_gaussians)

dataset.plot_conditional_y_given_x(X_test, y_test, mu_fn = dataset.test_mu_fn, sigma_fn = dataset.test_sigma_fn, pi_fn = dataset.test_pi_fn, predictive_density = dens)




# plt.figure(figsize=(8, 5))
# plt.scatter(X[:, 0], y, c=mode_ids, cmap='tab10', label='Data', alpha=0.6)
# for i in range(len(X)):
#     plt.plot([X[i, 0]]*len(y_grid), y_grid, dens[i], color='gray', alpha=0.2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('GP Predicted Density and Data')
# plt.legend()
# plt.show()
