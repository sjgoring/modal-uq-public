import sys
import os
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from modal_uq.models.gp import GaussianProcessModel
from modal_uq.models.bnn import BNNModel
from modal_uq.datasets.synthetic_conditional import SyntheticMultiModalConditionalDataset#
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C

# Generate synthetic data
n_samples = 2000
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


# Train GP model
# model = GaussianProcessModel(), model_name = "GP", model_fit_args = None
model = BNNModel()
model_name = "BNN"
model_fit_args = {"width": 5, "hidden": 1, "epochs" : 10}



t1 = os.times()
print("Beginning {} fit at {} with {} samples".format(model_name, datetime.datetime.now().time(), X_train.shape[0]))
model.fit(X_train, y_train, model_fit_args)
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


dens=model.predict_density(feat_grid, y_grid)

print(dens.shape)


quit()

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
