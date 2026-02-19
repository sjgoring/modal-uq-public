import sys
import os
import numpy as np
# ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modal_uq.models.bnn_vi import BayesianNNVI
from modal_uq.uncertainty.quest import QUESTUncertainty
#import torch
import torch.optim as optim

# Small smoke test
X = np.linspace(0, 1, 50).reshape(-1, 1).astype(np.float32)
y = np.sin(2 * np.pi * X).ravel().astype(np.float32)

# Build model with small hidden size for speed
model = BayesianNNVI(input_dim=1, hidden_dims=[8], n_components=2, n_mc_samples=4,
                     n_epochs=1, batch_size=16, seed=0)
# Build internal network (since fit normally does it)
model.model = model._build_model().to(model.device)
model.optimizer = optim.Adam(model.model.parameters(), lr=model.learning_rate)

# Do not train; just evaluate posterior density grid and meta-quest
print('Calling get_parameter_posterior...')
post = model.get_parameter_posterior()
print('Posterior keys:', post.keys())
print('Num params:', post['means'].shape)

print('Computing parameter posterior density on grid...')
dens, axes = model.parameter_posterior_density_on_grid(grid_points_per_dim=16)
print('Density shape:', dens.shape)

print('Running meta-quest via QUESTUncertainty.meta_quest...')
quest = QUESTUncertainty(alpha=0.1, decomposition='epistemic', grid_points=128)
meta = quest.meta_quest(model=model)
print('meta-QUEST:', meta)
