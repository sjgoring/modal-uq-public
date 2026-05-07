"""
QUEST Selective Prediction with Deep and Mixture-of-Experts Ensembles.

This package provides implementations of selective prediction experiments using
QUEST uncertainty measures on synthetic heteroskedastic 1D data. It supports
two ensemble architectures:

- **DeepEnsemble**: K-component Gaussian-mixture-output PyTorch networks
- **MoEEnsemble**: Mixture-of-Experts regressors (cgmm-based)

Main entry point for experiments is run_experiment(), which orchestrates:
  1. Data generation (1D DGP with Gaussian, bimodal, or skewed noise)
  2. Model training (deep or MoE ensemble)
  3. Uncertainty measure computation (12 variants: variance, entropy, QUEST)
  4. Selective prediction curves (loss vs coverage) and AURC computation
  5. Multi-seed aggregation and NPZ output

For usage, see README.md or run:
    python -m deep_moe.selective_prediction --help
"""

__version__ = "0.1.0"

from .deep_ensemble import DeepEnsemble
from .moe_ensemble import MoEEnsemble
from .selective_prediction import run_experiment
from .plots import plot_loss_curves, plot_aurc_bars

# Convenience imports for commonly-used submodules
from . import active_learning, mpe_dataset

__all__ = [
  "__version__",
  "DeepEnsemble",
  "MoEEnsemble",
  "run_experiment",
  "plot_loss_curves",
  "plot_aurc_bars",
  "active_learning",
  "mpe_dataset",
]
