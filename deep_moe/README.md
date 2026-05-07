# deep_moe — Uncertainty-Driven Active Learning & Selective Prediction

This repository contains implementations and utilities for uncertainty quantification and active learning experiments using deep ensembles and mixture-of-experts models on synthetic and real-world datasets.

## Overview

This package provides code for two experiments:

1. **Selective Prediction**: Evaluate which uncertainty measures best identify hard-to-predict test points. Given a trained model, rank test instances by uncertainty and measure how well rankings correlate with actual prediction error.

2. **Active Learning**: Use uncertainty measures to prioritize which unlabeled points to acquire, training iteratively by selecting the most informative samples according to various acquisition strategies.

Both workflows rely on the same suite of uncertainty measures (variance, entropy, QUEST), enabling you to assess which measures are most effective for decision-making.

## Workflows

### Selective Prediction (Offline Evaluation)

**Goal**: Benchmark uncertainty measures by analyzing how well they identify hard-to-predict test points.

**Process**:
1. Train an ensemble model on labeled data
2. Compute uncertainty measures for all test points
3. Rank test points by each measure
4. Plot selective loss curves: how does test error decrease as we focus on low-uncertainty points?
5. Summarize performance with AURC (Area Under Risk-Coverage curve)

**Best for**: Comparing uncertainty quantification methods in a controlled, reproducible setting with fixed train/test splits.

**Quick start** (synthetic heteroskedastic 1D data):
```bash
python -m deep_moe.selective_prediction --noise all --estimator all \
    --model moe --M 10 --K 3 --n-seeds 10 --n-jobs 4 \
    --output-dir results/full_moe
```

### Active Learning (Online Acquisition)

**Goal**: Train models efficiently by acquiring the most informative unlabeled points according to uncertainty measures.

**Process**:
1. Start with a small labeled pool and large unlabeled pool
2. Train an ensemble on current labeled data
3. Score all unlabeled points with uncertainty measures
4. Acquire (label) the top-k most uncertain points
5. Retrain and repeat for multiple rounds
6. Track learning curves and mode absolute error over time

**Best for**: Studying the effectiveness of different uncertainty measures as acquisition functions in budget-constrained scenarios.

**Quick start** (on the MPE real-world dataset):
```bash
python -m deep_moe.active_learning --dataset mpe --noise gaussian \
    --n-train 1200 --n-test 600 --d-init 200 --n-rounds 5 \
    --n-seeds 5 --n-jobs 1 --output-dir runs/moe_AL_mpe
```

Or using a config file (from the repository root):
```bash
python -m deep_moe.active_learning --config configs/selective_mpe.json
```

## Key Features

- **Two ensemble types**: Deep ensembles (PyTorch-based Gaussian-mixture outputs) or MoE (as used in the paper, cgmm-based Mixture-of-Experts regressors)
- **Three noise distributions**: Gaussian, bimodal, and skewed
- **Uncertainty measures**: variance and entropy (aleatoric/epistemic/total) plus QUEST variants
- **Two estimation targets**: oracle (true conditional density) or MLE (best ensemble member)
- **Parallel execution**: joblib-based parallelization across seeds
- **Reproducible output**: NPZ results with curves, AURCs, and standard errors across seeds

---

## Installation & Setup

### Prerequisites

Ensure you have Python 3.8-3.12 installed. This package requires PyTorch, scikit-learn, and the cgmm library for MoE support.

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "from deep_moe.selective_prediction import run_experiment; print('Imports successful')"
   ```
## Data & Models

### Data Generation (dgp.py)

Synthetic regression data is generated from:
$$X \sim \text{Uniform}[-2, 2], \quad y \sim p(y|x)$$

where the conditional density depends on the noise distribution:

- **Gaussian**: $p(y|x) = \mathcal{N}(\mu(x), \sigma^2(x))$
- **Bimodal**: mixture of two Gaussians with different means/variances
- **Skewed**: SkewNormal distribution

The mean function is $\mu(x) = \sin(\pi x)$ across all distributions. Standard deviations are heteroskedastic: smaller near $x=0$, larger at the boundaries.

### Ensemble Models

Two model classes are supported:

#### Deep Ensemble
- **Architecture**: $M$ independent Gaussian-mixture-output neural networks
- **Each member**: fully connected with $K$ mixture components per output
- **Training**: NLL loss + optional entropy regularization over $n_\text{epochs}$ epochs
- **Configuration**: `--model deep --M 10 --K 3 --hidden-dim 32 --n-hidden 2 --n-epochs 500`

#### Mixture of Experts (MoE)
- This is the approach used in the paper.
- **Architecture**: $M$ cgmm-based expert regressors with $K$ experts each
- **Training**: bootstrap resampling (default) for ensemble diversity
- **Configuration**: `--model moe --M 10 --K 3 --bootstrap` (or `--no-bootstrap`)

---

## Uncertainty Measures

### Variance Measures
- **AU** (Aleatoric Uncertainty): within-component variance from $\hat{p}_\star$
- **EU** (Epistemic Uncertainty): variance of component means across ensemble
- **TU** (Total Uncertainty): sum of AU and EU

### Entropy Measures
Same decomposition using differential entropy instead of variance.

### QUEST Measures
**QUEST** uses high-density region (HDR) thresholding:
- **AU**: HDR volume under the truth approximation $\hat{p}_\star$
- **EU**: HDR volume in the ensemble parameter space
- **TU**: HDR volume normalized by distance to ensemble predictions

Available as:
- `quest_*_01`: HDR at $\alpha = 0.1$
- `quest_*_g`: HDR computed on a coarser global grid

---

## Running Experiments

### Selective Prediction: Command-Line Interface

Run experiments with `selective_prediction.py`. All combinations of noise and estimator:
```bash
python selective_prediction.py --model moe --n-seeds 10 --n-jobs 4
```

**Common selective prediction configurations**:

**Gaussian noise, oracle estimator, deep ensemble (fast smoke test):**
```bash
python selective_prediction.py --noise gaussian --estimator oracle \
    --model deep --M 5 --K 3 --n-train 500 --n-test 200 \
    --n-seeds 3 --n-jobs 1 --output-dir results/smoke_test
```

**Full experiment with MoE:**
```bash
python selective_prediction.py --noise all --estimator all \
    --model moe --M 10 --K 3 --n-seeds 10 --n-jobs 4 \
    --output-dir results/full_moe
```

**Deep ensemble with regularization:**
```bash
python selective_prediction.py --noise bimodal --estimator mle \
    --model deep --M 20 --K 5 --entropy-bonus 0.05 \
    --n-epochs 1000 --n-seeds 5 --output-dir results/deep_reg
```

### Active Learning: Command-Line Interface

Run experiments with `active_learning.py`. Example with synthetic data:
```bash
python active_learning.py --dataset synthetic --noise gaussian \
    --n-train 1200 --n-test 600 --d-init 200 --n-rounds 5 \
    --n-seeds 5 --n-jobs 4 --output-dir runs/AL_synthetic
```

**Common active learning configurations**:

**MPE real-world dataset with Gaussian noise:**
```bash
python active_learning.py --dataset mpe --noise gaussian \
    --n-train 1200 --n-test 600 --d-init 200 --n-rounds 5 \
    --n-seeds 5 --n-jobs 1 --output-dir runs/moe_AL_mpe
```

**Deep ensemble with bimodal noise:**
```bash
python active_learning.py --dataset synthetic --noise bimodal \
    --model deep --M 10 --K 3 --d-init 150 --n-rounds 10 \
    --n-seeds 3 --n-jobs 1 --output-dir runs/AL_deep
```

**Via config file:**
```bash
python -m deep_moe.active_learning --config configs/selective_mpe.json
```

### Configuration Reference

#### Selective Prediction Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--noise` | `all` | Noise type: `gaussian`, `bimodal`, `skewed`, or `all` |
| `--estimator` | `all` | Truth approximation: `oracle`, `mle`, or `all` |
| `--model` | `deep` | Ensemble class: `deep` or `moe` |
| `--n-train` | 1000 | Training set size |
| `--n-test` | 500 | Test set size |
| `--M` | 10 | Number of ensemble members |
| `--K` | 3 | Components/experts per member |
| `--n-seeds` | 10 | Number of random seeds |
| `--base-seed` | 0 | Starting seed (each experiment uses `base_seed + i`) |
| `--n-jobs` | 1 | Parallel jobs for seed execution |
| `--n-coverage-points` | 20 | Number of points on the coverage curve (0.05 to 1.0) |
| `--output-dir` | `results` | Output directory for NPZ files |
| **Deep ensemble only:** | | |
| `--hidden-dim` | 32 | Hidden layer dimension |
| `--n-hidden` | 2 | Number of hidden layers |
| `--n-epochs` | 500 | Training epochs |
| `--batch-size` | 64 | Batch size for SGD |
| `--lr` | 0.001 | Learning rate |
| `--entropy-bonus` | 0.0 | Entropy regularization coefficient |
| **MoE only:** | | |
| `--no-bootstrap` | False | Disable bootstrap resampling (default: enabled) |

#### Active Learning Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `synthetic` | Data source: `synthetic` or `mpe` (real-world) |
| `--noise` | `gaussian` | Noise type for synthetic data: `gaussian`, `bimodal`, `skewed` |
| `--model` | `moe` | Ensemble class: `moe` or `deep` |
| `--n-train` | 1200 | Total labeled + unlabeled pool size |
| `--n-test` | 600 | Test set size |
| `--d-init` | 200 | Initial labeled set size |
| `--n-rounds` | 5 | Number of acquisition rounds |
| `--M` | 10 | Number of ensemble members |
| `--K` | 3 | Components/experts per member |
| `--n-seeds` | 5 | Number of random seeds |
| `--n-jobs` | 1 | Parallel jobs for seed execution |
| `--output-dir` | `runs` | Output directory for results |
| `--config` | (optional) | JSON config file (overrides CLI args if provided) |

---

## Output & Reproducibility

### Selective Prediction Output

Each experiment saves results to:
```
results_{noise_dist}_{estimator}.npz
```

**File contents**:
- **Metadata**:
  - `coverages`: 1D array of coverage fractions (0.05 to 1.0)
  - `noise_dist`: string identifier (gaussian, bimodal, skewed)
  - `estimator`: string identifier (oracle, mle)
  - `n_seeds`: number of aggregated seeds

- **Test loss summary**:
  - `test_loss_mean`: mean pointwise loss across all test points
  - `test_loss_se`: standard error across seeds

- **Per-measure arrays** (for each measure name, e.g., `quest_tu_g`):
  - `loss_mean_{name}`: selective loss curve (mean across seeds)
  - `loss_se_{name}`: standard error of loss curve
  - `aurc_mean_{name}`: scalar AURC value (mean)
  - `aurc_se_{name}`: scalar AURC standard error

**Measure names**: `var_au`, `var_eu`, `var_tu`, `ent_au`, `ent_eu`, `ent_tu`, `quest_au_01`, `quest_au_g`, `quest_eu_01`, `quest_eu_g`, `quest_tu_01`, `quest_tu_g`, `random`

### Active Learning Output

Results are saved to the specified output directory with the structure:
```
runs/experiment_name/
  ├── seed_*.npy          (per-seed results)
  ├── log_seed_*.txt      (per-seed logs)
  └── summary.npz         (aggregated across seeds)
```

### Seed Management & Reproducibility

Both workflows use consistent seed specification:
- **Training data**: generated with seed $s$
- **Test data**: generated with seed $s + 100000$ (ensures consistent split across experiments)
- **Model training**: uses seed $s$ for weight initialization
- **Uncertainty/baseline randomness**: generated with separate RNG offsets

Cross-seed aggregation: means and standard errors computed using numpy with `ddof=1` (unbiased estimator).

### Expected Runtime

**Selective Prediction**:
- Smoke test (3 seeds, gaussian, 500 train/200 test, deep): ~30 seconds
- Full experiment (10 seeds, all noise types, 1000 train, deep, 4 jobs): ~3–5 minutes
- MoE experiments (slower than deep): 2–3× runtime

**Active Learning**:
- Synthetic (5 seeds, 5 rounds, 1200 pool, moe): ~1–2 minutes
- MPE dataset (5 seeds, 5 rounds, moe): ~5–10 minutes

---

## Workflow Details

### Loss Function

The selective prediction loss is defined as:
$$\text{loss}(x) = \log p_\star(y^* | x) - \log p_\star(\hat{y} | x)$$

where:
- $p_\star$ is the truth approximation (oracle: true density, MLE: best member predictive)
- $y^*$ is the true label (or mode of true conditional if discrete)
- $\hat{y}$ is the ensemble's predicted mode

Both modes are found via grid search on a shared evaluation grid.

### Selective Loss Curves

For coverage fraction $c \in [0.05, 1.0]$:
$$L(c) = \mathbb{E}[\text{loss}(x) | x \text{ in lowest-}c\text{ uncertainty quantile}]$$

Lower curves indicate that low-uncertainty points have lower loss (good UM).

---

## Citation & Dependencies

### Third-Party Packages

- **PyTorch**: Deep neural network training
- **scikit-learn**: Preprocessing and utilities
- **cgmm**: Mixture-of-Experts regressors (MoE models)
- **joblib**: Parallel execution
- **numpy, scipy, matplotlib**: Numerics and visualization

