# Self-Pruning Neural Network on CIFAR-10

> A neural network that **learns to remove its own unnecessary weights during training** — no post-training pruning required.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

Standard neural network pruning is a **two-step process**: train first, prune later. This project collapses both into one by giving each weight its own learnable **gate** — a scalar in (0, 1) that multiplies the weight's output. A custom **L1 sparsity loss** then pushes unnecessary gates toward exactly zero during backpropagation, effectively pruning those connections on the fly.

The result is a sparse network that automatically discovers which connections matter — without any manual intervention after training.

---

## Key Features

- **`PrunableLinear`** — a drop-in replacement for `nn.Linear` with per-weight gates
- **Differentiable pruning** — gates are learned via standard gradient descent; no special solver needed
- **L1 sparsity regulariser** — encourages exact zeros (same geometry as Lasso regression)
- **Configurable λ** — one hyperparameter controls the full sparsity–accuracy trade-off
- **Self-contained script** — runs end-to-end in Google Colab with zero setup beyond pip installs

---

## Repository Structure

```
.
├── self_pruning_nn.py       # Main script (Colab-compatible, %%  cell markers)
├── report.md                # Auto-generated results report with table & plot
├── gate_distributions.png   # Gate histogram + training curves (generated at runtime)
├── best_pruned_model.pt     # Saved weights of the best model (generated at runtime)
└── README.md
```

---

## ⚙️ How It Works

### 1. The `PrunableLinear` Layer

Every weight `w_ij` is paired with a learnable scalar `gate_score_ij`. During the forward pass:

```
gates         = sigmoid(gate_scores)       # squash to (0, 1)
pruned_weights = weight ⊙ gates            # element-wise mask
output         = input @ pruned_weights.T + bias
```

When a gate collapses to ≈ 0, its weight contributes nothing — it is effectively removed. Because the whole operation is differentiable, gradients flow through both `weight` and `gate_scores` automatically.

### 2. Sparsity Loss

```
Total Loss = CrossEntropyLoss(ŷ, y)  +  λ · Σ sigmoid(gate_scores_ij)
```

The L1 penalty on gate values penalises the total active capacity of the network. The L1 norm is non-smooth at zero, which — just like in Lasso regression — produces **exact zeros** rather than merely small values.

### 3. The λ Trade-off

| λ | Effect |
|---|--------|
| Low  (`1e-5`) | Minimal pruning, high accuracy |
| Medium (`1e-4`) | Balanced sparsity and accuracy |
| High (`1e-3`) | Aggressive pruning, possible accuracy drop |

---

## Quick Start

### Option A — Google Colab (recommended)

```python
# Step 1: upload self_pruning_nn.py to your Colab session, then:
!pip install torch torchvision matplotlib numpy

# Step 2: run everything
exec(open("self_pruning_nn.py").read())
```

Or copy-paste the file directly into Colab cells — each `# %%` block is a natural cell boundary.

### Option B — Local

```bash
# Clone the repo
git clone https://github.com/shivanshikaagarwal1505/the-self-pruning-neural-network.git
cd self-pruning-nn

# Install dependencies
pip install torch torchvision matplotlib numpy

# Run
python self_pruning_nn.py
```

> **GPU recommended.** The script auto-detects CUDA. On CPU, reduce `EPOCHS` to 10–15 for a quick test.

---

## Sample Results

Results will vary by hardware and random seed. Typical outputs look like:

| Lambda | Test Accuracy | Sparsity Level |
|--------|:-------------:|:--------------:|
| 1e-5   | ~52%          | ~12%           |
| 1e-4   | ~50%          | ~45%           |
| 1e-3   | ~44%          | ~88%           |

The gate distribution plot (`gate_distributions.png`) will show a characteristic **bimodal pattern** — a large spike near 0 (pruned weights) and a cluster of active gates away from 0:

```
Count
  │  ▐█
  │  ▐█
  │  ▐█                         ▄▄
  │  ▐█                       ▄████
  └──────────────────────────────────▶ Gate value
     0                               1
     ↑ pruned                  ↑ active
```

---

## Architecture

```
Input (3×32×32)
      │  flatten
      ▼
PrunableLinear(3072 → 512) + BN + ReLU + Dropout(0.2)
      ▼
PrunableLinear(512 → 256)  + BN + ReLU + Dropout(0.2)
      ▼
PrunableLinear(256 → 128)  + BN + ReLU + Dropout(0.2)
      ▼
PrunableLinear(128 → 10)
      ▼
Output (10 classes)
```

---

## Configuration

All key hyperparameters are set as constants near the top of `self_pruning_nn.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LAMBDAS` | `[1e-5, 1e-4, 1e-3]` | Sparsity penalty values to compare |
| `EPOCHS` | `30` | Training epochs per experiment |
| `SEED` | `42` | Global random seed |
| `hidden_dims` | `(512, 256, 128)` | Hidden layer sizes |
| `lr` | `1e-3` | Adam learning rate |
| `batch_size` | `128` | Training batch size |

---

## Dependencies

| Package | Version |
|---------|---------|
| `torch` | ≥ 2.0 |
| `torchvision` | ≥ 0.15 |
| `matplotlib` | ≥ 3.5 |
| `numpy` | ≥ 1.22 |

CIFAR-10 is downloaded automatically via `torchvision.datasets` on first run (~170 MB).

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Case study prompt by **Tredence Analytics**
- CIFAR-10 dataset by [Krizhevsky et al.](https://www.cs.toronto.edu/~kriz/cifar.html)
- Pruning theory inspired by [Han et al., "Learning both Weights and Connections", NeurIPS 2015](https://arxiv.org/abs/1506.02626)
