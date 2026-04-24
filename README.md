# Self-Pruning Neural Network on CIFAR-10

> A neural network that **learns to remove its own unnecessary weights during training** — no post-training pruning required.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

Standard pruning is a two-stage process: train a network, then remove unimportant weights after the fact. This project collapses both stages into one. Every weight in the network is paired with a learnable scalar gate. During training, an L1 penalty on these gates pushes unneeded ones to exactly zero, effectively removing those weights from the network in real time.
 
The result is a single training run that simultaneously optimizes for classification accuracy and architectural sparsity, with the trade-off controlled by one hyperparameter, lambda.

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

## How It Works

### Gated Linear Layer
 
The core building block is `PrunableLinear`, a drop-in replacement for `nn.Linear`.
 
```
gates          = clamp(gate_scores, 0, 1)
pruned_weights = weight * gates
output         = pruned_weights @ x.T + bias
```
 
Each `gate_score` is a learnable parameter of the same shape as the weight matrix.
Clamping to [0, 1] keeps gates interpretable and avoids the vanishing gradient
problem that arises with sigmoid in the flat tails.
 
### Loss Function
 
```
Total Loss = CrossEntropyLoss(y_hat, y)  +  lambda * SparsityLoss
 
SparsityLoss = (1 / N_gates) * sum of all gate values
```
 
Dividing by the total gate count normalizes the sparsity term to [0, 1] regardless
of network size. This makes lambda directly interpretable: a value of 1.0 means the
optimizer treats closing one unit of gate capacity as equivalent to reducing
cross-entropy by 1.0.
 
### Why L1 Encourages Exact Zeros
 
The L1 sub-gradient is a constant +1 or -1 regardless of the current value.
This means the pull toward zero never weakens, unlike L2 regularization whose
gradient shrinks proportionally to the weight magnitude. L2 produces small values;
L1 produces exact zeros. This is the same geometric reason LASSO regression produces
sparse solutions while Ridge regression does not.
 
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

## Architecture

```
Input (3×32×32)
      │  flatten
      ▼
PrunableLinear(3072 → 512) -> BatchNorm1d -> ReLU -> Dropout(0.2)
      ▼
PrunableLinear(512 → 256) -> BatchNorm1d -> ReLU -> Dropout(0.2)
      ▼
PrunableLinear(256 → 128)  -> BatchNorm1d -> ReLU -> Dropout(0.2)
      ▼
PrunableLinear(128 → 10)
      ▼
Output (10 classes)
```
Total learnable gates: 1,737,984
---

## Configuration

All key hyperparameters are set as constants near the top of `self_pruning_nn.py`:

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 |
| Optimizer | Adam |
| Weight learning rate | 0.001 |
| Gate learning rate | 0.1 (separate param group) |
| Weight decay | 1e-4 (weights only, not gates) |
| LR schedule | Cosine annealing |
| Batch size | 128 |
| Epochs | 40 |
| Gate warmup | First 5 epochs (gates frozen) |
| Gradient clipping | max norm = 5.0 |
 
Gates use a 100x higher learning rate than weights. This allows gate_scores to
respond quickly to sparsity pressure without destabilizing weight optimization.
The 5-epoch warmup lets the network first learn a useful classification solution
before pruning pressure begins.

---
## Results

Results will vary by hardware and random seed. Typical outputs look like:

| Lambda | Test Accuracy | Sparsity Level |
|--------|:-------------:|:--------------:|
| 1.0    | ~59%          | ~48%           |
| 3.0    | ~60%          | ~72%           |
| 10.0   | ~44%          | ~88%           |

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
