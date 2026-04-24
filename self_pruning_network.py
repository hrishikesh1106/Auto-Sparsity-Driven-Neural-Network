"""
Self-Pruning Neural Network — Tredence AI Engineering Case Study
================================================================
A feed-forward network for CIFAR-10 classification where each weight
has a learnable "gate" (sigmoid-gated scalar).  The L1 penalty on gate
values drives most of them toward exactly zero, producing a sparse,
self-pruned network *during* training.

Usage:
    python self_pruning_network.py

Outputs:
    - Console: per-epoch loss/accuracy + final sparsity table
    - gate_distribution.png  : gate-value histogram for best model
    - results_table.txt       : clean text table (used in report)
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
#  PART 1 — PrunableLinear
# ─────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that pairs every weight with a
    learnable gate_score.  During the forward pass:

        gates        = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_w     = weight ⊙ gates                (element-wise)
        output       = x @ pruned_w.T + bias

    Gradients flow through *both* `weight` and `gate_scores` because
    all operations are differentiable PyTorch primitives.

    Why Sigmoid?
        • Squashes gate_scores to (0, 1) — a natural "open/closed" range.
        • Differentiable everywhere, so gradients reach gate_scores cleanly.
        • When combined with the L1 sparsity loss, gate_scores are pushed
          toward −∞, making sigmoid(gate_scores) → 0 (i.e., weight pruned).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias — initialised with Kaiming uniform
        # (matches nn.Linear's default behaviour)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # Gate scores — same shape as weight.
        # Initialised near 0 so sigmoid ≈ 0.5 at the start (half-open gates).
        # Using a small positive value gives the model a fair starting point
        # before the sparsity loss takes effect.
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Kaiming init for weight (standard practice for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Bias init matching nn.Linear
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — compute gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2 — element-wise gate the weights
        pruned_weights = self.weight * gates              # shape: (out, in)

        # Step 3 — standard affine transformation
        # F.linear computes: x @ pruned_weights.T + bias
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached from graph) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────────────────────
#  Network architecture
# ─────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    3-hidden-layer feed-forward network for CIFAR-10 (10 classes).

    Input : 32×32×3 → flattened to 3072
    Layers: PrunableLinear → BN → ReLU → Dropout (repeated ×3) → output

    Batch Norm is applied after each prunable layer so that pruning gates
    do not interfere with the normalisation statistics of subsequent layers.
    """

    def __init__(self, input_dim: int = 3072, num_classes: int = 10) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # Block 1
            PrunableLinear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Block 2
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Block 3
            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Output
            PrunableLinear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten spatial dims
        return self.layers(x)

    def prunable_layers(self):
        """Yield every PrunableLinear module in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ─────────────────────────────────────────────────────────────
#  PART 2 — Sparsity loss
# ─────────────────────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of all gate values across every PrunableLinear layer.

    Why L1 encourages sparsity:
        The gradient of |g| w.r.t. g is sign(g).  Since gates are in (0,1)
        (always positive after sigmoid), the gradient is always +1 — meaning
        the L1 penalty *always* pushes each gate downward, toward zero.
        Unlike L2 (gradient ∝ g), which slows down as g→0 and rarely reaches
        exactly zero, L1 maintains a constant pull regardless of the gate's
        current magnitude, reliably driving small gates all the way to zero.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.sum()
    return total


# ─────────────────────────────────────────────────────────────
#  PART 3 — Data loading
# ─────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256, data_dir: str = "./data"):
    """
    Download (if needed) and return CIFAR-10 train and test DataLoaders.
    Applies standard normalisation (mean/std computed over the training set).
    """
    # Normalisation statistics for CIFAR-10 (per-channel, pre-computed)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────
#  Training & evaluation helpers
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam, device):
    """Run one full pass over the training set.  Returns (avg_total_loss, accuracy)."""
    model.train()
    total_loss = correct = seen = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        # Classification loss (Cross-Entropy)
        ce_loss = F.cross_entropy(logits, labels)

        # Sparsity loss (L1 on all gates)
        sp_loss = sparsity_loss(model)

        # Combined loss
        loss = ce_loss + lam * sp_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += imgs.size(0)

    return total_loss / seen, correct / seen * 100


@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy (%) on the given data loader."""
    model.eval()
    correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        seen    += imgs.size(0)
    return correct / seen * 100


@torch.no_grad()
def compute_sparsity(model, threshold: float = 1e-2) -> float:
    """
    Fraction of weights whose gate value is below `threshold`.
    A high sparsity level (close to 1.0) means successful self-pruning.
    """
    total = pruned = 0
    for layer in model.prunable_layers():
        gates = layer.get_gates()
        total  += gates.numel()
        pruned += (gates < threshold).sum().item()
    return pruned / total * 100


@torch.no_grad()
def collect_all_gates(model) -> np.ndarray:
    """Collect every gate value across all PrunableLinear layers."""
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.get_gates().cpu().numpy().ravel())
    return np.concatenate(all_gates)


# ─────────────────────────────────────────────────────────────
#  Single training run
# ─────────────────────────────────────────────────────────────

def run_experiment(lam: float, train_loader, test_loader, device,
                   epochs: int = 20, lr: float = 1e-3):
    """
    Train a fresh SelfPruningNet with sparsity coefficient `lam`.
    Returns (test_accuracy, sparsity_level, final_gates_array).
    """
    model = SelfPruningNet().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  λ = {lam:.0e}   |  epochs = {epochs}  |  device = {device}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, lam, device)
        test_acc  = evaluate(model, test_loader, device)
        sparsity  = compute_sparsity(model)
        scheduler.step()

        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:5.2f}% | "
              f"Test Acc: {test_acc:5.2f}% | "
              f"Sparsity: {sparsity:5.1f}%  "
              f"[{time.time()-t0:.1f}s]")

    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    final_gates    = collect_all_gates(model)

    return final_test_acc, final_sparsity, final_gates, model


# ─────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────

def plot_gate_distribution(gates: np.ndarray, lam: float,
                           save_path: str = "gate_distribution.png"):
    """
    Histogram of gate values for the best model.
    A successful pruning shows a large spike at 0 and a cluster away from 0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        f"Gate Value Distribution  (λ = {lam:.0e})\n"
        f"Total gates: {len(gates):,}  |  "
        f"Near-zero (< 0.01): {(gates < 0.01).mean()*100:.1f}%",
        fontsize=13, fontweight="bold", y=1.02
    )

    # ── Left: full range ──────────────────────────────────────
    ax = axes[0]
    ax.hist(gates, bins=80, color="#2563EB", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Full Range [0, 1]", fontsize=11)
    ax.axvline(0.01, color="#EF4444", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    # ── Right: zoom on [0, 0.1] to show spike detail ─────────
    ax2 = axes[1]
    ax2.hist(gates[gates < 0.1], bins=60, color="#10B981", edgecolor="white",
             linewidth=0.3)
    ax2.set_xlabel("Gate Value", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Zoomed: [0, 0.1]", fontsize=11)
    ax2.axvline(0.01, color="#EF4444", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✓  Gate distribution plot saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    # ── Reproducibility ──────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Data ─────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)

    # ── Experiments: three values of λ ───────────────────────
    # λ = 1e-5  →  low sparsity pressure  (highest accuracy expected)
    # λ = 1e-4  →  medium sparsity pressure
    # λ = 1e-3  →  high sparsity pressure  (most gates pruned)
    lambdas = [1e-5, 1e-4, 1e-3]
    EPOCHS  = 20    # Increase to 40–60 for production runs

    results      = {}   # lam → (test_acc, sparsity, gates)
    best_lam     = None
    best_gates   = None
    best_acc     = -1

    for lam in lambdas:
        acc, sparsity, gates, _ = run_experiment(
            lam, train_loader, test_loader, device, epochs=EPOCHS
        )
        results[lam] = (acc, sparsity, gates)
        if acc > best_acc:
            best_acc   = acc
            best_lam   = lam
            best_gates = gates

    # ── Results table ─────────────────────────────────────────
    header = f"\n{'─'*52}\n{'Lambda':>12}  {'Test Accuracy':>14}  {'Sparsity (%)':>13}\n{'─'*52}"
    rows   = []
    for lam in lambdas:
        acc, sparsity, _ = results[lam]
        rows.append(f"{lam:>12.0e}  {acc:>13.2f}%  {sparsity:>12.1f}%")

    table_str = header + "\n" + "\n".join(rows) + f"\n{'─'*52}\n"
    print(table_str)

    # Save table to file
    with open("results_table.txt", "w") as f:
        f.write("Self-Pruning Neural Network — Results Summary\n")
        f.write("CIFAR-10 | Feed-Forward | 3 Hidden Layers\n")
        f.write(table_str)
    print("  ✓  Results table saved → results_table.txt")

    # ── Gate distribution plot for best model ─────────────────
    plot_gate_distribution(best_gates, best_lam, save_path="gate_distribution.png")

    print(f"\n  Best model: λ = {best_lam:.0e} | "
          f"Test Acc = {results[best_lam][0]:.2f}% | "
          f"Sparsity = {results[best_lam][1]:.1f}%\n")


if __name__ == "__main__":
    main()
