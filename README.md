<div align="center">

```
███████╗███████╗██╗     ███████╗      ██████╗ ██████╗ ██╗   ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
██╔════╝██╔════╝██║     ██╔════╝      ██╔══██╗██╔══██╗██║   ██║████╗  ██║██║████╗  ██║██╔════╝
███████╗█████╗  ██║     █████╗  █████╗██████╔╝██████╔╝██║   ██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
╚════██║██╔══╝  ██║     ██╔══╝  ╚════╝██╔═══╝ ██╔══██╗██║   ██║██║╚██╗██║██║██║╚██╗██║██║   ██║
███████║███████╗███████╗██║           ██║     ██║  ██║╚██████╔╝██║ ╚████║██║██║ ╚████║╚██████╔╝
╚══════╝╚══════╝╚══════╝╚═╝           ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝
```

### *A neural network that learns to destroy itself — and becomes stronger for it.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-00B4D8?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-30%20Passed-22C55E?style=for-the-badge&logo=pytest&logoColor=white)](#testing)
[![Tredence](https://img.shields.io/badge/Submission-Tredence%20AI%20Internship-FF6B35?style=for-the-badge)](https://tredence.com)

<br/>

</div>

---

<div align="center">

## ✂️ The Core Idea

</div>

> **Traditional pruning:** Train → Evaluate → Prune → Fine-tune *(4 steps, post-hoc)*
>
> **This approach:** Train *(1 step — pruning happens automatically, from epoch 1)*

Every single weight in this network has a **learnable gate** — a sigmoid scalar that can open or close the connection. An L1 sparsity penalty creates constant pressure to close gates. The classification loss fights back, keeping gates open only when they genuinely matter. The result? The network **learns which connections to kill**, producing a lean, efficient model without any post-training intervention.

---

<div align="center">

## 📁 Repository Structure

</div>

```
self-pruning-network/
│
├── 📄 self_pruning_network.py    ─── Core implementation (single file)
│   ├── PrunableLinear            ─── Custom gated linear layer
│   ├── SelfPruningNet            ─── Full network using prunable layers
│   ├── sparsity_loss()           ─── L1 penalty on sigmoid gate values
│   ├── train_one_epoch()         ─── Training loop with combined loss
│   ├── compute_sparsity()        ─── Measures % of pruned connections
│   └── main()                    ─── Runs all 3 lambda experiments
│
├── 📄 REPORT.md                  ─── Written analysis and results
├── 📄 requirements.txt           ─── Python dependencies
├── 📄 README.md                  ─── You are here
│
├── 📂 tests/
│   └── test_self_pruning.py      ─── 30 pytest unit + integration tests
│
├── 🖼️  gate_distribution.png     ─── [Generated] Gate histogram plot
└── 📊 results_table.txt          ─── [Generated] λ comparison table
```

---

<div align="center">

## ⚙️ Setup

</div>

**1 — Clone**
```bash
git clone https://github.com/<your-username>/self-pruning-network.git
cd self-pruning-network
```

**2 — Virtual environment** *(recommended)*
```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

**3 — Install dependencies**
```bash
pip install -r requirements.txt
```

<details>
<summary>🖥️ <strong>GPU Setup (CUDA)</strong> — click to expand</summary>

<br/>

| CUDA Version | Install Command |
|:---:|:---|
| **12.1** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **11.8** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| **CPU only** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

</details>

---

<div align="center">

## 🚀 Usage

</div>

```bash
python self_pruning_network.py
```

CIFAR-10 (~170 MB) downloads automatically on first run into `./data/`.
The script trains **three models** — one per λ value — sequentially.

<br/>

**What gets produced:**

| File | Description |
|:---|:---|
| `gate_distribution.png` | Gate histogram for the best-accuracy model |
| `results_table.txt` | Accuracy & sparsity summary across all λ runs |
| Console | Per-epoch: loss · train acc · test acc · sparsity % |

<br/>

**Tune it yourself** — edit `main()` in `self_pruning_network.py`:

```python
lambdas = [1e-5, 1e-4, 1e-3]   # sparsity pressure: low → medium → high
EPOCHS  = 20                     # ↑ to 40–60 on GPU for production accuracy
```

---

<div align="center">

## 🔬 How It Works

</div>

### Part 1 — `PrunableLinear`: The Gated Layer

A custom `nn.Module` that replaces every `nn.Linear` in the network. It holds **three learnable parameter tensors**:

| Parameter | Shape | Purpose |
|:---:|:---:|:---|
| `weight` | `(out_features, in_features)` | Standard connection weights |
| `bias` | `(out_features,)` | Standard bias term |
| `gate_scores` | `(out_features, in_features)` | Raw logits for each gate — **learned by the optimizer** |

**The forward pass in 3 lines:**
```python
gates          = torch.sigmoid(self.gate_scores)   # → values in (0, 1)
pruned_weights = self.weight * gates               # → element-wise mask
return F.linear(x, pruned_weights, self.bias)      # → standard affine
```

Gradients flow to **both** `weight` and `gate_scores` automatically — all ops are native PyTorch primitives, no custom `backward()` required.

<br/>

### Part 2 — Sparsity Loss: Why L1?

```
Total Loss = CrossEntropy(logits, labels)  +  λ · Σᵢ sigmoid(gate_scoreᵢ)
```

The penalty term is the **L1 norm of all gate values**. Here's why L1 (not L2) is the right choice:

```
∂(L1) / ∂(gate_score) = sign(gate) = +1   always (gates > 0 after sigmoid)
```

| Regularizer | Gradient near zero | Effect |
|:---:|:---:|:---|
| **L2** | `2g → 0` as `g → 0` | Shrinks gates but rarely reaches *exactly* zero |
| **L1** | `+1` *always* | Constant pull — reliably drives gates to **exactly** zero |

The L1 penalty creates a tug-of-war: the classification loss keeps important gates open; the sparsity loss kills everything else.

<br/>

### Part 3 — Training Loop

```python
# Single backward pass updates weights AND gates simultaneously
ce_loss = F.cross_entropy(logits, labels)
sp_loss = sparsity_loss(model)            # L1 on all sigmoid(gate_scores)
loss    = ce_loss + lam * sp_loss
loss.backward()
optimizer.step()
```

After training, any weight whose gate drops below `0.01` is considered **pruned**.

---

<div align="center">

## 📊 Results

*Trained for 20 epochs on CIFAR-10 · Adam optimizer · CosineAnnealingLR*

</div>

<div align="center">

| λ (Lambda) | Test Accuracy | Sparsity Level | Behaviour |
|:---:|:---:|:---:|:---|
| `1e-5` | ~52.3% | ~12% | 🟢 Gentle — almost all gates stay open |
| `1e-4` | ~49.1% | ~47% | 🟡 Balanced — clear bimodal gate distribution |
| `1e-3` | ~43.7% | ~81% | 🔴 Aggressive — 4 in 5 weights removed |

</div>

<br/>

**The gate distribution for `λ = 1e-4`** (best trade-off) shows the hallmark of successful pruning:

```
Count (log scale)
    │
    █                                          ← massive spike at 0
    █                                            (~47% of all gates)
    █
    █   .  .  .  .  .                          ← live-weight cluster
    █ .              . .                         (gates 0.3 – 0.7)
    └─────────────────────────── Gate Value
    0                            1
```

> 💡 For ~55%+ accuracy, train for **40–60 epochs on a GPU**. The architecture is intentionally simple (feed-forward, no convolutions) to keep the focus on the pruning mechanism.

---

<div align="center">

## 🧪 Testing

</div>

```bash
pytest tests/ -v
```

**30 tests** across 6 test classes:

| Test Class | Coverage |
|:---|:---|
| `TestPrunableLinear` | Output shapes, parameter registration, gradient flow, boundary gate behaviour |
| `TestSparsityLoss` | Positivity, scalar output, differentiability, monotonicity w.r.t. gate values |
| `TestComputeSparsity` | Range `[0, 100]`, all-pruned edge case, all-open edge case |
| `TestCollectAllGates` | Returns numpy array, correct total count, values in `[0, 1]` |
| `TestSelfPruningNet` | Output shape (image + flat input), layer count, full forward-backward pass |
| `TestSparsityVsLambda` | Higher λ produces lower mean gate value (integration test) |

---

<div align="center">

## 📦 Requirements

</div>

```
torch        >= 2.0.0
torchvision  >= 0.15.0
numpy        >= 1.24.0
matplotlib   >= 3.7.0
pytest       >= 7.4.0    # for tests only
```

---

<div align="center">

## 🏗️ Network Architecture

</div>

```
Input Image (32×32×3)
        │
        ▼  flatten
  [3072-dim vector]
        │
        ▼
┌─────────────────────────┐
│   PrunableLinear(3072→512) │  ← 1,572,864 gates
│   BatchNorm1d + ReLU      │
│   Dropout(0.3)            │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   PrunableLinear(512→256)  │  ← 131,072 gates
│   BatchNorm1d + ReLU      │
│   Dropout(0.3)            │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   PrunableLinear(256→128)  │  ← 32,768 gates
│   BatchNorm1d + ReLU      │
│   Dropout(0.2)            │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   PrunableLinear(128→10)   │  ← 1,280 gates
└─────────────────────────┘
        │
        ▼
  Class Logits (10)

  Total learnable gates: ~1,737,984
```

---

<div align="center">

<br/>

---

*Submitted by **Hrishikesh Ramachandran***
*VIT Chennai · Integrated M.Tech — Computer Science Engineering (Business Analytics)*
*Tredence AI Engineering Internship · 2025 Cohort*

---

<br/>

*"The connections you cut define the network as much as the ones you keep."*

</div>
