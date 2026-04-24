"""
tests/test_self_pruning.py
==========================
Unit and integration tests for the Self-Pruning Neural Network.

Run with:
    pytest tests/ -v
or from the repo root:
    pytest -v
"""

import math
import sys
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── make the parent directory importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_pruning_network import (
    PrunableLinear,
    SelfPruningNet,
    sparsity_loss,
    compute_sparsity,
    collect_all_gates,
    evaluate,
    train_one_epoch,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_layer():
    """A small PrunableLinear layer for quick tests."""
    return PrunableLinear(in_features=16, out_features=8)


@pytest.fixture
def small_net():
    """A small SelfPruningNet-like network for integration tests."""
    # Reuse the real class but note input_dim defaults to 3072.
    # We pass a tiny one to keep tests fast.
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                PrunableLinear(32, 64),
                nn.ReLU(),
                PrunableLinear(64, 10),
            )
        def forward(self, x):
            return self.net(x)
        def prunable_layers(self):
            for m in self.modules():
                if isinstance(m, PrunableLinear):
                    yield m
    return TinyNet()


@pytest.fixture
def tiny_loader():
    """A small synthetic DataLoader (input_dim=32, 10 classes)."""
    torch.manual_seed(0)
    X = torch.randn(128, 32)
    y = torch.randint(0, 10, (128,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=32, shuffle=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Part 1 — PrunableLinear
# ─────────────────────────────────────────────────────────────────────────────

class TestPrunableLinear:

    def test_output_shape(self, small_layer):
        """Forward pass produces the correct output shape."""
        x = torch.randn(4, 16)          # batch=4, in_features=16
        out = small_layer(x)
        assert out.shape == (4, 8), f"Expected (4, 8), got {out.shape}"

    def test_parameters_registered(self, small_layer):
        """weight, bias, and gate_scores must all be nn.Parameters."""
        param_names = {n for n, _ in small_layer.named_parameters()}
        assert "weight"      in param_names, "weight not registered as parameter"
        assert "bias"        in param_names, "bias not registered as parameter"
        assert "gate_scores" in param_names, "gate_scores not registered as parameter"

    def test_gate_scores_shape_matches_weight(self, small_layer):
        """gate_scores must have the same shape as weight."""
        assert small_layer.gate_scores.shape == small_layer.weight.shape, (
            f"gate_scores shape {small_layer.gate_scores.shape} != "
            f"weight shape {small_layer.weight.shape}"
        )

    def test_gates_bounded_in_zero_one(self, small_layer):
        """Gates (sigmoid of gate_scores) must lie strictly in (0, 1)."""
        gates = small_layer.get_gates()
        assert (gates > 0).all(),  "Some gate values are <= 0"
        assert (gates < 1).all(),  "Some gate values are >= 1"

    def test_gradient_flows_to_weight(self, small_layer):
        """Backprop must reach the weight parameter."""
        x   = torch.randn(4, 16)
        out = small_layer(x)
        loss = out.sum()
        loss.backward()
        assert small_layer.weight.grad is not None, "No gradient on weight"
        assert not torch.all(small_layer.weight.grad == 0), "weight gradient is all-zero"

    def test_gradient_flows_to_gate_scores(self, small_layer):
        """Backprop must reach the gate_scores parameter."""
        x   = torch.randn(4, 16)
        out = small_layer(x)
        loss = out.sum()
        loss.backward()
        assert small_layer.gate_scores.grad is not None, "No gradient on gate_scores"
        assert not torch.all(small_layer.gate_scores.grad == 0), (
            "gate_scores gradient is all-zero"
        )

    def test_zero_gate_scores_halves_effective_weight(self, small_layer):
        """
        With gate_scores all-zero, sigmoid = 0.5, so pruned_weights = 0.5 * weight.
        Output should equal F.linear(x, 0.5*weight, bias).
        """
        small_layer.gate_scores.data.fill_(0.0)
        x = torch.randn(4, 16)
        out      = small_layer(x)
        expected = F.linear(x, 0.5 * small_layer.weight, small_layer.bias)
        assert torch.allclose(out, expected, atol=1e-6), (
            "With gate_scores=0, output should equal F.linear(x, 0.5*w, b)"
        )

    def test_very_negative_gate_scores_near_zero_output(self):
        """
        With gate_scores → -∞, gates → 0, so the layer contributes near-zero
        weight signal (only bias remains).
        """
        layer = PrunableLinear(8, 4)
        layer.gate_scores.data.fill_(-100.0)   # sigmoid(-100) ≈ 0
        x   = torch.randn(2, 8)
        out = layer(x)
        # Output should be approximately just the bias
        expected = layer.bias.unsqueeze(0).expand(2, -1)
        assert torch.allclose(out, expected, atol=1e-3), (
            "With gate_scores=-100, output should be ≈ bias"
        )

    def test_very_positive_gate_scores_near_full_weight(self):
        """
        With gate_scores → +∞, gates → 1, so pruned_weights ≈ weight.
        Output should match a standard linear layer.
        """
        layer = PrunableLinear(8, 4)
        layer.gate_scores.data.fill_(100.0)    # sigmoid(100) ≈ 1
        x   = torch.randn(2, 8)
        out = layer(x)
        expected = F.linear(x, layer.weight, layer.bias)
        assert torch.allclose(out, expected, atol=1e-3), (
            "With gate_scores=100, output should be ≈ standard linear"
        )

    def test_extra_repr(self, small_layer):
        """extra_repr should include in_features and out_features."""
        r = small_layer.extra_repr()
        assert "16" in r and "8" in r


# ─────────────────────────────────────────────────────────────────────────────
#  Part 2 — sparsity_loss
# ─────────────────────────────────────────────────────────────────────────────

class TestSparsityLoss:

    def test_sparsity_loss_is_positive(self, small_net):
        """L1 of sigmoid values is always > 0."""
        loss = sparsity_loss(small_net)
        assert loss.item() > 0, "Sparsity loss should be positive"

    def test_sparsity_loss_is_scalar(self, small_net):
        """sparsity_loss must return a scalar tensor."""
        loss = sparsity_loss(small_net)
        assert loss.shape == torch.Size([]) or loss.numel() == 1, (
            f"Expected scalar, got shape {loss.shape}"
        )

    def test_sparsity_loss_is_differentiable(self, small_net):
        """Gradient must flow back through sparsity_loss to gate_scores."""
        loss = sparsity_loss(small_net)
        loss.backward()
        for layer in small_net.prunable_layers():
            assert layer.gate_scores.grad is not None, (
                "No gradient on gate_scores after sparsity_loss.backward()"
            )

    def test_high_gate_scores_give_higher_loss(self):
        """A network with all-positive gate_scores should have higher loss
        than one with all-negative gate_scores."""
        net_high = PrunableLinear(8, 4)
        net_low  = PrunableLinear(8, 4)
        net_high.gate_scores.data.fill_( 5.0)   # sigmoid ≈ 0.99
        net_low.gate_scores.data.fill_(-5.0)    # sigmoid ≈ 0.01

        class Wrap(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
            def forward(self, x): return self.layer(x)
            def prunable_layers(self): yield self.layer

        loss_high = sparsity_loss(Wrap(net_high))
        loss_low  = sparsity_loss(Wrap(net_low))
        assert loss_high > loss_low, (
            "Model with higher gate_scores should have larger sparsity loss"
        )

    def test_sparsity_loss_scales_with_layer_count(self):
        """Adding more PrunableLinear layers should increase the total loss."""
        class OneLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = PrunableLinear(8, 4)
            def forward(self, x): return self.l(x)
            def prunable_layers(self): yield self.l

        class TwoLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = PrunableLinear(8, 4)
                self.l2 = PrunableLinear(4, 2)
            def forward(self, x): return self.l2(self.l1(x))
            def prunable_layers(self):
                yield self.l1; yield self.l2

        # Force identical gate_scores so we can compare cleanly
        for m in [OneLayer(), TwoLayers()]:
            for layer in m.prunable_layers():
                layer.gate_scores.data.fill_(0.0)   # sigmoid=0.5

        loss_one = sparsity_loss(OneLayer())
        loss_two = sparsity_loss(TwoLayers())
        assert loss_two > loss_one, (
            "Two prunable layers should produce higher sparsity loss than one"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Compute sparsity
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSparsity:

    def test_sparsity_range(self, small_net):
        """compute_sparsity must return a value in [0, 100]."""
        s = compute_sparsity(small_net)
        assert 0.0 <= s <= 100.0, f"Sparsity {s} outside [0, 100]"

    def test_all_pruned_when_gates_zero(self):
        """With gate_scores = -100, almost all gates < 0.01 → sparsity ≈ 100%."""
        layer = PrunableLinear(8, 4)
        layer.gate_scores.data.fill_(-100.0)

        class Wrap(nn.Module):
            def __init__(self, l): super().__init__(); self.l = l
            def forward(self, x): return self.l(x)
            def prunable_layers(self): yield self.l

        s = compute_sparsity(Wrap(layer))
        assert s > 99.0, f"Expected sparsity ~100%, got {s:.2f}%"

    def test_none_pruned_when_gates_one(self):
        """With gate_scores = +100, all gates ≈ 1 → sparsity ≈ 0%."""
        layer = PrunableLinear(8, 4)
        layer.gate_scores.data.fill_(100.0)

        class Wrap(nn.Module):
            def __init__(self, l): super().__init__(); self.l = l
            def forward(self, x): return self.l(x)
            def prunable_layers(self): yield self.l

        s = compute_sparsity(Wrap(layer))
        assert s < 1.0, f"Expected sparsity ~0%, got {s:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
#  collect_all_gates
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectAllGates:

    def test_returns_numpy_array(self, small_net):
        import numpy as np
        gates = collect_all_gates(small_net)
        assert isinstance(gates, np.ndarray), "Expected numpy array"

    def test_gate_count_matches_total_weights(self, small_net):
        """Total gates collected == total weight elements across all PrunableLinear layers."""
        expected = sum(
            layer.weight.numel() for layer in small_net.prunable_layers()
        )
        gates = collect_all_gates(small_net)
        assert len(gates) == expected, (
            f"Expected {expected} gates, got {len(gates)}"
        )

    def test_all_values_in_zero_one(self, small_net):
        gates = collect_all_gates(small_net)
        assert (gates >= 0).all() and (gates <= 1).all(), (
            "Gate values outside [0, 1]"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SelfPruningNet
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfPruningNet:

    def test_output_shape(self):
        """Full network must produce (batch, 10) logits for CIFAR-10."""
        net = SelfPruningNet()
        x   = torch.randn(8, 3, 32, 32)    # standard CIFAR-10 image batch
        out = net(x)
        assert out.shape == (8, 10), f"Expected (8, 10), got {out.shape}"

    def test_output_shape_flat_input(self):
        """Also works when input is already flattened to (batch, 3072)."""
        net = SelfPruningNet()
        x   = torch.randn(4, 3072)
        out = net(x)
        assert out.shape == (4, 10)

    def test_prunable_layers_count(self):
        """There should be exactly 4 PrunableLinear layers."""
        net    = SelfPruningNet()
        layers = list(net.prunable_layers())
        assert len(layers) == 4, f"Expected 4 PrunableLinear layers, got {len(layers)}"

    def test_all_prunable_layers_are_correct_type(self):
        net = SelfPruningNet()
        for layer in net.prunable_layers():
            assert isinstance(layer, PrunableLinear), (
                f"Expected PrunableLinear, got {type(layer)}"
            )

    def test_forward_backward_full_net(self):
        """A full forward+backward pass should not raise and produce gradients."""
        net    = SelfPruningNet()
        x      = torch.randn(4, 3072)
        labels = torch.randint(0, 10, (4,))
        loss   = F.cross_entropy(net(x), labels) + 1e-4 * sparsity_loss(net)
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ─────────────────────────────────────────────────────────────────────────────
#  Integration — sparsity increases with λ
# ─────────────────────────────────────────────────────────────────────────────

class TestSparsityVsLambda:
    """
    Integration test: training with a higher λ for a few steps should
    result in a higher sparsity level than training with a lower λ.
    """

    @staticmethod
    def _mean_gate(lam: float, steps: int = 150) -> float:
        """
        Train a tiny 2-layer net for `steps` gradient steps.
        Returns the mean gate value — a proxy for sparsity that responds
        faster than the hard threshold (so the test stays quick on CPU).
        """
        torch.manual_seed(42)

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    PrunableLinear(32, 16), nn.ReLU(), PrunableLinear(16, 10)
                )
            def forward(self, x): return self.net(x)
            def prunable_layers(self):
                for m in self.modules():
                    if isinstance(m, PrunableLinear): yield m

        net = TinyNet()
        opt = torch.optim.Adam(net.parameters(), lr=5e-2)
        X   = torch.randn(64, 32)
        y   = torch.randint(0, 10, (64,))

        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(net(X), y) + lam * sparsity_loss(net)
            loss.backward()
            opt.step()

        # Mean gate value: a lower mean means gates are being pushed toward 0
        import numpy as np
        gates = collect_all_gates(net)
        return float(gates.mean())

    def test_higher_lambda_yields_lower_mean_gate(self):
        """
        Higher λ should drive gate values lower (toward 0) compared to low λ.
        We test mean gate value rather than the hard sparsity threshold so the
        test converges reliably in a small number of CPU steps.
        """
        mean_low  = self._mean_gate(lam=1e-6)
        mean_high = self._mean_gate(lam=5e-2)
        assert mean_high < mean_low, (
            f"Higher λ should produce lower mean gate: "
            f"low_λ mean={mean_low:.4f}  high_λ mean={mean_high:.4f}"
        )

    def test_mean_gate_is_non_negative(self):
        mean = self._mean_gate(lam=1e-4)
        assert mean >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  evaluate() helper
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_accuracy_range(self):
        """evaluate() must return a value in [0, 100]."""
        net = SelfPruningNet()
        X   = torch.randn(32, 3072)
        y   = torch.randint(0, 10, (32,))
        dl  = DataLoader(TensorDataset(X, y), batch_size=16)
        acc = evaluate(net, dl, torch.device("cpu"))
        assert 0.0 <= acc <= 100.0, f"Accuracy {acc} outside [0, 100]"

    def test_all_correct_gives_100(self):
        """If logits always point to the true class, accuracy should be 100%."""
        net = SelfPruningNet()

        # Monkey-patch forward to return perfect one-hot logits
        def perfect_forward(x):
            y_true = x[:, 0].long() % 10          # encode label in first column
            logits = torch.zeros(x.size(0), 10)
            logits.scatter_(1, y_true.unsqueeze(1), 100.0)
            return logits

        net.forward = perfect_forward

        labels = torch.randint(0, 10, (32,))
        X      = torch.zeros(32, 3072)
        X[:, 0] = labels.float()                  # encode label into input
        dl  = DataLoader(TensorDataset(X, labels), batch_size=16)
        acc = evaluate(net, dl, torch.device("cpu"))
        assert abs(acc - 100.0) < 1e-4, f"Expected 100%, got {acc:.2f}%"
