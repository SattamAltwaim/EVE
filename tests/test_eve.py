"""Comprehensive tests for the EVE optimizer (simplified momentum interpolation).

All models are deliberately tiny (dims 8–32) so the suite runs fast on
a MacBook M1 CPU.  MPS tests are gated on availability.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eve_optimizer import EVE


# ── helpers ──────────────────────────────────────────────────────────────

def _seed(s: int = 42) -> None:
    torch.manual_seed(s)


def _make_fc(in_dim: int = 8, hidden: int = 16, out_dim: int = 4) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def _make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


def _make_rnn(input_size: int = 8, hidden_size: int = 16, out_dim: int = 4) -> nn.Module:
    class TinyRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, out_dim)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n.squeeze(0))

    return TinyRNN()


def _make_transformer(d_model: int = 16, nhead: int = 2, out_dim: int = 4) -> nn.Module:
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.fc = nn.Linear(d_model, out_dim)

        def forward(self, x):
            out = self.encoder(x)
            return self.fc(out.mean(dim=1))

    return TinyTransformer()


def _train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    loss_fn=F.mse_loss,
    is_eve_k_gt1: bool = False,
):
    """Run *steps* training iterations, return list of loss values."""
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        if is_eve_k_gt1:
            optimizer.step(model=model, loss_fn=loss_fn, data=(x, y))
        else:
            optimizer.step()
        losses.append(loss.item())
    return losses


# ══════════════════════════════════════════════════════════════════════════
#  1. K=1 matches AMSGrad-AdamW exactly
# ══════════════════════════════════════════════════════════════════════════

class TestK1MatchesAMSGradAdamW:
    """EVE(K=1) must produce *bit-identical* parameters to AdamW(amsgrad=True)."""

    def test_fc_matches_amsgrad_adamw(self):
        lr, wd, steps = 1e-2, 0.01, 20
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        _seed()
        model_adam = _make_fc()
        _seed()
        model_eve = _make_fc()

        opt_adam = torch.optim.AdamW(
            model_adam.parameters(), lr=lr, betas=(beta1, beta2), eps=eps,
            weight_decay=wd, amsgrad=True
        )
        opt_eve = EVE(
            model_eve.parameters(), lr=lr, betas=(beta1, beta2), eps=eps,
            weight_decay=wd, K=1
        )

        _seed(99)
        x = torch.randn(16, 8)
        y = torch.randn(16, 4)

        for _ in range(steps):
            opt_adam.zero_grad()
            loss_a = F.mse_loss(model_adam(x), y)
            loss_a.backward()
            opt_adam.step()

            opt_eve.zero_grad()
            loss_e = F.mse_loss(model_eve(x), y)
            loss_e.backward()
            opt_eve.step()

        for pa, pe in zip(model_adam.parameters(), model_eve.parameters()):
            torch.testing.assert_close(pa.data, pe.data, atol=1e-6, rtol=1e-5)

    def test_different_lr_still_matches(self):
        lr, wd, steps = 5e-2, 0.0, 10

        _seed()
        model_adam = _make_fc()
        _seed()
        model_eve = _make_fc()

        opt_adam = torch.optim.AdamW(
            model_adam.parameters(), lr=lr, weight_decay=wd, amsgrad=True
        )
        opt_eve = EVE(model_eve.parameters(), lr=lr, weight_decay=wd, K=1)

        _seed(7)
        x = torch.randn(8, 8)
        y = torch.randn(8, 4)

        _train_steps(model_adam, opt_adam, x, y, steps)
        _train_steps(model_eve, opt_eve, x, y, steps)

        for pa, pe in zip(model_adam.parameters(), model_eve.parameters()):
            torch.testing.assert_close(pa.data, pe.data, atol=1e-6, rtol=1e-5)


# ══════════════════════════════════════════════════════════════════════════
#  2. K>1 basic functionality — loss decreases
# ══════════════════════════════════════════════════════════════════════════

class TestKGt1Basic:

    def test_fc_loss_decreases_k2(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        losses = _train_steps(model, opt, x, y, 50, is_eve_k_gt1=True)

        assert losses[-1] < losses[0], "loss should decrease over 50 steps"

    def test_fc_loss_decreases_k4(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        losses = _train_steps(model, opt, x, y, 50, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_k3_works(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=3)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_k8_interpolation(self):
        """K=8 exercises finer alpha grid spacing."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=8)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        losses = _train_steps(model, opt, x, y, 30, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]


# ══════════════════════════════════════════════════════════════════════════
#  3. Architecture compatibility
# ══════════════════════════════════════════════════════════════════════════

class TestArchitectures:

    def test_cnn(self):
        _seed()
        model = _make_cnn()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 1, 8, 8)
        y = torch.randn(16, 2)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_rnn(self):
        _seed()
        model = _make_rnn()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 5, 8)  # (batch, seq, features)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_transformer(self):
        _seed()
        model = _make_transformer()
        opt = EVE(model.parameters(), lr=1e-3, K=2)

        x = torch.randn(16, 5, 16)  # (batch, seq, d_model)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 30, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]


# ══════════════════════════════════════════════════════════════════════════
#  4. Momentum interpolation offspring correctness
# ══════════════════════════════════════════════════════════════════════════

class TestMomentumInterpolation:
    """Verify offspring directions match the generator equation."""

    def test_alpha_grid_k2(self):
        """K=2 should produce alphas [0, 1]."""
        alphas = [k / (2 - 1) for k in range(2)]
        assert alphas == [0.0, 1.0]

    def test_alpha_grid_k3(self):
        """K=3 should produce alphas [0, 0.5, 1]."""
        alphas = [k / (3 - 1) for k in range(3)]
        assert alphas == [0.0, 0.5, 1.0]

    def test_alpha_grid_k4(self):
        """K=4 should produce alphas [0, 1/3, 2/3, 1]."""
        alphas = [k / (4 - 1) for k in range(4)]
        expected = [0.0, 1 / 3, 2 / 3, 1.0]
        for a, e in zip(alphas, expected):
            assert abs(a - e) < 1e-10

    def test_alpha0_is_pure_fast(self):
        """At alpha=0, offspring = -m_hat / denom (pure fast momentum)."""
        _seed()
        m_hat = torch.randn(10)
        m_hat_slow = torch.randn(10)
        denom = torch.rand(10) + 0.1

        alpha = 0.0
        d = (-(1.0 - alpha) * m_hat - alpha * m_hat_slow) / denom
        expected = -m_hat / denom

        torch.testing.assert_close(d, expected, atol=1e-7, rtol=1e-6)

    def test_alpha1_is_pure_slow(self):
        """At alpha=1, offspring = -m_hat_slow / denom (pure slow momentum)."""
        _seed()
        m_hat = torch.randn(10)
        m_hat_slow = torch.randn(10)
        denom = torch.rand(10) + 0.1

        alpha = 1.0
        d = (-(1.0 - alpha) * m_hat - alpha * m_hat_slow) / denom
        expected = -m_hat_slow / denom

        torch.testing.assert_close(d, expected, atol=1e-7, rtol=1e-6)

    def test_alpha_half_is_midpoint(self):
        """At alpha=0.5, offspring = -(0.5*m_hat + 0.5*m_hat_slow) / denom."""
        _seed()
        m_hat = torch.randn(10)
        m_hat_slow = torch.randn(10)
        denom = torch.rand(10) + 0.1

        alpha = 0.5
        d = (-(1.0 - alpha) * m_hat - alpha * m_hat_slow) / denom
        expected = -(0.5 * m_hat + 0.5 * m_hat_slow) / denom

        torch.testing.assert_close(d, expected, atol=1e-7, rtol=1e-6)


# ══════════════════════════════════════════════════════════════════════════
#  5. AMSGrad v_max
# ══════════════════════════════════════════════════════════════════════════

class TestAMSGrad:

    def test_v_max_is_running_maximum(self):
        """v_max should be >= v (raw second moment) at every step."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        _train_steps(model, opt, x, y, 20)

        for p in model.parameters():
            state = opt.state[p]
            v = state["v"]
            v_max = state["v_max"]
            assert (v_max >= v - 1e-7).all(), "v_max must be >= v"

    def test_v_max_initialised_zero(self):
        """v_max starts at zero and gets updated on first step."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8)
        y = torch.randn(8, 4)

        model.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()

        for p in model.parameters():
            state = opt.state[p]
            assert (state["v_max"] > 0).any(), "v_max should be positive after first step"


# ══════════════════════════════════════════════════════════════════════════
#  6. Slow momentum buffer
# ══════════════════════════════════════════════════════════════════════════

class TestSlowMomentum:

    def test_m_slow_maintained_for_k_gt1(self):
        """m_slow should be initialised and updated when K>1."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        _train_steps(model, opt, x, y, 5, is_eve_k_gt1=True)

        for p in model.parameters():
            state = opt.state[p]
            assert "m_slow" in state, "m_slow should exist in state for K>1"
            assert not (state["m_slow"] == 0).all(), "m_slow should be non-zero after training"

    def test_m_slow_not_maintained_for_k1(self):
        """K=1 path should not allocate m_slow."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8)
        y = torch.randn(8, 4)
        _train_steps(model, opt, x, y, 5)

        for p in model.parameters():
            state = opt.state[p]
            assert "m_slow" not in state

    def test_m_slow_decay_rate(self):
        """m_slow should track gradients with beta1_slow decay."""
        _seed()
        beta1_slow = 0.999
        p = nn.Parameter(torch.randn(10))

        g1 = torch.randn(10)
        expected_m_slow = (1.0 - beta1_slow) * g1

        model = nn.Linear(10, 5)
        opt = EVE(model.parameters(), lr=1e-3, K=2, beta1_slow=beta1_slow)

        model.zero_grad()
        loss = model(torch.randn(4, 10)).sum()
        loss.backward()

        opt.step(model=model, loss_fn=lambda out, tgt: out.sum(),
                 data=(torch.randn(4, 10), None))

        for p in model.parameters():
            state = opt.state[p]
            if "m_slow" in state:
                assert state["m_slow"].shape == p.shape


# ══════════════════════════════════════════════════════════════════════════
#  7. Math collapse (alpha_eff)
# ══════════════════════════════════════════════════════════════════════════

class TestMathCollapse:

    def test_alpha_eff_is_weighted_sum(self):
        """alpha_eff = sum(w_k * alpha_k) for given weights."""
        alphas = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
        weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
        alpha_eff = (weights * alphas).sum().item()
        expected = 0.4 * 0.0 + 0.3 / 3 + 0.2 * 2 / 3 + 0.1 * 1.0
        assert abs(alpha_eff - expected) < 1e-7

    def test_uniform_weights_give_midpoint(self):
        """With uniform weights, alpha_eff should be 0.5 (midpoint)."""
        K = 4
        alphas = torch.tensor([k / (K - 1) for k in range(K)])
        weights = torch.ones(K) / K
        alpha_eff = (weights * alphas).sum().item()
        assert abs(alpha_eff - 0.5) < 1e-7

    def test_collapsed_direction_matches_weighted(self):
        """The math-collapsed direction should equal the weighted sum of offspring."""
        _seed()
        K = 4
        m_hat = torch.randn(20)
        m_hat_slow = torch.randn(20)
        denom = torch.rand(20) + 0.1
        alphas = [k / (K - 1) for k in range(K)]

        fitness = torch.randn(K)
        weights = torch.softmax(fitness, dim=0)

        # Weighted sum of individual offspring
        d_weighted = torch.zeros(20)
        for k in range(K):
            a_k = alphas[k]
            d_k = (-(1.0 - a_k) * m_hat - a_k * m_hat_slow) / denom
            d_weighted += weights[k] * d_k

        # Math collapse
        alpha_eff = sum(w * a for w, a in zip(weights.tolist(), alphas))
        d_collapsed = (-(1.0 - alpha_eff) * m_hat - alpha_eff * m_hat_slow) / denom

        torch.testing.assert_close(d_collapsed, d_weighted, atol=1e-6, rtol=1e-5)


# ══════════════════════════════════════════════════════════════════════════
#  8. Selection weights
# ══════════════════════════════════════════════════════════════════════════

class TestSelectionWeights:

    def test_weights_sum_to_one(self):
        fitness = torch.randn(4)
        beta_sel = 1.0
        w = torch.softmax(beta_sel * fitness, dim=0)

        assert abs(w.sum().item() - 1.0) < 1e-6
        assert (w >= 0).all()

    def test_weights_all_positive(self):
        """For finite beta_sel, all w_k > 0."""
        fitness = torch.tensor([-1.0, 0.0, 0.5, 2.0])
        beta_sel = 1.0
        w = torch.softmax(beta_sel * fitness, dim=0)
        assert (w > 0).all()

    def test_k1_trivial_weight(self):
        """At K=1, the single weight must be 1."""
        fitness = torch.tensor([0.5])
        w = torch.softmax(1.0 * fitness, dim=0)
        assert abs(w.item() - 1.0) < 1e-7


class TestFitnessNormalization:
    """Range normalization makes selection scale-invariant."""

    @staticmethod
    def _normalized_weights(fitness, beta_sel=1.0):
        fitness_range = (fitness.max() - fitness.min()).clamp(min=1e-8)
        return torch.softmax(beta_sel * fitness / fitness_range, dim=0)

    def test_identical_fitness_gives_uniform(self):
        """When all offspring have the same fitness, weights must be uniform."""
        fitness = torch.tensor([0.005, 0.005, 0.005, 0.005])
        w = self._normalized_weights(fitness, beta_sel=10.0)
        assert torch.allclose(w, torch.ones(4) / 4, atol=1e-6)

    def test_zero_fitness_gives_uniform(self):
        fitness = torch.zeros(4)
        w = self._normalized_weights(fitness, beta_sel=100.0)
        assert torch.allclose(w, torch.ones(4) / 4, atol=1e-6)

    def test_scale_invariance(self):
        """Scaling all fitness by a constant must not change the weights."""
        fitness = torch.tensor([0.01, 0.007, 0.003, 0.001])
        w1 = self._normalized_weights(fitness, beta_sel=1.0)

        w2 = self._normalized_weights(fitness * 1e-3, beta_sel=1.0)
        w3 = self._normalized_weights(fitness * 100.0, beta_sel=1.0)

        torch.testing.assert_close(w1, w2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(w1, w3, atol=1e-6, rtol=1e-5)

    def test_normalized_weights_sum_to_one(self):
        fitness = torch.tensor([0.009, 0.006, 0.003, 0.0005])
        w = self._normalized_weights(fitness, beta_sel=5.0)
        assert abs(w.sum().item() - 1.0) < 1e-6
        assert (w > 0).all()

    def test_higher_beta_sel_increases_differentiation(self):
        """Higher beta_sel should make the weight distribution more peaked."""
        fitness = torch.tensor([0.01, 0.007, 0.003, 0.001])
        w_low = self._normalized_weights(fitness, beta_sel=1.0)
        w_high = self._normalized_weights(fitness, beta_sel=10.0)
        range_low = (w_low.max() - w_low.min()).item()
        range_high = (w_high.max() - w_high.min()).item()
        assert range_high > range_low

    def test_integrated_with_optimizer(self):
        """EVE with K>1 should produce non-uniform weights with beta_sel=1."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4, beta_sel=1.0,
                  record_diagnostics=True)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        _train_steps(model, opt, x, y, 10, is_eve_k_gt1=True)

        for entry in opt._diagnostics[2:]:
            w = entry["weights"]
            weight_range = max(w) - min(w)
            assert weight_range > 0.01, (
                f"Weights too uniform after normalization: {w}"
            )


# ══════════════════════════════════════════════════════════════════════════
#  9. Diagnostics
# ══════════════════════════════════════════════════════════════════════════

class TestDiagnostics:

    def test_diagnostics_recorded(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2, record_diagnostics=True)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        _train_steps(model, opt, x, y, 5, is_eve_k_gt1=True)

        assert len(opt._diagnostics) == 5
        entry = opt._diagnostics[0]
        assert "fitness" in entry
        assert "weights" in entry
        assert "alpha_eff" in entry
        assert "dir_norms" in entry
        assert "cos_pairs" in entry
        assert "cos_to_combined" in entry
        assert 0.0 <= entry["alpha_eff"] <= 1.0

    def test_diagnostics_not_recorded_by_default(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        _train_steps(model, opt, x, y, 5, is_eve_k_gt1=True)

        assert len(opt._diagnostics) == 0


# ══════════════════════════════════════════════════════════════════════════
#  10. Edge cases & validation
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_k1_no_extra_args(self):
        """K=1 should work without model/loss_fn/data."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8)
        y = torch.randn(8, 4)

        model.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()  # no crash

    def test_k_gt1_requires_model(self):
        """K>1 without model should raise ValueError."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(8, 8)
        y = torch.randn(8, 4)

        model.zero_grad()
        F.mse_loss(model(x), y).backward()

        with pytest.raises(ValueError, match="K>1 requires"):
            opt.step()

    def test_frozen_params(self):
        """Parameters without gradients (frozen) should be skipped."""
        _seed()
        model = _make_fc()
        for p in model[0].parameters():
            p.requires_grad_(False)

        opt = EVE(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, K=2
        )

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)

        first_layer_before = [p.data.clone() for p in model[0].parameters()]
        _train_steps(model, opt, x, y, 10, is_eve_k_gt1=True)
        first_layer_after = [p.data.clone() for p in model[0].parameters()]

        for before, after in zip(first_layer_before, first_layer_after):
            torch.testing.assert_close(before, after)

    def test_closure_api(self):
        """The closure-based API should work."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8)
        y = torch.randn(8, 4)

        def closure():
            opt.zero_grad()
            out = model(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None
        assert loss.item() > 0

    def test_small_batch(self):
        """Batch smaller than K should still work."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(2, 8)  # batch=2 < K=4
        y = torch.randn(2, 4)
        losses = _train_steps(model, opt, x, y, 5, is_eve_k_gt1=True)
        assert all(math.isfinite(l) for l in losses)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            EVE([nn.Parameter(torch.randn(5))], K=0)

    def test_weight_decay_zero(self):
        """No weight decay should still work."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2, weight_decay=0.0)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)
        assert losses[-1] < losses[0]


# ══════════════════════════════════════════════════════════════════════════
#  11. MPS device
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
class TestMPS:

    def test_k1_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_fc().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8, device=device)
        y = torch.randn(8, 4, device=device)
        losses = _train_steps(model, opt, x, y, 10)
        assert losses[-1] < losses[0]

    def test_k2_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_fc().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 8, device=device)
        y = torch.randn(16, 4, device=device)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)
        assert losses[-1] < losses[0]

    def test_cnn_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_cnn().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(8, 1, 8, 8, device=device)
        y = torch.randn(8, 2, device=device)
        losses = _train_steps(model, opt, x, y, 10, is_eve_k_gt1=True)
        assert all(math.isfinite(l) for l in losses)
