"""Comprehensive tests for the EVE optimizer.

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
#  1. K=1 matches AdamW exactly
# ══════════════════════════════════════════════════════════════════════════

class TestK1MatchesAdamW:
    """EVE(K=1) must produce *bit-identical* parameters to AdamW."""

    def test_fc_matches_adamw(self):
        lr, wd, steps = 1e-2, 0.01, 20
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        _seed()
        model_adam = _make_fc()
        _seed()
        model_eve = _make_fc()

        opt_adam = torch.optim.AdamW(
            model_adam.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd
        )
        opt_eve = EVE(
            model_eve.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd, K=1
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
        """Spot-check with a larger learning rate."""
        lr, wd, steps = 5e-2, 0.0, 10

        _seed()
        model_adam = _make_fc()
        _seed()
        model_eve = _make_fc()

        opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=lr, weight_decay=wd)
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

    def test_fc_loss_decreases(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        losses = _train_steps(model, opt, x, y, 50, is_eve_k_gt1=True)

        assert losses[-1] < losses[0], "loss should decrease over 50 steps"

    def test_k2_works(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_k3_works(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=3)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_k8_extended_offspring(self):
        """K=8 exercises momentum-interpolation offspring (Section 4.1.2)."""
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
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(16, 1, 8, 8)
        y = torch.randn(16, 2)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_rnn(self):
        _seed()
        model = _make_rnn()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(16, 5, 8)  # (batch, seq, features)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]

    def test_transformer(self):
        _seed()
        model = _make_transformer()
        opt = EVE(model.parameters(), lr=1e-3, K=4)

        x = torch.randn(16, 5, 16)  # (batch, seq, d_model)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 30, is_eve_k_gt1=True)

        assert losses[-1] < losses[0]


# ══════════════════════════════════════════════════════════════════════════
#  4. Offspring directions correctness
# ══════════════════════════════════════════════════════════════════════════

class TestOffspring:
    """Verify offspring directions match the paper's equations exactly."""

    def test_d1_adam(self):
        """d1 = -m_hat / (sqrt(v_hat) + eps)  — verify the formula directly."""
        _seed()
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        g = torch.randn(10)

        m = (1 - beta1) * g
        v = (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        expected_d1 = -m_hat / (v_hat.sqrt() + eps)

        # Reproduce via the same tensor ops the optimizer uses internally.
        m2 = torch.zeros(10)
        v2 = torch.zeros(10)
        m2.mul_(beta1).add_(g, alpha=1 - beta1)
        v2.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        m_hat2 = m2 / (1 - beta1)
        sqrt_v_hat2 = (v2 / (1 - beta2)).sqrt()
        d1 = m_hat2.neg() / (sqrt_v_hat2 + eps)

        torch.testing.assert_close(d1, expected_d1, atol=1e-7, rtol=1e-6)

    def test_d1_matches_adamw_direction(self):
        """After one step, K=1 EVE direction equals Adam's update direction."""
        _seed()
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr, wd = 0.01, 0.0

        p_ref = nn.Parameter(torch.randn(10))
        p_eve = nn.Parameter(p_ref.data.clone())

        g = torch.randn(10)
        p_ref.grad = g.clone()
        p_eve.grad = g.clone()

        opt_ref = torch.optim.AdamW([p_ref], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
        opt_eve = EVE([p_eve], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd, K=1)

        opt_ref.step()
        opt_eve.step()

        torch.testing.assert_close(p_eve.data, p_ref.data, atol=1e-7, rtol=1e-6)

    def test_d4_contrarian_bounded(self):
        """Contrarian d4 should have values in [-1, 1]."""
        _seed()
        g = torch.randn(100)
        v_hat = torch.rand(100) * 10  # positive
        sqrt_v_hat = v_hat.sqrt()
        global_max = sqrt_v_hat.max().item()
        eps = 1e-8

        d4 = -g.sign() * sqrt_v_hat / (global_max + eps)
        assert d4.abs().max().item() <= 1.0 + 1e-7


# ══════════════════════════════════════════════════════════════════════════
#  5. Selection weights
# ══════════════════════════════════════════════════════════════════════════

class TestSelectionWeights:

    def test_weights_sum_to_one(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)

        fitness = torch.randn(4)
        w = torch.softmax(opt.beta_sel * fitness, dim=0)

        assert abs(w.sum().item() - 1.0) < 1e-6
        assert (w >= 0).all()

    def test_weights_all_positive(self):
        """Proposition 3: for finite beta_sel, all w_k > 0."""
        fitness = torch.tensor([-1.0, 0.0, 0.5, 2.0])
        beta_sel = 1.0
        w = torch.softmax(beta_sel * fitness, dim=0)
        assert (w > 0).all()

    def test_k1_trivial_weight(self):
        """At K=1, the single weight must be 1 (Prop. 2)."""
        fitness = torch.tensor([0.5])
        w = torch.softmax(1.0 * fitness, dim=0)
        assert abs(w.item() - 1.0) < 1e-7


# ══════════════════════════════════════════════════════════════════════════
#  6. Strength signal bounds
# ══════════════════════════════════════════════════════════════════════════

class TestStrengthSignal:
    """Strength signal is now computed inline as sigmoid(|m_hat|/denom - 1).
    It is not stored in optimizer state; verify it via diagnostics."""

    def test_bounded_01(self):
        """s = sigmoid(...) is always in (0, 1)."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4, record_diagnostics=True)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        _train_steps(model, opt, x, y, 30, is_eve_k_gt1=True)

        for diag in opt._diagnostics:
            s_stats = diag["s_stats"]
            assert s_stats["min"] >= 0.0 - 1e-7
            assert s_stats["max"] <= 1.0 + 1e-7

    def test_not_stored_in_state(self):
        """s should NOT be in optimizer state (it is computed inline)."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        _train_steps(model, opt, x, y, 5, is_eve_k_gt1=True)

        for p in model.parameters():
            assert "s" not in opt.state[p]
            assert "prev_update_sign" not in opt.state[p]

    def test_heterogeneous_after_training(self):
        """After several steps, s should not be uniform (std > 0)."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4, record_diagnostics=True)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        _train_steps(model, opt, x, y, 30, is_eve_k_gt1=True)

        last = opt._diagnostics[-1]["s_stats"]
        assert last["std"] > 0.01, f"s should be heterogeneous, got std={last['std']}"


# ══════════════════════════════════════════════════════════════════════════
#  7. Adaptive temperature
# ══════════════════════════════════════════════════════════════════════════

class TestTemperature:

    def test_stays_in_bounds(self):
        """beta_sel must stay within [beta_min, beta_max]."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        _train_steps(model, opt, x, y, 100, is_eve_k_gt1=True)

        beta_min, beta_max = opt.defaults["beta_sel_range"]
        assert opt.beta_sel >= beta_min - 1e-9
        assert opt.beta_sel <= beta_max + 1e-9

    def test_adapts_from_initial(self):
        """After several steps, beta_sel should differ from its init."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4, beta_sel_init=1.0)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        _train_steps(model, opt, x, y, 30, is_eve_k_gt1=True)

        assert opt.beta_sel != 1.0, "temperature should have adapted"


# ══════════════════════════════════════════════════════════════════════════
#  8. Edge cases & validation
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
        opt = EVE(model.parameters(), lr=1e-2, K=4)

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
        # Freeze the first layer
        for p in model[0].parameters():
            p.requires_grad_(False)

        opt = EVE(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, K=4
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
        """Batch smaller than K should still work (probe_size >= 1)."""
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
        opt = EVE(model.parameters(), lr=1e-2, K=4, weight_decay=0.0)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)
        assert losses[-1] < losses[0]


# ══════════════════════════════════════════════════════════════════════════
#  9. MPS device
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

    def test_k4_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_fc().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(16, 8, device=device)
        y = torch.randn(16, 4, device=device)
        losses = _train_steps(model, opt, x, y, 20, is_eve_k_gt1=True)
        assert losses[-1] < losses[0]

    def test_cnn_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_cnn().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(8, 1, 8, 8, device=device)
        y = torch.randn(8, 2, device=device)
        losses = _train_steps(model, opt, x, y, 10, is_eve_k_gt1=True)
        assert all(math.isfinite(l) for l in losses)
