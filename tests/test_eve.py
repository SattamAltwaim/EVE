"""Tests for the EVE optimizer (next-batch probing variant).

All models are deliberately tiny so the suite runs fast on CPU.
MPS tests are gated on availability.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eve_optimizer import EVE


# -- helpers ---------------------------------------------------------------

def _seed(s: int = 42) -> None:
    torch.manual_seed(s)


def _make_fc(in_dim: int = 8, hidden: int = 16, out_dim: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim),
    )


def _make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4),
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


def _train_steps_k1(model, optimizer, x, y, steps, loss_fn=F.mse_loss):
    """K=1 training loop (no probe needed)."""
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _make_probe_fn(model, x_probe, y_probe, loss_fn=F.mse_loss):
    """Build a probe closure over the next batch."""
    def probe_fn():
        with torch.no_grad():
            return loss_fn(model(x_probe), y_probe)
    return probe_fn


def _train_steps_k2(model, optimizer, x, y, x_probe, y_probe, steps, loss_fn=F.mse_loss):
    """K>=2 training loop with probing."""
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        probe = _make_probe_fn(model, x_probe, y_probe, loss_fn)
        optimizer.step(probe_fn=probe, model=model)
        losses.append(loss.item())
    return losses


# ==========================================================================
#  1. K=1 matches AdamW exactly
# ==========================================================================

class TestK1MatchesAdamW:
    """EVE(K=1) must produce identical parameters to AdamW."""

    def test_fc_matches_adamw(self):
        lr, wd, steps = 1e-2, 0.01, 20
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        _seed()
        model_adam = _make_fc()
        _seed()
        model_eve = _make_fc()

        opt_adam = torch.optim.AdamW(
            model_adam.parameters(), lr=lr, betas=(beta1, beta2),
            eps=eps, weight_decay=wd,
        )
        opt_eve = EVE(
            model_eve.parameters(), lr=lr, betas=(beta1, beta2),
            eps=eps, weight_decay=wd, K=1,
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

        opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=lr, weight_decay=wd)
        opt_eve = EVE(model_eve.parameters(), lr=lr, weight_decay=wd, K=1)

        _seed(7)
        x = torch.randn(8, 8)
        y = torch.randn(8, 4)

        _train_steps_k1(model_adam, opt_adam, x, y, steps)
        _train_steps_k1(model_eve, opt_eve, x, y, steps)

        for pa, pe in zip(model_adam.parameters(), model_eve.parameters()):
            torch.testing.assert_close(pa.data, pe.data, atol=1e-6, rtol=1e-5)


# ==========================================================================
#  2. K>=2 probing -- loss decreases
# ==========================================================================

class TestProbing:

    def test_k2_loss_decreases(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        x_probe = torch.randn(32, 8)
        y_probe = torch.randn(32, 4)

        losses = _train_steps_k2(model, opt, x, y, x_probe, y_probe, 50)
        assert losses[-1] < losses[0], "loss should decrease over 50 steps with K=2"

    def test_k4_loss_decreases(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        x_probe = torch.randn(32, 8)
        y_probe = torch.randn(32, 4)

        losses = _train_steps_k2(model, opt, x, y, x_probe, y_probe, 50)
        assert losses[-1] < losses[0], "loss should decrease over 50 steps with K=4"

    def test_k1_loss_decreases(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        losses = _train_steps_k1(model, opt, x, y, 50)
        assert losses[-1] < losses[0]


# ==========================================================================
#  3. Winner selection correctness
# ==========================================================================

class TestWinnerSelection:

    def test_winner_is_argmin(self):
        """The winner index must correspond to the lowest probe loss."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4, record_diagnostics=True)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        x_probe = torch.randn(16, 8)
        y_probe = torch.randn(16, 4)

        _train_steps_k2(model, opt, x, y, x_probe, y_probe, 10)

        for rec in opt.diagnostics.records:
            losses = rec.offspring_losses
            assert rec.winner_idx == losses.index(min(losses))


# ==========================================================================
#  4. Elite inheritance
# ==========================================================================

class TestEliteInheritance:

    def test_elite_updates_after_step(self):
        """After a K>=2 step, elite should be non-zero."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        x_probe = torch.randn(16, 8)
        y_probe = torch.randn(16, 4)

        # Before step: elite is zero
        for grp in opt.param_groups:
            for p in grp["params"]:
                if p in opt.state and "elite" in opt.state[p]:
                    assert torch.all(opt.state[p]["elite"] == 0)

        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        probe = _make_probe_fn(model, x_probe, y_probe)
        opt.step(probe_fn=probe, model=model)

        # After step: at least one elite is non-zero
        any_nonzero = False
        for grp in opt.param_groups:
            for p in grp["params"]:
                if p in opt.state and "elite" in opt.state[p]:
                    if opt.state[p]["elite"].abs().sum() > 0:
                        any_nonzero = True
        assert any_nonzero, "elite should be non-zero after first step"

    def test_elite_starts_zero(self):
        """On step 1 with K>=2, all blends collapse toward Adam since elite=0."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=4, record_diagnostics=True)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        x_probe = torch.randn(16, 8)
        y_probe = torch.randn(16, 4)

        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        probe = _make_probe_fn(model, x_probe, y_probe)
        opt.step(probe_fn=probe, model=model)

        # All offspring losses should be similar on step 1 (elite=0 means
        # all blends are close to Adam direction).
        rec = opt.diagnostics.records[0]
        losses = rec.offspring_losses
        spread = max(losses) - min(losses)
        mean_loss = sum(losses) / len(losses)
        assert spread / (mean_loss + 1e-12) < 0.5, (
            f"On step 1 with elite=0, offspring should be similar. "
            f"Spread={spread:.4f}, mean={mean_loss:.4f}"
        )


# ==========================================================================
#  5. Architecture compatibility
# ==========================================================================

class TestArchitectures:

    def test_cnn_with_batchnorm(self):
        _seed()
        model = _make_cnn()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 1, 8, 8)
        y = torch.randn(16, 2)
        x_probe = torch.randn(16, 1, 8, 8)
        y_probe = torch.randn(16, 2)

        losses = _train_steps_k2(model, opt, x, y, x_probe, y_probe, 20)
        assert losses[-1] < losses[0]

    def test_rnn(self):
        _seed()
        model = _make_rnn()
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 5, 8)
        y = torch.randn(16, 4)
        x_probe = torch.randn(16, 5, 8)
        y_probe = torch.randn(16, 4)

        losses = _train_steps_k2(model, opt, x, y, x_probe, y_probe, 20)
        assert losses[-1] < losses[0]

    def test_cnn_k1(self):
        _seed()
        model = _make_cnn()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(16, 1, 8, 8)
        y = torch.randn(16, 2)
        losses = _train_steps_k1(model, opt, x, y, 20)
        assert losses[-1] < losses[0]


# ==========================================================================
#  6. Diagnostics recording
# ==========================================================================

class TestDiagnostics:

    def test_diagnostics_recorded(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2, record_diagnostics=True)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        x_probe = torch.randn(16, 8)
        y_probe = torch.randn(16, 4)

        _train_steps_k2(model, opt, x, y, x_probe, y_probe, 5)

        assert len(opt.diagnostics.records) == 5
        for rec in opt.diagnostics.records:
            assert rec.step > 0
            assert 0 <= rec.winner_idx < 2
            assert len(rec.offspring_losses) == 2
            assert math.isfinite(rec.elite_adam_cosine)
            assert math.isfinite(rec.elite_norm)
            assert math.isfinite(rec.adam_norm)

    def test_diagnostics_off_by_default(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8)
        y = torch.randn(8, 4)
        _train_steps_k1(model, opt, x, y, 3)
        assert opt.diagnostics is None

    def test_winner_counts(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=2, record_diagnostics=True)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        x_probe = torch.randn(16, 8)
        y_probe = torch.randn(16, 4)

        _train_steps_k2(model, opt, x, y, x_probe, y_probe, 10)

        counts = opt.diagnostics.winner_counts()
        total = sum(counts.values())
        assert total == 10
        for idx in counts:
            assert idx in (0, 1)

    def test_to_dataframe(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=3, record_diagnostics=True)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        x_probe = torch.randn(16, 8)
        y_probe = torch.randn(16, 4)

        _train_steps_k2(model, opt, x, y, x_probe, y_probe, 5)

        df = opt.diagnostics.to_dataframe()
        assert len(df) == 5
        assert "step" in df.columns
        assert "winner_idx" in df.columns
        assert "loss_k0" in df.columns
        assert "loss_k1" in df.columns
        assert "loss_k2" in df.columns


# ==========================================================================
#  7. Edge cases & validation
# ==========================================================================

class TestEdgeCases:

    def test_frozen_params(self):
        _seed()
        model = _make_fc()
        for p in model[0].parameters():
            p.requires_grad_(False)

        opt = EVE(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-2, K=1,
        )

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)

        first_layer_before = [p.data.clone() for p in model[0].parameters()]
        _train_steps_k1(model, opt, x, y, 10)
        first_layer_after = [p.data.clone() for p in model[0].parameters()]

        for before, after in zip(first_layer_before, first_layer_after):
            torch.testing.assert_close(before, after)

    def test_closure_api(self):
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

    def test_invalid_K(self):
        with pytest.raises(ValueError):
            EVE([nn.Parameter(torch.randn(5))], K=0)
        with pytest.raises(ValueError):
            EVE([nn.Parameter(torch.randn(5))], K=-1)

    def test_weight_decay_zero(self):
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1, weight_decay=0.0)

        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        losses = _train_steps_k1(model, opt, x, y, 20)
        assert losses[-1] < losses[0]

    def test_k1_ignores_probe_fn(self):
        """K=1 should work even if probe_fn is passed (it's ignored)."""
        _seed()
        model = _make_fc()
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8)
        y = torch.randn(8, 4)

        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()

        called = [False]
        def probe_fn():
            called[0] = True
            return torch.tensor(0.0)

        opt.step(probe_fn=probe_fn)
        assert not called[0], "K=1 should not call probe_fn"


# ==========================================================================
#  8. MPS device
# ==========================================================================

@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available",
)
class TestMPS:

    def test_k1_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_fc().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=1)

        x = torch.randn(8, 8, device=device)
        y = torch.randn(8, 4, device=device)
        losses = _train_steps_k1(model, opt, x, y, 10)
        assert losses[-1] < losses[0]

    def test_k2_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_fc().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(16, 8, device=device)
        y = torch.randn(16, 4, device=device)
        x_probe = torch.randn(16, 8, device=device)
        y_probe = torch.randn(16, 4, device=device)
        losses = _train_steps_k2(model, opt, x, y, x_probe, y_probe, 20)
        assert losses[-1] < losses[0]

    def test_cnn_mps(self):
        _seed()
        device = torch.device("mps")
        model = _make_cnn().to(device)
        opt = EVE(model.parameters(), lr=1e-2, K=2)

        x = torch.randn(8, 1, 8, 8, device=device)
        y = torch.randn(8, 2, device=device)
        x_probe = torch.randn(8, 1, 8, 8, device=device)
        y_probe = torch.randn(8, 2, device=device)
        losses = _train_steps_k2(model, opt, x, y, x_probe, y_probe, 10)
        assert all(math.isfinite(l) for l in losses)
