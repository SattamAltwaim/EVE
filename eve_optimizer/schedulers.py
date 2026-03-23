"""Temperature schedulers for EVE's selection temperature (beta_sel).

These schedulers anneal beta_sel over training, typically from a low value
(exploration -- near-uniform offspring blending) to a high value
(exploitation -- greedy, winner-take-all selection).

API mirrors ``torch.optim.lr_scheduler`` conventions:
    - Constructor takes the EVE optimizer as first argument.
    - ``.step()`` is called once per epoch to advance the schedule.
    - ``.get_beta_sel()`` returns the current temperature value.
    - ``.state_dict()`` / ``.load_state_dict()`` for checkpointing.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from torch.optim import Optimizer


class _BetaSelScheduler:
    """Base class for beta_sel temperature schedulers.

    Args:
        optimizer: An EVE optimizer instance.
        last_epoch: Index of the last epoch.  Set to ``-1`` on fresh start.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.beta_sel_init: float = optimizer.defaults["beta_sel"]
        self.last_epoch = last_epoch
        self._last_beta_sel = self.beta_sel_init

        if last_epoch == -1:
            self.last_epoch = 0
            self._last_beta_sel = self.get_beta_sel()
            self.optimizer.defaults["beta_sel"] = self._last_beta_sel

    def get_beta_sel(self) -> float:
        """Compute the beta_sel value for the current epoch."""
        raise NotImplementedError

    def step(self) -> None:
        """Advance the schedule by one epoch."""
        self.last_epoch += 1
        beta_sel = self.get_beta_sel()
        self.optimizer.defaults["beta_sel"] = beta_sel
        self._last_beta_sel = beta_sel

    def get_last_beta_sel(self) -> float:
        """Return the most recently computed beta_sel value."""
        return self._last_beta_sel

    def state_dict(self) -> Dict[str, Any]:
        return {
            "last_epoch": self.last_epoch,
            "beta_sel_init": self.beta_sel_init,
            "_last_beta_sel": self._last_beta_sel,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.last_epoch = state_dict["last_epoch"]
        self.beta_sel_init = state_dict["beta_sel_init"]
        self._last_beta_sel = state_dict["_last_beta_sel"]
        self.optimizer.defaults["beta_sel"] = self._last_beta_sel


class StepBetaSel(_BetaSelScheduler):
    """Multiply beta_sel by ``gamma`` every ``step_size`` epochs.

    Starts at the optimizer's initial ``beta_sel`` and increases by a factor
    of ``gamma`` every ``step_size`` epochs, clamped to ``beta_sel_max``.

    Args:
        optimizer: An EVE optimizer instance.
        step_size: Period of beta_sel increase (in epochs).
        gamma: Multiplicative factor applied each step.
        beta_sel_max: Upper bound for beta_sel.
        last_epoch: Index of the last epoch (``-1`` for fresh start).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 10,
        gamma: float = 2.0,
        beta_sel_max: float = 5.0,
        last_epoch: int = -1,
    ) -> None:
        self.step_size = step_size
        self.gamma = gamma
        self.beta_sel_max = beta_sel_max
        super().__init__(optimizer, last_epoch)

    def get_beta_sel(self) -> float:
        raw = self.beta_sel_init * (self.gamma ** (self.last_epoch // self.step_size))
        return min(raw, self.beta_sel_max)

    def state_dict(self) -> Dict[str, Any]:
        d = super().state_dict()
        d.update(step_size=self.step_size, gamma=self.gamma,
                 beta_sel_max=self.beta_sel_max)
        return d

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.step_size = state_dict["step_size"]
        self.gamma = state_dict["gamma"]
        self.beta_sel_max = state_dict["beta_sel_max"]
        super().load_state_dict(state_dict)


class CosineAnnealingBetaSel(_BetaSelScheduler):
    """Anneal beta_sel from its initial value to ``beta_sel_max`` via cosine.

    Follows a half-cosine curve over ``T_max`` epochs::

        beta_sel(t) = init + (max - init) * (1 - cos(pi * t / T_max)) / 2

    This yields the initial value at epoch 0 and ``beta_sel_max`` at
    epoch ``T_max``.

    Args:
        optimizer: An EVE optimizer instance.
        T_max: Number of epochs over which the schedule completes.
        beta_sel_max: Target beta_sel at the end of the schedule.
        last_epoch: Index of the last epoch (``-1`` for fresh start).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        beta_sel_max: float = 5.0,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = T_max
        self.beta_sel_max = beta_sel_max
        super().__init__(optimizer, last_epoch)

    def get_beta_sel(self) -> float:
        t = min(self.last_epoch, self.T_max)
        return (
            self.beta_sel_init
            + (self.beta_sel_max - self.beta_sel_init)
            * (1.0 - math.cos(math.pi * t / self.T_max))
            / 2.0
        )

    def state_dict(self) -> Dict[str, Any]:
        d = super().state_dict()
        d.update(T_max=self.T_max, beta_sel_max=self.beta_sel_max)
        return d

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.T_max = state_dict["T_max"]
        self.beta_sel_max = state_dict["beta_sel_max"]
        super().load_state_dict(state_dict)
