"""EVE -- Evolutionary Virtual Exploration optimizer.

One elite direction.  One Adam direction.  K blends between them,
probed on the next batch.  The winner is applied and becomes the
next elite.

K=1 is AdamW.  One new hyperparameter.

Algorithm (per step):
    1. Forward + backward on B_t  -> gradient g_t
    2. Update moments m_t, v_t.  Compute Adam direction a_t.
    3. Spawn K offspring: d_k = lerp(a_t, elite, k/(K-1))
    4. Probe each d_k on B_{t+1}: L_k = loss(theta + lr*d_k ; B_{t+1})
    5. Winner: k* = argmin L_k
    6. Apply:  theta <- theta + lr*d_{k*} - lr*wd*theta
    7. Elite:  elite <- d_{k*}
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ._foreach import (
    foreach_adam_direction,
    foreach_apply_direction,
    foreach_apply_weight_decay,
    foreach_copy,
    foreach_lerp_directions,
    foreach_moment_update,
    foreach_snapshot,
)
from ._probing import probe_offspring
from .diagnostics import DiagnosticsRecorder


class EVE(torch.optim.Optimizer):
    r"""EVE: Evolutionary Virtual Exploration.

    Args:
        params: iterable of parameters or parameter-group dicts.
        lr: learning rate (default: 1e-3).
        betas: coefficients for running averages of gradient and its
            square (default: (0.9, 0.999)).
        eps: numerical stabiliser (default: 1e-8).
        weight_decay: decoupled weight decay (default: 0.01).
        K: number of offspring directions.  K=1 is pure AdamW.
            K>=2 enables evolutionary probing (default: 2).
        record_diagnostics: if True, store per-step diagnostics
            accessible via ``self.diagnostics`` (default: False).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        K: int = 2,
        record_diagnostics: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")

        defaults: Dict[str, Any] = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, K=K,
        )
        super().__init__(params, defaults)

        self._global_step: int = 0
        self.diagnostics: Optional[DiagnosticsRecorder] = (
            DiagnosticsRecorder() if record_diagnostics else None
        )

        # Pre-compute blend alphas once.
        # alpha=0 -> pure Adam (d_1), alpha=1 -> pure elite (d_K).
        if K >= 2:
            self._blend_alphas = torch.linspace(0.0, 1.0, K)
        else:
            self._blend_alphas = torch.zeros(1)

    # ------------------------------------------------------------------
    #  Internals: collect param lists for foreach ops
    # ------------------------------------------------------------------

    def _collect_params(
        self,
    ) -> Tuple[
        List[Tensor],   # param data
        List[Tensor],   # grads
        List[Tensor],   # ms
        List[Tensor],   # vs
        List[Tensor],   # elites
        float,          # beta1
        float,          # beta2
        float,          # eps
        float,          # lr
        float,          # wd
        int,            # K
    ]:
        """Single pass over param_groups to build flat lists for foreach ops.

        Initialises optimizer state on first encounter.
        """
        param_data: List[Tensor] = []
        grads: List[Tensor] = []
        ms: List[Tensor] = []
        vs: List[Tensor] = []
        elites: List[Tensor] = []

        # Use first group's hypers (multi-group support preserved via defaults).
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        wd = group["weight_decay"]
        K = group["K"]

        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["elite"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                param_data.append(p.data)
                grads.append(p.grad)
                ms.append(state["m"])
                vs.append(state["v"])
                elites.append(state["elite"])

        return param_data, grads, ms, vs, elites, beta1, beta2, eps, lr, wd, K

    def _increment_step(self) -> int:
        """Increment per-parameter step counters and return the step number."""
        step = None
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                self.state[p]["step"] += 1
                if step is None:
                    step = self.state[p]["step"]
        return step if step is not None else 0

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], Tensor]] = None,
        probe_fn: Optional[Callable[[], Tensor]] = None,
        model: Optional[nn.Module] = None,
    ) -> Optional[Tensor]:
        """Perform a single optimisation step.

        Args:
            closure: re-evaluates model and returns loss (optional).
                Executed with grad enabled before the step.
            probe_fn: callable returning scalar loss on the *next* batch.
                Required when K >= 2.  Ignored when K = 1.
                Called under torch.no_grad(); the optimizer handles
                parameter save/restore around each probe.
            model: the model being optimised.  Only needed when K >= 2
                and the model contains BatchNorm (or similar) layers,
                so their running statistics can be saved/restored
                during probing.

        Returns:
            Loss from *closure* if provided, else None.
        """
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        collected = self._collect_params()
        param_data, grads, ms, vs, elites, beta1, beta2, eps, lr, wd, K = collected

        if not param_data:
            return loss

        step = self._increment_step()

        # -- Phase 1: moment updates (vectorised) --------------------------
        foreach_moment_update(ms, vs, grads, beta1, beta2)

        # -- Phase 2: Adam direction (vectorised) --------------------------
        adam_dirs = foreach_adam_direction(ms, vs, step, beta1, beta2, eps)

        # -- K=1 fast path: pure AdamW, no probing -------------------------
        if K == 1:
            foreach_apply_weight_decay(param_data, lr, wd)
            foreach_apply_direction(param_data, adam_dirs, lr)
            foreach_copy(elites, adam_dirs)
            if self.diagnostics is not None:
                losses_t = torch.zeros(1, device=param_data[0].device)
                self.diagnostics.record(
                    self._global_step, 0, losses_t, adam_dirs, elites,
                )
            return loss

        # -- K>=2 path: evolutionary probing --------------------------------
        alphas = self._blend_alphas.to(param_data[0].device)

        probe_losses, winner_idx = probe_offspring(
            param_data, adam_dirs, elites, alphas, lr, probe_fn, model,
        )

        # Build winning direction
        alpha_star = alphas[winner_idx].item()
        d_star = foreach_lerp_directions(adam_dirs, elites, alpha_star)

        # Apply update: weight decay then winning direction
        foreach_apply_weight_decay(param_data, lr, wd)
        foreach_apply_direction(param_data, d_star, lr)

        # Elite inheritance: elite <- d_{k*}
        foreach_copy(elites, d_star)

        # Diagnostics
        if self.diagnostics is not None:
            self.diagnostics.record(
                self._global_step, winner_idx, probe_losses, adam_dirs, elites,
            )

        return loss
