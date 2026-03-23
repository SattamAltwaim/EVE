"""
EVE — Evolutionary Virtual Exploration optimizer (simplified).

Implements the EVE framework from the simpler proposal:
  - Momentum spectrum offspring      (Eq. 1, Section 2.1)
  - Probe-based fitness evaluation   (Eq. 2, Section 2.2)
  - Soft natural selection           (Eq. 3, Section 2.3)
  - Math collapse → α_eff update    (Eq. 4, Section 2.3)

At K=1 the optimizer is *exactly* AMSGrad-AdamW with zero overhead.

Probe implementation uses in-place perturbation (perturb → forward → restore),
which avoids functional_call/vmap overhead and handles BatchNorm models
correctly.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module


class EVE(torch.optim.Optimizer):
    r"""EVE: Evolutionary Virtual Exploration.

    At ``K=1`` this is AMSGrad-AdamW (probes are skipped, no overhead).
    At ``K>1`` the optimizer generates *K* offspring directions by
    interpolating between fast momentum (β₁) and slow momentum (β₁_slow),
    evaluates them via forward-only probes on the training batch, and
    selects among them with temperature-scaled softmax.  The final update
    exploits the "math collapse": since all offspring share the same
    generator equation, the weighted combination reduces to a single
    effective interpolation factor α_eff.

    Args:
        params: iterable of parameters or parameter-group dicts.
        lr: learning rate (η).
        betas: coefficients for first / second moment EMAs (β₁, β₂).
        eps: numerical stabiliser (ε).
        weight_decay: decoupled weight decay coefficient (λ).
        K: brood size — number of offspring directions.
        beta1_slow: slow momentum decay rate (β₁_slow).
        beta_sel: selection temperature for softmax (β_sel).
        record_diagnostics: if True, append per-step internal state to
            ``self._diagnostics`` for analysis.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        K: int = 2,
        beta1_slow: float = 0.999,
        beta_sel: float = 1.0,
        record_diagnostics: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if K < 1:
            raise ValueError(f"Brood size K must be >= 1, got {K}")
        if not 0.0 <= beta1_slow < 1.0:
            raise ValueError(f"Invalid beta1_slow: {beta1_slow}")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            K=K,
            beta1_slow=beta1_slow,
            beta_sel=beta_sel,
        )
        super().__init__(params, defaults)

        self.record_diagnostics: bool = record_diagnostics
        self._diagnostics: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def step(
        self,
        closure: Optional[Callable[[], Tensor]] = None,
        *,
        model: Optional[Module] = None,
        loss_fn: Optional[Callable[..., Tensor]] = None,
        data: Optional[Tuple[Any, Any]] = None,
        current_loss: Optional[float] = None,
    ) -> Optional[Tensor]:
        """Perform a single optimisation step.

        For **K=1** (AMSGrad-AdamW) no extra arguments are needed.

        For **K>1** the probe evaluation requires ``model``, ``loss_fn``,
        and ``data``.  Pass ``current_loss=loss.item()`` to avoid a
        redundant forward pass.
        """
        ret_loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                ret_loss = closure()

        if current_loss is None and ret_loss is not None:
            current_loss = ret_loss.item()

        K: int = self.defaults["K"]

        if K > 1 and (model is None or loss_fn is None or data is None):
            raise ValueError(
                "EVE with K>1 requires `model`, `loss_fn`, and `data` "
                "arguments for probe-based fitness evaluation."
            )

        # ── K=1: fused AMSGrad-AdamW ──────────────────────────────────
        if K == 1:
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lr = group["lr"]
                wd = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["v"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["v_max"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    state["step"] += 1
                    m, v, v_max = state["m"], state["v"], state["v_max"]

                    m.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2_sqrt = math.sqrt(1.0 - beta2 ** state["step"])

                    torch.maximum(v_max, v, out=v_max)

                    step_size = lr / bc1
                    denom = (v_max.sqrt() / bc2_sqrt).add_(eps)

                    if wd != 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.addcdiv_(m, denom, value=-step_size)

            return ret_loss

        # ── K>1 path ──────────────────────────────────────────────────

        assert model is not None and loss_fn is not None and data is not None
        inp, tgt = data

        if current_loss is None:
            with torch.no_grad():
                current_loss = loss_fn(
                    model(self._unpack_input(inp)), tgt
                ).item()

        # ── Phase 1: moment updates ───────────────────────────────────
        # Precompute m_hat, m_hat_slow, denom per parameter for use
        # during probe (on-the-fly offspring) and final update.
        m_hat_map: Dict[int, Tensor] = {}
        m_hat_slow_map: Dict[int, Tensor] = {}
        denom_map: Dict[int, Tensor] = {}
        params_with_grad: List[Tuple[Dict, Tensor]] = []

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta1_slow = group["beta1_slow"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["m_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v_max"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                m, m_slow = state["m"], state["m_slow"]
                v, v_max = state["v"], state["v_max"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m_slow.mul_(beta1_slow).add_(grad, alpha=1.0 - beta1_slow)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step = state["step"]
                bc1 = 1.0 - beta1 ** step
                bc1_slow = 1.0 - beta1_slow ** step
                bc2_sqrt = math.sqrt(1.0 - beta2 ** step)

                torch.maximum(v_max, v, out=v_max)

                ptr = p.data_ptr()
                m_hat_map[ptr] = m / bc1
                m_hat_slow_map[ptr] = m_slow / bc1_slow
                denom_map[ptr] = (v_max.sqrt() / bc2_sqrt).add_(group["eps"])

                params_with_grad.append((group, p))

        # ── Phase 2: alpha grid ───────────────────────────────────────
        alphas = [k / (K - 1) for k in range(K)]

        # ── Phase 3: probe-based fitness evaluation (Eq. 2) ──────────
        probe_inp = self._unpack_input(inp)
        probe_tgt = tgt

        saved: Dict[int, Tensor] = {
            p.data_ptr(): p.data.clone()
            for _, p in params_with_grad
        }

        was_training = model.training
        model.eval()

        with torch.no_grad():
            L_base: Tensor = loss_fn(model(probe_inp), probe_tgt)

            probe_losses: List[Tensor] = []
            for k in range(K):
                a_k = alphas[k]
                for group, p in params_with_grad:
                    ptr = p.data_ptr()
                    d_k = (
                        m_hat_map[ptr].mul(-(1.0 - a_k))
                        .add_(m_hat_slow_map[ptr], alpha=-a_k)
                    ).div_(denom_map[ptr])
                    p.data.add_(d_k, alpha=group["lr"])

                probe_losses.append(loss_fn(model(probe_inp), probe_tgt))

                for _, p in params_with_grad:
                    p.data.copy_(saved[p.data_ptr()])

            losses = torch.stack(probe_losses)

        model.train(was_training)

        fitness = L_base - losses

        # ── Phase 4: soft selection + math collapse (Eqs. 3–4) ────────
        beta_sel = self.defaults["beta_sel"]
        fitness_range = (fitness.max() - fitness.min()).clamp(min=1e-8)
        weights = torch.softmax(beta_sel * fitness / fitness_range, dim=0)

        alphas_t = torch.tensor(alphas, device=weights.device, dtype=weights.dtype)
        alpha_eff: float = (weights * alphas_t).sum().item()

        # ── Diagnostics capture ───────────────────────────────────────
        if self.record_diagnostics:
            self._record_step_diagnostics(
                fitness, weights, alpha_eff, alphas,
                m_hat_map, m_hat_slow_map, denom_map,
                params_with_grad, current_loss,
            )

        # ── Phase 5: final update via generator equation (Eq. 4) ──────
        for group, p in params_with_grad:
            ptr = p.data_ptr()
            lr = group["lr"]
            wd = group["weight_decay"]

            d_final = (
                m_hat_map[ptr].mul(-(1.0 - alpha_eff))
                .add_(m_hat_slow_map[ptr], alpha=-alpha_eff)
            ).div_(denom_map[ptr])

            with torch.no_grad():
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(d_final, alpha=lr)

        return ret_loss

    # ------------------------------------------------------------------
    #  Diagnostics recording (moderate level)
    # ------------------------------------------------------------------

    def _record_step_diagnostics(
        self,
        fitness: Tensor,
        weights: Tensor,
        alpha_eff: float,
        alphas: List[float],
        m_hat_map: Dict[int, Tensor],
        m_hat_slow_map: Dict[int, Tensor],
        denom_map: Dict[int, Tensor],
        params_with_grad: List[Tuple[Dict, Tensor]],
        current_loss: Optional[float],
    ) -> None:
        """Capture per-step internal state for analysis.

        Re-materialises offspring directions on GPU to compute norms and
        cosine similarities.  Only called when record_diagnostics=True.
        """
        K = fitness.shape[0]

        flat_dirs: List[List[Tensor]] = [[] for _ in range(K)]
        flat_combined: List[Tensor] = []

        for _group, p in params_with_grad:
            ptr = p.data_ptr()
            m_hat = m_hat_map[ptr]
            m_hat_slow = m_hat_slow_map[ptr]
            denom = denom_map[ptr]

            dirs_p: List[Tensor] = []
            for k in range(K):
                a_k = alphas[k]
                d_k = (-(1.0 - a_k) * m_hat - a_k * m_hat_slow) / denom
                dirs_p.append(d_k)
                flat_dirs[k].append(d_k.detach().reshape(-1))

            d_final = (-(1.0 - alpha_eff) * m_hat - alpha_eff * m_hat_slow) / denom
            flat_combined.append(d_final.detach().reshape(-1))

        dir_vecs: List[Tensor] = [torch.cat(fd) for fd in flat_dirs]
        combined_vec: Tensor = torch.cat(flat_combined)

        dir_norms: List[float] = [v.norm().item() for v in dir_vecs]

        eps_cs = 1e-8
        dir_norms_t = [v.norm().clamp(min=eps_cs) for v in dir_vecs]
        dir_unit: List[Tensor] = [v / n for v, n in zip(dir_vecs, dir_norms_t)]
        combined_norm = combined_vec.norm().clamp(min=eps_cs)
        combined_unit: Tensor = combined_vec / combined_norm

        labels = [f"d{i+1}" for i in range(K)]
        cos_pairs: Dict[str, float] = {}
        for i in range(K):
            for j in range(i + 1, K):
                cos_pairs[f"{labels[i]}-{labels[j]}"] = (
                    (dir_unit[i] * dir_unit[j]).sum().item()
                )

        cos_to_combined: List[float] = [
            (dir_unit[k] * combined_unit).sum().item() for k in range(K)
        ]

        self._diagnostics.append({
            "loss":            current_loss,
            "fitness":         fitness.detach().cpu().tolist(),
            "weights":         weights.detach().cpu().tolist(),
            "alpha_eff":       alpha_eff,
            "beta_sel":        self.defaults["beta_sel"],
            "dir_norms":       dir_norms,
            "cos_pairs":       cos_pairs,
            "cos_to_combined": cos_to_combined,
        })

    # ------------------------------------------------------------------
    #  Input handling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_input(inp: Any) -> Any:
        """Unpack a single-element tuple, pass everything else through."""
        if isinstance(inp, (tuple, list)) and len(inp) == 1:
            return inp[0]
        return inp

    @staticmethod
    def _slice_input(inp: Any, n: int) -> Any:
        """Slice the first *n* samples from a tensor or tuple of tensors."""
        if isinstance(inp, (tuple, list)):
            return tuple(x[:n] for x in inp)
        return inp[:n]
