"""
EVE — Evolutionary Virtual Exploration optimizer.

Faithful implementation of every equation in the EVE paper draft:
  - Offspring direction construction  (Eqs. 7–10, Section 4.1)
  - Probe-based fitness evaluation    (Eq. 14,    Section 4.2)
  - Soft natural selection            (Eq. 18,    Section 4.3)
  - EVE parameter update              (Eq. 20,    Section 4.3)
  - Strength signal                   (Eq. 25,    Section 4.4)
  - Adaptive selection temperature    (Eq. 26,    Section 4.5)

At K=1 the optimizer is *exactly* AdamW with zero overhead.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.func import functional_call, vmap
from torch.nn import Module


class EVE(torch.optim.Optimizer):
    r"""EVE: Evolutionary Virtual Exploration.

    At ``K=1`` this is exactly AdamW (probes are skipped, no overhead).
    At ``K>1`` the optimizer constructs *K* offspring directions, evaluates
    them via forward-only probes on a shared sub-batch, and selects among
    them with temperature-scaled softmax.

    Args:
        params: iterable of parameters or parameter-group dicts.
        lr: learning rate (eta).
        betas: coefficients for first / second moment EMAs (beta1, beta2).
        eps: numerical stabiliser (epsilon).
        weight_decay: decoupled weight decay coefficient (lambda).
        K: brood size — number of offspring directions.
        gamma_s: strength-signal decay rate.
        rho: target entropy ratio for adaptive temperature (0 = pure
            exploitation, 1 = pure exploration).
        alpha_beta: selection-temperature adaptation rate.
        beta_sel_init: initial selection temperature (beta_sel).
        beta_sel_range: (min, max) clamp for selection temperature.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        K: int = 4,
        gamma_s: float = 0.99,
        rho: float = 0.5,
        alpha_beta: float = 0.01,
        beta_sel_init: float = 1.0,
        beta_sel_range: Tuple[float, float] = (0.1, 100.0),
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

        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            K=K,
            gamma_s=gamma_s,
            rho=rho,
            alpha_beta=alpha_beta,
            beta_sel_init=beta_sel_init,
            beta_sel_range=beta_sel_range,
        )
        super().__init__(params, defaults)

        self.beta_sel: float = beta_sel_init
        self._prev_loss: Optional[float] = None
        self._global_step: int = 0
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
    ) -> Optional[Tensor]:
        """Perform a single optimisation step.

        For **K=1** (exact AdamW) no extra arguments are needed — just
        call ``step()`` after the usual ``zero_grad / forward / backward``
        sequence.

        For **K>1** the probe evaluation requires ``model``, ``loss_fn``,
        and ``data``:

        .. code-block:: python

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step(model=model, loss_fn=loss_fn, data=(x, y))

        Args:
            closure: re-evaluates the model and returns the loss (optional).
            model: the ``nn.Module`` being trained (required when K>1).
            loss_fn: ``loss_fn(model_output, target) -> scalar`` (req. K>1).
            data: ``(input, target)`` for probe evaluation (req. K>1).
                  *input* may be a single tensor or a tuple of tensors.

        Returns:
            The training loss when *closure* is provided, else ``None``.
        """
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        current_loss: Optional[float] = loss.item() if loss is not None else None

        K: int = self.defaults["K"]

        if K > 1 and (model is None or loss_fn is None or data is None):
            raise ValueError(
                "EVE with K>1 requires `model`, `loss_fn`, and `data` "
                "arguments for probe-based fitness evaluation."
            )

        self._global_step += 1

        # ── K=1: fused single-pass AdamW  (Proposition 2) ────────────────
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
                        state["s"] = torch.full_like(p, 0.5)

                    state["step"] += 1
                    m, v = state["m"], state["v"]

                    m.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

                    bc1 = 1.0 - beta1 ** state["step"]
                    step_size = lr / bc1
                    bc2_sqrt = math.sqrt(1.0 - beta2 ** state["step"])

                    if wd != 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    denom = (v.sqrt() / bc2_sqrt).add_(eps)
                    p.data.addcdiv_(m, denom, value=-step_size)

            return loss

        # ── K>1 path ─────────────────────────────────────────────────────

        # Obtain current training loss for strength signal when no closure.
        if current_loss is None and data is not None and model is not None:
            inp, tgt = data
            with torch.no_grad():
                current_loss = loss_fn(model(self._unpack_input(inp)), tgt).item()

        # ── Phase 1: strength-signal update (Eq. 25) ─────────────────────
        self._update_strength_signal(current_loss)

        # ── Phase 2: moment updates + offspring construction ─────────────
        offspring_map: Dict[int, Tensor] = {}
        sqrt_v_hat_map: Dict[int, Tensor] = {}
        ptr_to_lr: Dict[int, float] = {}

        # 2a. First pass — update moments, compute sqrt(v_hat), find global
        #     max(sqrt(v_hat)) needed by the contrarian offspring (Eq. 10).
        global_max_sqrt_v: float = 0.0
        params_with_grad: List[Tuple[Dict, Tensor]] = []

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

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
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["s"] = torch.full_like(p, 0.5)
                    state["prev_update_sign"] = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc2 = 1.0 - beta2 ** state["step"]
                sqrt_v_hat = (v / bc2).sqrt()
                sqrt_v_hat_map[p.data_ptr()] = sqrt_v_hat

                local_max = sqrt_v_hat.max().item()
                if local_max > global_max_sqrt_v:
                    global_max_sqrt_v = local_max

                ptr_to_lr[p.data_ptr()] = group["lr"]
                params_with_grad.append((group, p))

        # 2b. Second pass — construct offspring directions (Eqs. 7–10).
        #     Directions are stacked into a (K, *shape) tensor per parameter
        #     for efficient vmap probe and einsum-based weighted combination.
        for group, p in params_with_grad:
            grad = p.grad
            state = self.state[p]
            eps = group["eps"]
            grp_K = group["K"]
            beta1 = group["betas"][0]
            m, s = state["m"], state["s"]

            bc1 = 1.0 - beta1 ** state["step"]
            m_hat = m / bc1
            sqrt_v_hat = sqrt_v_hat_map[p.data_ptr()]
            denom = sqrt_v_hat + eps

            directions: List[Tensor] = []

            # d1 — Adam (Eq. 7)
            directions.append(m_hat.neg() / denom)

            if grp_K >= 2:
                # d2 — Gradient (Eq. 8)
                directions.append(grad.neg() / denom)

            if grp_K >= 3:
                # d3 — Complementary (Eq. 9)
                directions.append((grad.neg() * (1.0 - s)) / denom)

            if grp_K >= 4:
                # d4 — Contrarian (Eq. 10)
                directions.append(
                    grad.sign().neg() * (sqrt_v_hat / (global_max_sqrt_v + eps))
                )

            # Extended offspring for K > 4 (Section 4.1.2)
            if grp_K > 4:
                for j in range(1, grp_K - 3):
                    alpha_j = j / (grp_K - 3)
                    interp = m_hat * alpha_j + grad * (1.0 - alpha_j)
                    directions.append(interp.neg() / denom)

            offspring_map[p.data_ptr()] = torch.stack(directions)

        # ── Phase 4: probe-based fitness evaluation (Eq. 14) ─────────────
        assert model is not None and loss_fn is not None and data is not None

        inp, tgt = data
        batch_len = self._input_len(inp)
        probe_size = max(1, batch_len // K)
        probe_inp = self._slice_input(inp, probe_size)
        probe_tgt = tgt[:probe_size]
        fwd_args = self._unpack_fwd_args(probe_inp)

        # Build stacked candidate parameters {name: (K, *shape)} for vmap.
        stacked_params: Dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            ptr = param.data_ptr()
            if ptr in offspring_map:
                lr_k = ptr_to_lr.get(ptr, self.defaults["lr"])
                stacked_params[name] = (
                    param.data.unsqueeze(0) + lr_k * offspring_map[ptr]
                )
            else:
                stacked_params[name] = param.data.unsqueeze(0).expand(
                    K, *param.shape
                )

        was_training = model.training
        model.eval()

        def _probe_loss(params: Dict[str, Tensor]) -> Tensor:
            out = functional_call(model, params, fwd_args)
            return loss_fn(out, probe_tgt)

        with torch.no_grad():
            L_base = loss_fn(
                model(self._unpack_input(probe_inp)), probe_tgt
            )

            if not hasattr(self, "_vmap_ok"):
                try:
                    losses = vmap(_probe_loss)(stacked_params)
                    self._vmap_ok = True
                except Exception:
                    self._vmap_ok = False

            if self._vmap_ok:
                losses = vmap(_probe_loss)(stacked_params)
            else:
                losses = torch.stack([
                    _probe_loss({n: stacked_params[n][k] for n in stacked_params})
                    for k in range(K)
                ])

        model.train(was_training)

        fitness = L_base - losses

        # ── Phase 5: soft selection (Eq. 18) ─────────────────────────────
        weights = torch.softmax(self.beta_sel * fitness, dim=0)

        # ── Diagnostics capture ──────────────────────────────────────────
        if self.record_diagnostics:
            self._record_step_diagnostics(
                fitness, weights, offspring_map, params_with_grad,
                current_loss,
            )

        # ── Phase 6: weighted update (Eq. 20) ───────────────────────────
        for group, p in params_with_grad:
            dir_stack = offspring_map[p.data_ptr()]
            lr = group["lr"]
            wd = group["weight_decay"]
            state = self.state[p]

            combined = torch.einsum("k...,k->...", dir_stack, weights)

            with torch.no_grad():
                if wd != 0.0:
                    state["prev_update_sign"] = (combined - wd * p.data).sign()
                else:
                    state["prev_update_sign"] = combined.sign()

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(combined, alpha=lr)

        # ── Phase 7: adaptive temperature (Eq. 26) ──────────────────────
        self._adapt_temperature(weights, K)

        self._prev_loss = current_loss
        return loss

    # ------------------------------------------------------------------
    #  Strength signal  (Section 4.4, Eq. 25)
    # ------------------------------------------------------------------

    def _update_strength_signal(self, current_loss: Optional[float]) -> None:
        """Update per-dimension strength signal s_t.

        s_{t+1,d} = γ_s · s_{t,d}
                   + (1 − γ_s) · σ(δ_t · sign(Δθ_{t,d}) · sign(−g_{t+1,d}))

        where δ_t = L_prev − L_current (inter-step loss improvement),
        Δθ is the previous step's displacement, and g_{t+1} is the
        current gradient.
        """
        if self._prev_loss is None or current_loss is None:
            return

        delta_t: float = self._prev_loss - current_loss

        for group in self.param_groups:
            gamma_s: float = group["gamma_s"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "s" not in state or "prev_update_sign" not in state:
                    continue

                s = state["s"]
                prev_sign = state["prev_update_sign"]
                neg_grad_sign = p.grad.neg().sign()

                # Eq. 25: sigmoid(δ_t · sign(Δθ) · sign(−g_{t+1}))
                raw = delta_t * prev_sign * neg_grad_sign
                update_val = torch.sigmoid(raw)

                s.mul_(gamma_s).add_(update_val, alpha=1.0 - gamma_s)

    # ------------------------------------------------------------------
    #  Adaptive selection temperature  (Section 4.5, Eq. 26)
    # ------------------------------------------------------------------

    def _adapt_temperature(self, weights: Tensor, K: int) -> None:
        """Multiplicative temperature update targeting entropy ratio rho.

        β_sel ← β_sel · exp(α_β · (H_t − H*))
        where H_t = −Σ w_k log w_k,  H* = ρ · log K.
        """
        if K <= 1:
            return

        rho: float = self.defaults["rho"]
        alpha_beta: float = self.defaults["alpha_beta"]
        beta_min, beta_max = self.defaults["beta_sel_range"]

        H_max = math.log(K)
        H_star = rho * H_max

        log_w = torch.log(weights.clamp(min=1e-30))
        H_t: float = -(weights * log_w).sum().item()

        self.beta_sel *= math.exp(alpha_beta * (H_t - H_star))
        self.beta_sel = max(beta_min, min(beta_max, self.beta_sel))

    # ------------------------------------------------------------------
    #  Diagnostics recording
    # ------------------------------------------------------------------

    def _record_step_diagnostics(
        self,
        fitness: Tensor,
        weights: Tensor,
        offspring_map: Dict[int, Tensor],
        params_with_grad: List[Tuple[Dict, Tensor]],
        current_loss: Optional[float],
    ) -> None:
        """Capture per-step internal state for analysis (CPU, detached)."""
        K = fitness.shape[0]

        flat_dirs: List[List[Tensor]] = [[] for _ in range(K)]
        flat_combined: List[Tensor] = []
        all_s: List[Tensor] = []

        for _group, p in params_with_grad:
            dirs = offspring_map[p.data_ptr()]
            for k in range(K):
                flat_dirs[k].append(dirs[k].detach().reshape(-1))
            combined = torch.einsum("k...,k->...", dirs, weights)
            flat_combined.append(combined.detach().reshape(-1))
            state = self.state[p]
            if "s" in state:
                all_s.append(state["s"].detach().reshape(-1))

        dir_vecs = [torch.cat(fd).cpu() for fd in flat_dirs]
        combined_vec = torch.cat(flat_combined).cpu()

        dir_norms = [v.norm().item() for v in dir_vecs]

        cos_pairs: Dict[str, float] = {}
        labels = [f"d{i+1}" for i in range(K)]
        for i in range(K):
            for j in range(i + 1, K):
                key = f"{labels[i]}-{labels[j]}"
                cos_pairs[key] = torch.nn.functional.cosine_similarity(
                    dir_vecs[i].unsqueeze(0), dir_vecs[j].unsqueeze(0)
                ).item()

        cos_to_combined = [
            torch.nn.functional.cosine_similarity(
                dir_vecs[k].unsqueeze(0), combined_vec.unsqueeze(0)
            ).item()
            for k in range(K)
        ]

        s_stats: Dict[str, float] = {}
        if all_s:
            s_cat = torch.cat(all_s).cpu()
            s_stats = {
                "mean": s_cat.mean().item(),
                "std": s_cat.std().item(),
                "min": s_cat.min().item(),
                "max": s_cat.max().item(),
                "median": s_cat.median().item(),
            }

        self._diagnostics.append({
            "step": self._global_step,
            "loss": current_loss,
            "fitness": fitness.detach().cpu().tolist(),
            "weights": weights.detach().cpu().tolist(),
            "beta_sel": self.beta_sel,
            "dir_norms": dir_norms,
            "cos_pairs": cos_pairs,
            "cos_to_combined": cos_to_combined,
            "s_stats": s_stats,
        })

    # ------------------------------------------------------------------
    #  Input handling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _input_len(inp: Any) -> int:
        """Return the batch dimension length for a tensor or tuple of tensors."""
        if isinstance(inp, (tuple, list)):
            return len(inp[0])
        return len(inp)

    @staticmethod
    def _slice_input(inp: Any, n: int) -> Any:
        """Slice the first *n* samples from a tensor or tuple of tensors."""
        if isinstance(inp, (tuple, list)):
            return tuple(x[:n] for x in inp)
        return inp[:n]

    @staticmethod
    def _unpack_input(inp: Any) -> Any:
        """Unpack a single-element tuple, pass everything else through."""
        if isinstance(inp, (tuple, list)) and len(inp) == 1:
            return inp[0]
        return inp

    @staticmethod
    def _unpack_fwd_args(inp: Any) -> tuple:
        """Wrap *inp* so it can be splatted into ``functional_call``'s args."""
        if isinstance(inp, (tuple, list)):
            return tuple(inp)
        return (inp,)
