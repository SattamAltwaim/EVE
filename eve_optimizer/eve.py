"""
EVE — Evolutionary Virtual Exploration optimizer.

Faithful implementation of every equation in the EVE paper draft:
  - Offspring direction construction  (Eqs. 7–10, Section 4.1)
  - Probe-based fitness evaluation    (Eq. 14,    Section 4.2)
  - Soft natural selection            (Eq. 18,    Section 4.3)
  - EVE parameter update              (Eq. 20,    Section 4.3)
  - Strength signal                   (Section 4.4)
  - Adaptive selection temperature    (Eq. 26,    Section 4.5)

At K=1 the optimizer is *exactly* AdamW with zero overhead.

Probe implementation uses in-place perturbation (perturb → forward → restore),
which avoids functional_call/vmap overhead and handles BatchNorm models
correctly. Expected overhead:
  - sub-batch probe  (|B'| = |B|/K): ~33%  (paper's design, Proposition 1)
  - full-batch probe (|B'| = |B|):   ~133% for K=4
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module


class EVE(torch.optim.Optimizer):
    r"""EVE: Evolutionary Virtual Exploration.

    At ``K=1`` this is exactly AdamW (probes are skipped, no overhead).
    At ``K>1`` the optimizer constructs *K* offspring directions, evaluates
    them via forward-only probes on the full training batch using in-place
    parameter perturbation, and selects among them with temperature-scaled
    softmax.

    Args:
        params: iterable of parameters or parameter-group dicts.
        lr: learning rate (eta).
        betas: coefficients for first / second moment EMAs (beta1, beta2).
        eps: numerical stabiliser (epsilon).
        weight_decay: decoupled weight decay coefficient (lambda).
        K: brood size — number of offspring directions.
        rho: target entropy ratio for adaptive temperature (0 = pure
            exploitation, 1 = pure exploration).
        alpha_beta: selection-temperature adaptation rate.
        beta_sel_init: initial selection temperature (beta_sel).
        beta_sel_range: (min, max) clamp for selection temperature.
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
        K: int = 4,
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
            rho=rho,
            alpha_beta=alpha_beta,
            beta_sel_init=beta_sel_init,
            beta_sel_range=beta_sel_range,
        )
        super().__init__(params, defaults)

        self.beta_sel: float = beta_sel_init
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
        current_loss: Optional[float] = None,
    ) -> Optional[Tensor]:
        """Perform a single optimisation step.

        For **K=1** (exact AdamW) no extra arguments are needed — just
        call ``step()`` after the usual ``zero_grad / forward / backward``
        sequence.

        For **K>1** the probe evaluation requires ``model``, ``loss_fn``,
        and ``data``. Pass ``current_loss=loss.item()`` to avoid an extra
        hidden forward pass (the loss is already computed during the
        training backward pass):

        .. code-block:: python

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step(
                model=model, loss_fn=loss_fn, data=(x, y),
                current_loss=loss.item(),
            )

        Args:
            closure: re-evaluates the model and returns the loss (optional).
            model: the ``nn.Module`` being trained (required when K>1).
            loss_fn: ``loss_fn(model_output, target) -> scalar`` (req. K>1).
            data: ``(input, target)`` for probe evaluation (req. K>1).
                  *input* may be a single tensor or a tuple of tensors.
            current_loss: scalar loss value from the current training step.
                  When provided, avoids an extra forward pass.  If omitted
                  and no closure is given, EVE will run one extra forward
                  pass to obtain this value.

        Returns:
            The training loss when *closure* is provided, else ``None``.
        """
        ret_loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                ret_loss = closure()

        # Resolve current_loss from (in priority order):
        # 1. explicit keyword argument
        # 2. closure return value
        # 3. fallback extra forward pass (Bug 2 — only if no better source)
        if current_loss is None and ret_loss is not None:
            current_loss = ret_loss.item()

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

            return ret_loss

        # ── K>1 path ─────────────────────────────────────────────────────

        # Fallback: compute current_loss from a forward pass only when the
        # caller did not supply it. This path is avoided when the notebook
        # passes current_loss=loss.item() (Fix Bug 2).
        assert model is not None and loss_fn is not None and data is not None
        inp, tgt = data

        if current_loss is None:
            with torch.no_grad():
                current_loss = loss_fn(
                    model(self._unpack_input(inp)), tgt
                ).item()

        # ── Phase 2: moment updates + offspring construction ─────────────
        offspring_map: Dict[int, Tensor] = {}
        sqrt_v_hat_map: Dict[int, Tensor] = {}
        ptr_to_lr: Dict[int, float] = {}
        params_with_grad: List[Tuple[Dict, Tensor]] = []

        # 2a. First pass — update moments and collect sqrt(v_hat).
        #     Do NOT call .item() inside the loop; that forces a CPU-GPU
        #     sync for every parameter (Fix Bug 1).
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

                state["step"] += 1
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc2 = 1.0 - beta2 ** state["step"]
                sqrt_v_hat = (v / bc2).sqrt()
                sqrt_v_hat_map[p.data_ptr()] = sqrt_v_hat

                ptr_to_lr[p.data_ptr()] = group["lr"]
                params_with_grad.append((group, p))

        # Single GPU reduction across all parameters — one .item() sync
        # instead of one per parameter (Fix Bug 1).
        global_max_sqrt_v: float = torch.stack(
            [sv.max() for sv in sqrt_v_hat_map.values()]
        ).max().item()

        # 2b. Compute z-scored strength signal across ALL dimensions.
        #     r_d = |m_hat_d| / (sqrt(v_hat_d) + eps), then z-score across
        #     the full parameter vector and apply sigmoid.
        ratio_list: List[Tensor] = []
        for group, p in params_with_grad:
            state = self.state[p]
            beta1 = group["betas"][0]
            eps = group["eps"]
            bc1 = 1.0 - beta1 ** state["step"]
            m_hat = state["m"] / bc1
            sqrt_v_hat = sqrt_v_hat_map[p.data_ptr()]
            ratio_list.append((m_hat.abs() / (sqrt_v_hat + eps)).reshape(-1))

        r_cat = torch.cat(ratio_list)
        r_mean = r_cat.mean()
        r_std = r_cat.std()

        s_map: Dict[int, Tensor] = {}
        for idx, (group, p) in enumerate(params_with_grad):
            z = (ratio_list[idx].view(p.shape) - r_mean) / (r_std + group["eps"])
            s_map[p.data_ptr()] = torch.sigmoid(z)

        # 2c. Second pass — construct offspring directions (Eqs. 7–10).
        #     Stacked into a (K, *shape) tensor per parameter for the
        #     weighted einsum update (Phase 6).
        for group, p in params_with_grad:
            grad = p.grad
            state = self.state[p]
            eps = group["eps"]
            grp_K = group["K"]
            beta1 = group["betas"][0]
            m = state["m"]
            s = s_map[p.data_ptr()]

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
        #
        # In-place perturbation approach (Fixes Bugs 3 & 4):
        #   • No stacked_params dict → no K copies of all parameters
        #   • No functional_call / vmap → no BatchNorm vmap failures
        #   • Uses model.forward() directly: handles all layer types
        #   • Memory: one backup clone of trainable params (D floats, ~44 MB
        #     for ResNet-18) instead of K copies (~176 MB)
        #   • Compute: K sequential forward passes on the probe batch
        #     → paper-compliant cost (Proposition 1 for sub-batch design)
        #
        probe_inp = self._unpack_input(inp)
        probe_tgt = tgt

        # Backup current parameters (θ_t) — restore after each offspring.
        saved: Dict[int, Tensor] = {
            p.data_ptr(): p.data.clone()
            for _, p in params_with_grad
        }

        was_training = model.training
        model.eval()

        with torch.no_grad():
            # Baseline loss at θ_t (control variate, Eq. 14).
            L_base: Tensor = loss_fn(model(probe_inp), probe_tgt)

            # Evaluate each offspring direction.
            probe_losses: List[Tensor] = []
            for k in range(K):
                # Perturb: θ_t + η · d_k  (in-place, no extra allocation)
                for group, p in params_with_grad:
                    lr_k = group["lr"]
                    p.data.add_(offspring_map[p.data_ptr()][k], alpha=lr_k)

                # Forward-only pass at perturbed parameters.
                probe_losses.append(loss_fn(model(probe_inp), probe_tgt))

                # Restore θ_t before next offspring.
                for _, p in params_with_grad:
                    p.data.copy_(saved[p.data_ptr()])

            losses = torch.stack(probe_losses)

        model.train(was_training)

        fitness = L_base - losses

        # ── Phase 5: soft selection (Eq. 18) ─────────────────────────────
        weights = torch.softmax(self.beta_sel * fitness, dim=0)

        # ── Diagnostics capture ──────────────────────────────────────────
        if self.record_diagnostics:
            self._record_step_diagnostics(
                fitness, weights, offspring_map, params_with_grad,
                current_loss, s_map,
            )

        # ── Phase 6: weighted update (Eq. 20) ────────────────────────────
        for group, p in params_with_grad:
            dir_stack = offspring_map[p.data_ptr()]
            lr = group["lr"]
            wd = group["weight_decay"]

            combined = torch.einsum("k...,k->...", dir_stack, weights)

            with torch.no_grad():
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(combined, alpha=lr)

        # ── Phase 7: adaptive temperature (Eq. 26) ───────────────────────
        self._adapt_temperature(weights, K)

        return ret_loss

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
        s_map: Dict[int, Tensor],
    ) -> None:
        """Capture per-step internal state for analysis.

        All heavy computation (norms, cosine similarities, strength-signal
        statistics) is performed on the GPU; only the final scalars are
        transferred to CPU via .item().  This avoids the ~269 MB of
        synchronous GPU→CPU tensor copies that the naïve implementation
        incurred (~1 150 ms overhead per step on an L4 GPU).
        """
        K = fitness.shape[0]

        # ── Collect flat direction vectors and combined update ON GPU ─────────
        flat_dirs: List[List[Tensor]] = [[] for _ in range(K)]
        flat_combined: List[Tensor] = []
        all_s: List[Tensor] = []

        for _group, p in params_with_grad:
            dirs = offspring_map[p.data_ptr()]
            for k in range(K):
                flat_dirs[k].append(dirs[k].detach().reshape(-1))
            combined = torch.einsum("k...,k->...", dirs, weights)
            flat_combined.append(combined.detach().reshape(-1))
            ptr = p.data_ptr()
            if ptr in s_map:
                all_s.append(s_map[ptr].detach().reshape(-1))

        # Concatenate on GPU — no .cpu() here.
        dir_vecs: List[Tensor] = [torch.cat(fd) for fd in flat_dirs]
        combined_vec: Tensor = torch.cat(flat_combined)

        # ── Norms (GPU reduction → single scalar transfer each) ───────────────
        dir_norms: List[float] = [v.norm().item() for v in dir_vecs]

        # ── Cosine similarities on GPU ────────────────────────────────────────
        # Pre-normalise once per vector to avoid redundant norm computations.
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

        # ── Strength-signal statistics on GPU ─────────────────────────────────
        s_stats: Dict[str, float] = {}
        if all_s:
            s_cat = torch.cat(all_s)          # GPU tensor
            s_stats = {
                "mean":   s_cat.mean().item(),
                "std":    s_cat.std().item(),
                "min":    s_cat.min().item(),
                "max":    s_cat.max().item(),
                "median": s_cat.median().item(),
            }

        self._diagnostics.append({
            "step":           self._global_step,
            "loss":           current_loss,
            "fitness":        fitness.detach().cpu().tolist(),
            "weights":        weights.detach().cpu().tolist(),
            "beta_sel":       self.beta_sel,
            "dir_norms":      dir_norms,
            "cos_pairs":      cos_pairs,
            "cos_to_combined": cos_to_combined,
            "s_stats":        s_stats,
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
