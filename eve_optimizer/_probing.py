"""Next-batch probing infrastructure.

Handles parameter snapshot/restore, BatchNorm state management,
and the K-offspring probe loop.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ._foreach import (
    foreach_apply_direction,
    foreach_lerp_directions,
    foreach_restore,
    foreach_snapshot,
)

_NORM_LAYERS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
)


def save_bn_state(model: nn.Module) -> Dict[str, Tensor]:
    """Snapshot running statistics for all normalisation layers."""
    saved: Dict[str, Tensor] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, _NORM_LAYERS):
            continue
        if hasattr(mod, "running_mean") and mod.running_mean is not None:
            saved[name + ".running_mean"] = mod.running_mean.clone()
        if hasattr(mod, "running_var") and mod.running_var is not None:
            saved[name + ".running_var"] = mod.running_var.clone()
        if hasattr(mod, "num_batches_tracked") and mod.num_batches_tracked is not None:
            saved[name + ".num_batches_tracked"] = mod.num_batches_tracked.clone()
    return saved


def restore_bn_state(model: nn.Module, saved: Dict[str, Tensor]) -> None:
    """Restore running statistics from a previous snapshot."""
    for name, mod in model.named_modules():
        if not isinstance(mod, _NORM_LAYERS):
            continue
        key_m = name + ".running_mean"
        key_v = name + ".running_var"
        key_n = name + ".num_batches_tracked"
        if key_m in saved:
            mod.running_mean.copy_(saved[key_m])
        if key_v in saved:
            mod.running_var.copy_(saved[key_v])
        if key_n in saved:
            mod.num_batches_tracked.copy_(saved[key_n])


def probe_offspring(
    param_data: List[Tensor],
    adam_dirs: List[Tensor],
    elites: List[Tensor],
    blend_alphas: Tensor,
    lr: float,
    probe_fn: Callable[[], Tensor],
    model: Optional[nn.Module] = None,
) -> Tuple[Tensor, int]:
    """Evaluate K offspring directions on the next batch.

    For each offspring k with blend coefficient alpha_k:
        d_k = (1 - alpha_k) * adam_dir + alpha_k * elite
        Apply theta + lr * d_k, evaluate probe_fn, restore theta.

    Args:
        param_data: list of param.data tensors (will be mutated then restored).
        adam_dirs: Adam direction per parameter.
        elites: elite direction per parameter.
        blend_alphas: 1-D tensor of K blend coefficients.
        lr: learning rate.
        probe_fn: callable returning scalar loss on the next batch.
        model: optional, for BatchNorm state save/restore.

    Returns:
        (losses, winner_idx) where losses is shape (K,) and winner_idx is int.
    """
    K = blend_alphas.shape[0]
    snapshot = foreach_snapshot(param_data)
    bn_state = save_bn_state(model) if model is not None else None

    losses = torch.empty(K, device=param_data[0].device)

    with torch.no_grad():
        for k in range(K):
            alpha = blend_alphas[k].item()
            d_k = foreach_lerp_directions(adam_dirs, elites, alpha)
            foreach_apply_direction(param_data, d_k, lr)
            losses[k] = probe_fn()
            foreach_restore(param_data, snapshot)
            if bn_state is not None:
                restore_bn_state(model, bn_state)

    winner_idx = int(losses.argmin().item())
    return losses, winner_idx
