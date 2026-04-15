"""Vectorized parameter operations using torch._foreach_*.

Every function here operates on lists of tensors (one per parameter),
eliminating Python-level iteration over individual parameters.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def foreach_moment_update(
    ms: List[Tensor],
    vs: List[Tensor],
    grads: List[Tensor],
    beta1: float,
    beta2: float,
) -> None:
    """In-place EMA update for first and second moment buffers."""
    torch._foreach_mul_(ms, beta1)
    torch._foreach_add_(ms, grads, alpha=1.0 - beta1)
    torch._foreach_mul_(vs, beta2)
    torch._foreach_addcmul_(vs, grads, grads, value=1.0 - beta2)


def foreach_adam_direction(
    ms: List[Tensor],
    vs: List[Tensor],
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
) -> List[Tensor]:
    """Compute bias-corrected Adam direction: -m_hat / (sqrt(v_hat) + eps).

    Returns a new list of tensors (does not mutate ms/vs).
    """
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    # m_hat = m / bc1
    m_hat = torch._foreach_div(ms, bc1)
    # v_hat = v / bc2
    v_hat = torch._foreach_div(vs, bc2)
    # sqrt_v_hat = sqrt(v_hat)
    sqrt_v_hat = torch._foreach_sqrt(v_hat)
    # denom = sqrt_v_hat + eps
    torch._foreach_add_(sqrt_v_hat, eps)
    # direction = -m_hat / denom
    torch._foreach_neg_(m_hat)
    torch._foreach_div_(m_hat, sqrt_v_hat)
    return m_hat


def foreach_lerp_directions(
    adam_dirs: List[Tensor],
    elites: List[Tensor],
    alpha: float,
) -> List[Tensor]:
    """Blend: (1 - alpha) * adam + alpha * elite.

    alpha=0 -> pure Adam.  alpha=1 -> pure elite.
    Returns a new list of tensors.
    """
    if alpha == 0.0:
        return [d.clone() for d in adam_dirs]
    if alpha == 1.0:
        return [e.clone() for e in elites]
    # result = (1 - alpha) * adam + alpha * elite
    scaled_adam = torch._foreach_mul(adam_dirs, 1.0 - alpha)
    scaled_elite = torch._foreach_mul(elites, alpha)
    torch._foreach_add_(scaled_adam, scaled_elite)
    return scaled_adam


def foreach_apply_direction(
    params: List[Tensor],
    dirs: List[Tensor],
    lr: float,
) -> None:
    """In-place: param += lr * direction."""
    torch._foreach_add_(params, dirs, alpha=lr)


def foreach_apply_weight_decay(
    params: List[Tensor],
    lr: float,
    wd: float,
) -> None:
    """In-place decoupled weight decay: param *= (1 - lr * wd)."""
    if wd == 0.0:
        return
    torch._foreach_mul_(params, 1.0 - lr * wd)


def foreach_snapshot(params: List[Tensor]) -> List[Tensor]:
    """Clone all parameter data tensors."""
    return [p.clone() for p in params]


def foreach_restore(params: List[Tensor], snapshot: List[Tensor]) -> None:
    """Restore parameter data from a snapshot (in-place copy)."""
    torch._foreach_copy_(params, snapshot)


def foreach_copy(dst: List[Tensor], src: List[Tensor]) -> None:
    """Copy src tensors into dst tensors (in-place)."""
    torch._foreach_copy_(dst, src)
