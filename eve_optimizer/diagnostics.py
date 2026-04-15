"""Lightweight diagnostics for EVE optimizer internals.

Records only scalars and small lists per step -- no full parameter
tensors -- so overhead is negligible even for long runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class StepRecord:
    """Single optimization step snapshot."""

    step: int
    winner_idx: int
    offspring_losses: List[float]
    elite_adam_cosine: float
    elite_norm: float
    adam_norm: float


class DiagnosticsRecorder:
    """Accumulates per-step diagnostic records."""

    def __init__(self) -> None:
        self.records: List[StepRecord] = []

    def record(
        self,
        step: int,
        winner_idx: int,
        offspring_losses: Tensor,
        adam_dirs: List[Tensor],
        elites: List[Tensor],
    ) -> None:
        elite_flat = torch.cat([e.reshape(-1) for e in elites])
        adam_flat = torch.cat([a.reshape(-1) for a in adam_dirs])

        e_norm = elite_flat.norm().item()
        a_norm = adam_flat.norm().item()

        if e_norm > 0.0 and a_norm > 0.0:
            cosine = (
                torch.dot(elite_flat, adam_flat) / (e_norm * a_norm)
            ).item()
        else:
            cosine = 0.0

        self.records.append(
            StepRecord(
                step=step,
                winner_idx=winner_idx,
                offspring_losses=offspring_losses.detach().cpu().tolist(),
                elite_adam_cosine=cosine,
                elite_norm=e_norm,
                adam_norm=a_norm,
            )
        )

    def winner_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for r in self.records:
            counts[r.winner_idx] = counts.get(r.winner_idx, 0) + 1
        return counts

    def to_dataframe(self):
        """Convert to pandas DataFrame. Requires pandas."""
        import pandas as pd

        rows = []
        for r in self.records:
            row = {
                "step": r.step,
                "winner_idx": r.winner_idx,
                "elite_adam_cosine": r.elite_adam_cosine,
                "elite_norm": r.elite_norm,
                "adam_norm": r.adam_norm,
            }
            for i, loss in enumerate(r.offspring_losses):
                row[f"loss_k{i}"] = loss
            rows.append(row)
        return pd.DataFrame(rows)
