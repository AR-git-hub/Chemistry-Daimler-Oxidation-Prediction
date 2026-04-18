"""Deep Sets model and dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class ScenarioSetDataset(Dataset):
    """Dataset of variable-length component sets."""

    def __init__(
        self,
        components: Sequence[np.ndarray],
        context: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> None:
        self.components = [torch.from_numpy(x.astype(np.float32)) for x in components]
        self.context = torch.from_numpy(context.astype(np.float32))
        self.targets = None if targets is None else torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, idx: int):
        x = self.components[idx]
        c = self.context[idx]
        if self.targets is None:
            return x, c
        return x, c, self.targets[idx]


def collate_train(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-size scenario sets for train/validation."""
    xs, cs, ys = zip(*batch)
    max_len = max(t.shape[0] for t in xs)
    feat_dim = xs[0].shape[1]
    batch_size = len(xs)

    padded = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, tensor in enumerate(xs):
        cur_len = tensor.shape[0]
        padded[i, :cur_len] = tensor
        mask[i, :cur_len] = 1.0
    targets = torch.stack(list(ys), dim=0)
    context = torch.stack(list(cs), dim=0)
    return padded, mask, context, targets


def collate_infer(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-size scenario sets for inference."""
    xs, cs = zip(*batch)
    max_len = max(t.shape[0] for t in xs)
    feat_dim = xs[0].shape[1]
    batch_size = len(xs)

    padded = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, tensor in enumerate(xs):
        cur_len = tensor.shape[0]
        padded[i, :cur_len] = tensor
        mask[i, :cur_len] = 1.0
    context = torch.stack(list(cs), dim=0)
    return padded, mask, context


class DeepSetsRegressor(nn.Module):
    """Permutation-invariant regressor for scenario component sets.

    Use ``output_dim=1`` when training one network per target; ``output_dim=2`` for a
    legacy joint head (older checkpoints).
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.0,
        use_heterogeneity: bool = False,
    ) -> None:
        super().__init__()
        self.use_heterogeneity = bool(use_heterogeneity)
        dp = float(dropout)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
        )
        rho_in = hidden_dim * 3 + 1 + (1 if self.use_heterogeneity else 0)
        self.rho = nn.Sequential(
            nn.Linear(rho_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        mask3d = mask.unsqueeze(-1)
        h_masked = h * mask3d

        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_mean = h_masked.sum(dim=1) / lengths

        minus_inf = torch.finfo(h.dtype).min
        h_for_max = h.masked_fill(mask3d == 0, minus_inf)
        pooled_max = h_for_max.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        size_feature = lengths / lengths.max().clamp_min(1.0)
        context_emb = self.context_encoder(context)
        parts = [pooled_mean, pooled_max, context_emb, size_feature]
        if self.use_heterogeneity:
            ex2 = h_masked.pow(2).sum(dim=1) / lengths
            pooled_var = (ex2 - pooled_mean.pow(2)).clamp_min(0.0)
            hetero = torch.sqrt(pooled_var.mean(dim=1, keepdim=True) + 1e-8)
            parts.append(hetero)
        pooled = torch.cat(parts, dim=1)
        return self.rho(pooled)

