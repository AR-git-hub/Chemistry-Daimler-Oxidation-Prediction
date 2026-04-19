"""Set models: self-attention, Transformer encoder, and Transformer + cross-attention to context.

All expose ``forward(x, mask, context)`` like ``DeepSetsRegressor`` (x: B×L×F, mask: B×L, context: B×C)."""

from __future__ import annotations

import torch
from torch import nn


def _key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """True where position is padding (PyTorch convention)."""
    return mask <= 0.0


class SetAttentionRegressor(nn.Module):
    """Single self-attention block + feed-forward + attention pooling + context head."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        d_model: int,
        nhead: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.ctx_enc = nn.Sequential(
            nn.Linear(context_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.self_attn = nn.MultiheadAttention(
            self.d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        head_in = self.d_model * 2 + 1
        self.head = nn.Sequential(
            nn.Linear(head_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        kpm = _key_padding_mask(mask)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ff(h))
        q = self.pool_query.expand(h.size(0), -1, -1)
        scores = (h * q).sum(dim=-1)
        scores = scores.masked_fill(kpm, -1e4)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        ctx_e = self.ctx_enc(context)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        feat = torch.cat([pooled, ctx_e, lengths.log1p()], dim=1)
        return self.head(feat)


class SetTransformerRegressor(nn.Module):
    """Transformer encoder (no positional encoding) + masked mean pool + context MLP."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.ctx_enc = nn.Sequential(
            nn.Linear(context_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        head_in = self.d_model * 2 + 1
        self.head = nn.Sequential(
            nn.Linear(head_in, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        kpm = _key_padding_mask(mask)
        h = self.encoder(h, src_key_padding_mask=kpm)
        kpm3 = kpm.unsqueeze(-1)
        h_masked = h.masked_fill(kpm3, 0.0)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = h_masked.sum(dim=1) / lengths
        ctx_e = self.ctx_enc(context)
        feat = torch.cat([pooled, ctx_e, lengths.log1p()], dim=1)
        return self.head(feat)


class SetTransformerMeanMaxRegressor(nn.Module):
    """Transformer encoder + **masked mean and max pool** (no cross-attention to context).

    Different graph from ``transformer_attention_max``: no context-as-query path; extrema
    come only from set self-attention then global max/mean readout.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.ctx_enc = nn.Sequential(
            nn.Linear(context_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        head_in = self.d_model * 3 + 1
        self.head = nn.Sequential(
            nn.Linear(head_in, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        kpm = _key_padding_mask(mask)
        h = self.encoder(h, src_key_padding_mask=kpm)
        kpm3 = kpm.unsqueeze(-1)
        h_masked = h.masked_fill(kpm3, 0.0)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = h_masked.sum(dim=1) / lengths
        minus_inf = torch.finfo(h.dtype).min
        h_for_max = h.masked_fill(kpm3, minus_inf)
        max_pool = h_for_max.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        ctx_e = self.ctx_enc(context)
        feat = torch.cat([mean_pool, max_pool, ctx_e, lengths.log1p()], dim=1)
        return self.head(feat)


class SetTransformerCrossAttentionRegressor(nn.Module):
    """Transformer encoder + one cross-attention (context queries set) + concat head."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.ctx_to_query = nn.Sequential(
            nn.Linear(context_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.cross_attn = nn.MultiheadAttention(
            self.d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(self.d_model)
        head_in = self.d_model * 2 + 1
        self.head = nn.Sequential(
            nn.Linear(head_in, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        kpm = _key_padding_mask(mask)
        h = self.encoder(h, src_key_padding_mask=kpm)
        q = self.ctx_to_query(context).unsqueeze(1)
        cross_out, _ = self.cross_attn(
            query=q,
            key=h,
            value=h,
            key_padding_mask=kpm,
            need_weights=False,
        )
        pooled = self.norm_cross(cross_out.squeeze(1))
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ctx_e = self.ctx_to_query(context)
        feat = torch.cat([pooled, ctx_e, lengths.log1p()], dim=1)
        return self.head(feat)


class SetTransformerCrossAttentionMaxRegressor(nn.Module):
    """Encoder + cross-attention readout **plus** masked mean/max pools (DeepSets-style global extrema).

    Concatenates context-query cross pool, set mean pool, set max pool, context embedding, and log set size.
    Stronger inductive bias for heterogeneous component magnitudes than cross-attention alone.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.ctx_to_query = nn.Sequential(
            nn.Linear(context_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.cross_attn = nn.MultiheadAttention(
            self.d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(self.d_model)
        head_in = self.d_model * 4 + 1
        self.head = nn.Sequential(
            nn.Linear(head_in, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        kpm = _key_padding_mask(mask)
        h = self.encoder(h, src_key_padding_mask=kpm)
        kpm3 = kpm.unsqueeze(-1)
        h_masked = h.masked_fill(kpm3, 0.0)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = h_masked.sum(dim=1) / lengths
        minus_inf = torch.finfo(h.dtype).min
        h_for_max = h.masked_fill(kpm3, minus_inf)
        max_pool = h_for_max.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        ctx_e = self.ctx_to_query(context)
        q = ctx_e.unsqueeze(1)
        cross_out, _ = self.cross_attn(
            query=q,
            key=h,
            value=h,
            key_padding_mask=kpm,
            need_weights=False,
        )
        pooled_cross = self.norm_cross(cross_out.squeeze(1))
        feat = torch.cat([pooled_cross, mean_pool, max_pool, ctx_e, lengths.log1p()], dim=1)
        return self.head(feat)


def _nhead_for_d_model(d_model: int, requested: int) -> int:
    d = int(d_model)
    h = max(1, min(int(requested), d // 2))
    while h > 1 and d % h != 0:
        h -= 1
    return h


def build_set_regressor(
    architecture: str,
    *,
    input_dim: int,
    context_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    use_heterogeneity: bool,
    encoder_hidden_dim: int | None,
    rho_hidden_dim: int | None,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_mult: int,
) -> nn.Module:
    """Dispatch by ``architecture`` string (matches ``metadata.json``)."""
    arch = architecture.lower().strip()
    if arch in ("deepsets", "deep_sets", "deepset"):
        from .model import DeepSetsRegressor

        return DeepSetsRegressor(
            input_dim=input_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            use_heterogeneity=use_heterogeneity,
            encoder_hidden_dim=encoder_hidden_dim,
            rho_hidden_dim=rho_hidden_dim,
        )
    if arch in ("deepsets_sumpool", "deepsets_sum_pool", "deepset_sumpool"):
        from .model import DeepSetsWithSumPoolRegressor

        return DeepSetsWithSumPoolRegressor(
            input_dim=input_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            use_heterogeneity=use_heterogeneity,
            encoder_hidden_dim=encoder_hidden_dim,
            rho_hidden_dim=rho_hidden_dim,
        )
    d_model = int(hidden_dim)
    nhead = _nhead_for_d_model(d_model, transformer_heads)
    ffn = max(32, int(transformer_ffn_mult) * d_model)
    drop = float(dropout)
    if arch in ("attention", "attn", "self_attention"):
        return SetAttentionRegressor(
            input_dim=input_dim,
            context_dim=context_dim,
            d_model=d_model,
            nhead=nhead,
            dropout=drop,
            output_dim=output_dim,
        )
    if arch in ("transformer", "set_transformer"):
        return SetTransformerRegressor(
            input_dim=input_dim,
            context_dim=context_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(transformer_layers),
            dim_feedforward=ffn,
            dropout=drop,
            output_dim=output_dim,
        )
    if arch in ("transformer_meanmax", "transformer_mean_max", "set_transformer_meanmax"):
        return SetTransformerMeanMaxRegressor(
            input_dim=input_dim,
            context_dim=context_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(transformer_layers),
            dim_feedforward=ffn,
            dropout=drop,
            output_dim=output_dim,
        )
    if arch in ("transformer_attention", "transformer_attn", "set_transformer_cross"):
        return SetTransformerCrossAttentionRegressor(
            input_dim=input_dim,
            context_dim=context_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(transformer_layers),
            dim_feedforward=ffn,
            dropout=drop,
            output_dim=output_dim,
        )
    if arch in ("transformer_attention_max", "transformer_attn_max", "set_transformer_cross_max"):
        return SetTransformerCrossAttentionMaxRegressor(
            input_dim=input_dim,
            context_dim=context_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(transformer_layers),
            dim_feedforward=ffn,
            dropout=drop,
            output_dim=output_dim,
        )
    raise ValueError(f"Unknown architecture: {architecture!r}")
