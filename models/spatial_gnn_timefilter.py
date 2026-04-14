"""规则网格版 SpatialGNN + TimeFilter。
输入形状约定：x=[B, T, N, F]，node_mask=[B, T, N]。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone
from models.components.spatial_gnn import SpatialGraphEncoder
from models.slopemine_formal_wrappers import resolve_patch_len


class LongTokenPatchEmbed(nn.Module):
    """支持长 token 序列的 patch embedding。"""

    def __init__(self, dim: int, patch_len: int, stride: int | None = None, pos: bool = True) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(patch_len if stride is None else stride)
        self.patch_proj = nn.Linear(self.patch_len, int(dim))
        self.use_pos = bool(pos)

    def _sinusoidal_position(self, length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype) * (-(torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim))
        )
        pe = torch.zeros(1, length, dim, device=device, dtype=dtype)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x)
        if self.use_pos:
            x = x + self._sinusoidal_position(x.size(1), x.size(2), x.device, x.dtype)
        return x


class SpatialGNNTimeFilter(nn.Module):
    """先做逐时间步空间 GNN，再送入原始 TimeFilter temporal backbone。"""

    def __init__(
        self,
        *,
        input_dim: int,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        adjacency: torch.Tensor,
        model_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(model_cfg["d_model"])
        self.d_ff = int(model_cfg["d_ff"])
        self.e_layers = int(model_cfg["e_layers"])
        self.dropout = float(model_cfg["dropout"])
        self.n_heads = int(model_cfg["n_heads"])
        self.alpha = float(model_cfg.get("alpha", 0.1))
        self.top_p = float(model_cfg.get("top_p", 0.5))
        self.patch_len = resolve_patch_len(self.seq_len, int(model_cfg.get("patch_len", 8)))
        self.stride = self.patch_len
        self.num_patches = self.seq_len // self.patch_len

        if self.seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={self.seq_len} 必须被 patch_len={self.patch_len} 整除。")

        self.spatial_encoder = SpatialGraphEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            adjacency=adjacency,
            gnn_type=str(model_cfg.get("gnn_type", "graphsage")),
            num_layers=int(model_cfg.get("gnn_layers", 2)),
            dropout=float(model_cfg.get("gnn_dropout", self.dropout)),
        )
        self.channel_readout = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.norm = Normalize(self.num_nodes, affine=False)
        self.patch_embed = LongTokenPatchEmbed(self.hidden_dim, self.patch_len, self.stride, pos=True)
        self.backbone = TimeFilter_Backbone(
            self.hidden_dim,
            self.num_nodes,
            self.d_ff,
            self.n_heads,
            self.e_layers,
            self.top_p,
            self.dropout,
            in_dim=self.seq_len * self.num_nodes // self.patch_len,
        )
        self.head = nn.Linear(self.hidden_dim * self.num_patches, self.pred_len)
        self._mask_cache: dict[str, torch.Tensor] = {}

    def _get_timefilter_mask(self, device: torch.device) -> torch.Tensor:
        cache_key = str(device)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        token_count = self.seq_len * self.num_nodes // self.patch_len
        time_patch_count = self.seq_len // self.patch_len
        dtype = torch.float32
        masks = []
        for token_idx in range(token_count):
            spatial = (
                (torch.arange(token_count, device=device) % time_patch_count == token_idx % time_patch_count)
                & (torch.arange(token_count, device=device) != token_idx)
            ).to(dtype)
            temporal = (
                (torch.arange(token_count, device=device) >= token_idx // time_patch_count * time_patch_count)
                & (torch.arange(token_count, device=device) < token_idx // time_patch_count * time_patch_count + time_patch_count)
                & (torch.arange(token_count, device=device) != token_idx)
            ).to(dtype)
            spatiotemporal = torch.ones(token_count, device=device, dtype=dtype) - spatial - temporal
            spatiotemporal[token_idx] = 0.0
            masks.append(torch.stack([spatial, temporal, spatiotemporal], dim=0))
        tensor = torch.stack(masks, dim=0)
        self._mask_cache[cache_key] = tensor
        return tensor

    def forward(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> dict[str, Any]:
        """x=[B, T, N, F], node_mask=[B, T, N]。"""

        if x.ndim != 4:
            raise ValueError(f"SpatialGNNTimeFilter expects x=[B,T,N,F], got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len or x.shape[2] != self.num_nodes:
            raise ValueError(
                f"Input shape mismatch: expected seq_len={self.seq_len}, num_nodes={self.num_nodes}, got {tuple(x.shape)}"
            )

        if node_mask is None:
            node_mask = torch.ones(x.shape[0], x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
        else:
            node_mask = node_mask.float()
            x = x * node_mask.unsqueeze(-1)

        spatial_embeddings = self.spatial_encoder(x, node_mask)
        scalar_series = self.channel_readout(spatial_embeddings).squeeze(-1) * node_mask
        scalar_series = self.norm(scalar_series, "norm")

        tokens = self.patch_embed(scalar_series.permute(0, 2, 1).reshape(x.shape[0], -1))
        backbone_out, moe_loss = self.backbone(tokens, self._get_timefilter_mask(tokens.device), self.alpha)
        pred = self.head(
            backbone_out.reshape(x.shape[0], self.num_nodes, self.num_patches, self.hidden_dim).flatten(start_dim=-2)
        ).permute(0, 2, 1)
        pred = self.norm(pred, "denorm")

        aux: dict[str, Any] = {
            "moe_loss": moe_loss,
            "scalar_series": scalar_series if return_aux else None,
            "spatial_embeddings": spatial_embeddings if return_aux else None,
        }
        return {
            "pred": pred,
            "aux": aux,
        }
