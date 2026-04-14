"""轻量级 dense Spatial GNN 组件，供规则网格版 TimeFilter 使用。"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGraphSAGELayer(nn.Module):
    """基于 dense adjacency 的 GraphSAGE mean aggregator。"""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        neighbor = torch.einsum("ij,bjd->bid", adj_norm, x)
        out = self.linear(torch.cat([x, neighbor], dim=-1))
        out = F.gelu(out)
        return self.dropout(out)


class DenseGCNLayer(nn.Module):
    """基于 dense adjacency 的 GCN layer。"""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("ij,bjd->bid", adj_norm, x)
        out = self.linear(out)
        out = F.gelu(out)
        return self.dropout(out)


class SpatialGraphEncoder(nn.Module):
    """逐时间步共享参数的空间 GNN 编码器。"""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        adjacency: torch.Tensor,
        gnn_type: Literal["graphsage", "gcn"] = "graphsage",
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("SpatialGraphEncoder 至少需要 1 层。")

        adjacency = adjacency.float()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency 必须是 [N, N] dense 矩阵。")

        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adj_row_norm", adjacency / degree, persistent=False)

        adjacency_with_self = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
        degree_with_self = adjacency_with_self.sum(dim=-1)
        inv_sqrt = degree_with_self.clamp_min(1.0).pow(-0.5)
        self.register_buffer(
            "adj_gcn_norm",
            inv_sqrt[:, None] * adjacency_with_self * inv_sqrt[None, :],
            persistent=False,
        )

        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.residual_norm = nn.LayerNorm(int(hidden_dim))
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.gnn_type = str(gnn_type)

        layer_cls = DenseGraphSAGELayer if self.gnn_type == "graphsage" else DenseGCNLayer
        self.layers = nn.ModuleList([layer_cls(int(hidden_dim), float(dropout)) for _ in range(int(num_layers))])

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, T, N, F] -> [B, T, N, H]"""

        batch_size, seq_len, num_nodes, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, num_nodes, -1)
        h0 = self.input_proj(x_flat)
        h = h0
        adj = self.adj_row_norm if self.gnn_type == "graphsage" else self.adj_gcn_norm
        for layer in self.layers:
            h = layer(h, adj)
        out = self.output_norm(self.residual_norm(h0) + h)

        if node_mask is not None:
            mask = node_mask.reshape(batch_size * seq_len, num_nodes, 1).float()
            out = out * mask
        return out.reshape(batch_size, seq_len, num_nodes, -1)
