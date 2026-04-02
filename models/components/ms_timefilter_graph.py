"""MS-TimeFilter 的 PGGC / EDDR 稀疏图组件。
相关模块：models/ms_timefilter.py、scripts/slopemine_v2/run_ms_timefilter_experiments_v1.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GraphBuffers:
    spatial_indices: torch.Tensor
    spatial_logits: torch.Tensor
    spatial_mask: torch.Tensor
    temporal_indices: torch.Tensor
    temporal_logits: torch.Tensor
    temporal_mask: torch.Tensor
    prior_indices: torch.Tensor
    prior_logits: torch.Tensor
    prior_mask: torch.Tensor


def _pack_neighbor_lists(
    neighbors: list[list[int]],
    logits: list[list[float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_degree = max(1, max(len(items) for items in neighbors))
    row_count = len(neighbors)

    index_tensor = torch.zeros((row_count, max_degree), dtype=torch.long)
    logit_tensor = torch.zeros((row_count, max_degree), dtype=torch.float32)
    mask_tensor = torch.zeros((row_count, max_degree), dtype=torch.bool)
    for row_idx, (cols, values) in enumerate(zip(neighbors, logits)):
        degree = len(cols)
        if degree == 0:
            continue
        index_tensor[row_idx, :degree] = torch.tensor(cols, dtype=torch.long)
        logit_tensor[row_idx, :degree] = torch.tensor(values, dtype=torch.float32)
        mask_tensor[row_idx, :degree] = True
    return index_tensor, logit_tensor, mask_tensor


def build_prior_graph_buffers(
    patch_meta: pd.DataFrame,
    *,
    time_block_count: int,
    w_s: float = 1.0,
    w_t: float = 1.0,
) -> GraphBuffers:
    """构造 patch-major / block-minor 顺序下的时空先验图。"""

    patch_meta = patch_meta.reset_index(drop=True).copy()
    patch_lookup = {
        (int(row.patch_x_idx), int(row.patch_y_idx), int(row.zone_id_v2)): int(row_idx)
        for row_idx, row in enumerate(patch_meta.itertuples(index=False))
    }

    spatial_patch_neighbors: list[list[int]] = [[] for _ in range(len(patch_meta))]
    for row_idx, row in enumerate(patch_meta.itertuples(index=False)):
        x_idx = int(row.patch_x_idx)
        y_idx = int(row.patch_y_idx)
        zone_id = int(row.zone_id_v2)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_key = (x_idx + dx, y_idx + dy, zone_id)
            if neighbor_key in patch_lookup:
                spatial_patch_neighbors[row_idx].append(patch_lookup[neighbor_key])

    token_count = len(patch_meta) * int(time_block_count)
    spatial_neighbors: list[list[int]] = [[] for _ in range(token_count)]
    temporal_neighbors: list[list[int]] = [[] for _ in range(token_count)]
    prior_neighbors: list[list[int]] = [[] for _ in range(token_count)]
    spatial_logits: list[list[float]] = [[] for _ in range(token_count)]
    temporal_logits: list[list[float]] = [[] for _ in range(token_count)]
    prior_logits: list[list[float]] = [[] for _ in range(token_count)]

    for patch_idx in range(len(patch_meta)):
        for block_idx in range(int(time_block_count)):
            token_idx = patch_idx * int(time_block_count) + block_idx
            for neighbor_patch_idx in spatial_patch_neighbors[patch_idx]:
                neighbor_token_idx = neighbor_patch_idx * int(time_block_count) + block_idx
                spatial_neighbors[token_idx].append(neighbor_token_idx)
                spatial_logits[token_idx].append(float(w_s))
                prior_neighbors[token_idx].append(neighbor_token_idx)
                prior_logits[token_idx].append(float(w_s))

            for offset in (-1, 1):
                next_block = block_idx + offset
                if next_block < 0 or next_block >= int(time_block_count):
                    continue
                neighbor_token_idx = patch_idx * int(time_block_count) + next_block
                temporal_neighbors[token_idx].append(neighbor_token_idx)
                temporal_logits[token_idx].append(float(w_t))
                prior_neighbors[token_idx].append(neighbor_token_idx)
                prior_logits[token_idx].append(float(w_t))

    spatial_indices, spatial_values, spatial_mask = _pack_neighbor_lists(spatial_neighbors, spatial_logits)
    temporal_indices, temporal_values, temporal_mask = _pack_neighbor_lists(temporal_neighbors, temporal_logits)
    prior_indices, prior_values, prior_mask = _pack_neighbor_lists(prior_neighbors, prior_logits)
    return GraphBuffers(
        spatial_indices=spatial_indices,
        spatial_logits=spatial_values,
        spatial_mask=spatial_mask,
        temporal_indices=temporal_indices,
        temporal_logits=temporal_values,
        temporal_mask=temporal_mask,
        prior_indices=prior_indices,
        prior_logits=prior_values,
        prior_mask=prior_mask,
    )


def _expand_graph_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).expand(batch_size, -1, -1)
    return tensor


def sparse_row_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
    weights = torch.softmax(masked_logits, dim=-1) * mask.float()
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights / denom


def sparse_message_passing(
    x: torch.Tensor,
    *,
    neighbor_indices: torch.Tensor,
    neighbor_logits: torch.Tensor,
    neighbor_mask: torch.Tensor,
    message_proj: nn.Linear,
    self_proj: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, hidden_dim = x.shape
    neighbor_indices = _expand_graph_tensor(neighbor_indices, batch_size)
    neighbor_logits = _expand_graph_tensor(neighbor_logits, batch_size)
    neighbor_mask = _expand_graph_tensor(neighbor_mask, batch_size)

    alpha = sparse_row_softmax(neighbor_logits, neighbor_mask)
    projected = message_proj(x)
    self_term = self_proj(x)
    batch_index = torch.arange(batch_size, device=x.device)[:, None, None]
    neighbor_states = projected[batch_index, neighbor_indices]
    aggregated = (alpha.unsqueeze(-1) * neighbor_states).sum(dim=2)
    return F.gelu(aggregated + self_term), alpha


def chunked_topk_similarity(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按行块计算精确 top-k 相似度，避免一次性构造完整 dense score。"""

    batch_size, token_count, hidden_dim = query.shape
    top_k = max(1, min(int(top_k), token_count - 1))
    scale = 1.0 / math.sqrt(float(hidden_dim))
    key_t = key.transpose(1, 2)

    value_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    for start in range(0, token_count, int(chunk_size)):
        end = min(token_count, start + int(chunk_size))
        scores = torch.matmul(query[:, start:end, :], key_t) * scale
        row_ids = torch.arange(start, end, device=query.device)
        local_ids = torch.arange(end - start, device=query.device)
        scores[:, local_ids, row_ids] = -1e9
        top_values, top_indices = torch.topk(scores, k=top_k, dim=-1)
        value_chunks.append(top_values)
        index_chunks.append(top_indices)
    return torch.cat(value_chunks, dim=1), torch.cat(index_chunks, dim=1)


def build_dense_debug_matrix(
    *,
    token_count: int,
    neighbor_indices: torch.Tensor,
    neighbor_logits: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """仅用于调试输出，把稀疏邻接恢复成 dense 热图。"""

    weights = sparse_row_softmax(
        _expand_graph_tensor(neighbor_logits, 1),
        _expand_graph_tensor(neighbor_mask, 1),
    )[0]
    indices = _expand_graph_tensor(neighbor_indices, 1)[0]
    mask = _expand_graph_tensor(neighbor_mask, 1)[0]
    dense = torch.zeros((token_count, token_count), dtype=torch.float32)
    for row_idx in range(token_count):
        cols = indices[row_idx, mask[row_idx]]
        vals = weights[row_idx, mask[row_idx]]
        if len(cols) > 0:
            dense[row_idx, cols.cpu()] += vals.cpu()
    return dense


class PGGCLayer(nn.Module):
    """物理制导图构建与一次图传播。"""

    def __init__(
        self,
        *,
        hidden_dim: int,
        lambda_prior: float,
        top_k: int,
        learned_chunk_size: int,
        graph_buffers: GraphBuffers,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lambda_prior = float(lambda_prior)
        self.top_k = int(top_k)
        self.learned_chunk_size = int(learned_chunk_size)

        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.message_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.register_buffer("spatial_indices", graph_buffers.spatial_indices, persistent=False)
        self.register_buffer("spatial_logits", graph_buffers.spatial_logits, persistent=False)
        self.register_buffer("spatial_mask", graph_buffers.spatial_mask, persistent=False)
        self.register_buffer("temporal_indices", graph_buffers.temporal_indices, persistent=False)
        self.register_buffer("temporal_logits", graph_buffers.temporal_logits, persistent=False)
        self.register_buffer("temporal_mask", graph_buffers.temporal_mask, persistent=False)
        self.register_buffer("prior_indices", graph_buffers.prior_indices, persistent=False)
        self.register_buffer("prior_logits", graph_buffers.prior_logits, persistent=False)
        self.register_buffer("prior_mask", graph_buffers.prior_mask, persistent=False)

    def forward(self, x: torch.Tensor, return_debug: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        learned_values, learned_indices = chunked_topk_similarity(
            self.query_proj(x),
            self.key_proj(x),
            top_k=self.top_k,
            chunk_size=self.learned_chunk_size,
        )
        learned_mask = torch.ones_like(learned_indices, dtype=torch.bool)

        batch_size, token_count, _ = x.shape
        prior_indices = self.prior_indices.unsqueeze(0).expand(batch_size, -1, -1)
        prior_logits = (self.lambda_prior * self.prior_logits).unsqueeze(0).expand(batch_size, -1, -1)
        prior_mask = self.prior_mask.unsqueeze(0).expand(batch_size, -1, -1)

        final_indices = torch.cat([learned_indices, prior_indices], dim=-1)
        final_logits = torch.cat([learned_values, prior_logits], dim=-1)
        final_mask = torch.cat([learned_mask, prior_mask], dim=-1)

        propagated, final_alpha = sparse_message_passing(
            x,
            neighbor_indices=final_indices,
            neighbor_logits=final_logits,
            neighbor_mask=final_mask,
            message_proj=self.message_proj,
            self_proj=self.self_proj,
        )

        graph_state: dict[str, Any] = {
            "spatial_indices": self.spatial_indices,
            "spatial_logits": self.spatial_logits,
            "spatial_mask": self.spatial_mask,
            "temporal_indices": self.temporal_indices,
            "temporal_logits": self.temporal_logits,
            "temporal_mask": self.temporal_mask,
            "learned_indices": learned_indices,
            "learned_logits": learned_values,
            "learned_mask": learned_mask,
            "final_indices": final_indices,
            "final_logits": final_logits,
            "final_mask": final_mask,
            "final_alpha": final_alpha,
        }
        if return_debug:
            graph_state["a_prior_dense"] = build_dense_debug_matrix(
                token_count=token_count,
                neighbor_indices=self.prior_indices,
                neighbor_logits=self.lambda_prior * self.prior_logits,
                neighbor_mask=self.prior_mask,
            )
            graph_state["a_learned_dense"] = build_dense_debug_matrix(
                token_count=token_count,
                neighbor_indices=learned_indices[0].detach().cpu(),
                neighbor_logits=learned_values[0].detach().cpu(),
                neighbor_mask=learned_mask[0].detach().cpu(),
            )
            graph_state["a_final_dense"] = build_dense_debug_matrix(
                token_count=token_count,
                neighbor_indices=final_indices[0].detach().cpu(),
                neighbor_logits=final_logits[0].detach().cpu(),
                neighbor_mask=final_mask[0].detach().cpu(),
            )
        return propagated, graph_state


class EDDRLayer(nn.Module):
    """事件驱动的三专家动态路由。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.gate_from_x = nn.Linear(self.hidden_dim, 3)
        self.gate_from_blast = nn.Linear(self.hidden_dim, 3)

        self.spatial_message_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.spatial_self_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.temporal_message_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.temporal_self_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.st_message_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.st_self_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        blast_embed: torch.Tensor,
        graph_state: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate_from_x(x) + self.gate_from_blast(blast_embed)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        spatial_out, _ = sparse_message_passing(
            x,
            neighbor_indices=graph_state["spatial_indices"],
            neighbor_logits=graph_state["spatial_logits"],
            neighbor_mask=graph_state["spatial_mask"],
            message_proj=self.spatial_message_proj,
            self_proj=self.spatial_self_proj,
        )
        temporal_out, _ = sparse_message_passing(
            x,
            neighbor_indices=graph_state["temporal_indices"],
            neighbor_logits=graph_state["temporal_logits"],
            neighbor_mask=graph_state["temporal_mask"],
            message_proj=self.temporal_message_proj,
            self_proj=self.temporal_self_proj,
        )
        st_out, _ = sparse_message_passing(
            x,
            neighbor_indices=graph_state["final_indices"],
            neighbor_logits=graph_state["final_logits"],
            neighbor_mask=graph_state["final_mask"],
            message_proj=self.st_message_proj,
            self_proj=self.st_self_proj,
        )

        expert_outputs = torch.stack([spatial_out, temporal_out, st_out], dim=-2)
        fused = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=-2)
        return fused, gate_weights
