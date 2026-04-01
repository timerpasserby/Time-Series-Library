"""MS-TimeFilter 正式实验模型。
相关组件：models/components/weather_encoder.py、models/components/ms_timefilter_graph.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from layers.TimeFilter_layers import TimeFilter_Backbone
from models.TimeFilter import PatchEmbed
from models.components.ms_timefilter_graph import EDDRLayer, PGGCLayer, build_prior_graph_buffers
from models.components.weather_encoder import WeatherEncoder


class MSTimeFilter(nn.Module):
    """保留 TimeFilter temporal backbone 的多分支正式模型。"""

    def __init__(
        self,
        *,
        experiment_id: str,
        patch_meta: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        patch_count: int,
        internal_dim: int,
        weather_dim: int,
        blast_v2_dim: int,
        blast_v3_dim: int,
        model_cfg: dict[str, Any],
        graph_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.experiment_id = str(experiment_id)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.patch_count = int(patch_count)
        self.internal_dim = int(internal_dim)
        self.weather_dim = int(weather_dim)
        self.blast_v2_dim = int(blast_v2_dim)
        self.blast_v3_dim = int(blast_v3_dim)
        self.d_model = int(model_cfg["d_model"])
        self.d_ff = int(model_cfg["d_ff"])
        self.e_layers = int(model_cfg["e_layers"])
        self.dropout = float(model_cfg["dropout"])
        self.n_heads = int(model_cfg["n_heads"])
        self.patch_len = int(model_cfg["patch_len"])
        self.stride = int(model_cfg["stride"])
        self.alpha = float(model_cfg["alpha"])
        self.top_p = float(model_cfg["top_p"])
        self.time_block_count = self.seq_len // self.patch_len
        self.num_tokens = self.patch_count * self.time_block_count

        if self.seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={self.seq_len} 必须被 patch_len={self.patch_len} 整除。")

        self.use_weather = self.experiment_id in {"A2", "A3", "A4", "C1", "C2", "C3"}
        self.use_blast_v2 = self.experiment_id == "A3"
        self.use_blast_v3 = self.experiment_id in {"A4", "C1", "C2", "C3"}
        self.use_pggc = self.experiment_id in {"C1", "C2", "C3"}
        self.use_eddr = self.experiment_id in {"C2", "C3"}
        self.use_pir = self.experiment_id == "C3"

        self.internal_proj = nn.Linear(self.internal_dim, 1)
        with torch.no_grad():
            self.internal_proj.weight.zero_()
            self.internal_proj.bias.zero_()
            self.internal_proj.weight[0, 0] = 1.0

        self.patch_embed = PatchEmbed(self.d_model, self.patch_len, self.stride, pos=True)
        self.fusion_norm = nn.LayerNorm(self.d_model)
        self.backbone = TimeFilter_Backbone(
            self.d_model,
            self.patch_count,
            self.d_ff,
            self.n_heads,
            self.e_layers,
            self.top_p,
            self.dropout,
            in_dim=self.num_tokens,
        )
        self.head = nn.Linear(self.d_model * self.time_block_count, self.pred_len)

        self.weather_encoder = WeatherEncoder(
            input_dim=self.weather_dim,
            d_model=self.d_model,
            patch_len=self.patch_len,
        )
        self.blast_v2_proj = nn.Linear(self.blast_v2_dim, self.d_model)
        self.blast_v3_proj = nn.Linear(self.blast_v3_dim, self.d_model)

        graph_buffers = build_prior_graph_buffers(
            patch_meta,
            time_block_count=self.time_block_count,
            w_s=float(graph_cfg["w_s"]),
            w_t=float(graph_cfg["w_t"]),
        )
        self.pggc = PGGCLayer(
            hidden_dim=self.d_model,
            lambda_prior=float(graph_cfg["lambda_prior"]),
            top_k=int(graph_cfg["top_k"]),
            learned_chunk_size=int(graph_cfg["learned_chunk_size"]),
            graph_buffers=graph_buffers,
        )
        self.eddr = EDDRLayer(self.d_model)
        self._mask_cache: dict[str, torch.Tensor] = {}

    def _get_backbone_mask(self, device: torch.device) -> torch.Tensor:
        cache_key = str(device)
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached

        dtype = torch.float32
        token_count = self.num_tokens
        block_count = self.time_block_count
        masks = []
        for token_idx in range(token_count):
            spatial = ((torch.arange(token_count) % block_count == token_idx % block_count) & (torch.arange(token_count) != token_idx)).to(dtype).to(device)
            temporal = (
                (torch.arange(token_count) >= token_idx // block_count * block_count)
                & (torch.arange(token_count) < token_idx // block_count * block_count + block_count)
                & (torch.arange(token_count) != token_idx)
            ).to(dtype).to(device)
            rest = torch.ones(token_count, dtype=dtype, device=device) - spatial - temporal
            rest[token_idx] = 0.0
            masks.append(torch.stack([spatial, temporal, rest], dim=0))
        tensor = torch.stack(masks, dim=0)
        self._mask_cache[cache_key] = tensor
        return tensor

    def _build_token_mask(self, enc_mask: torch.Tensor) -> torch.Tensor:
        batch_size = enc_mask.shape[0]
        token_mask = enc_mask.permute(0, 2, 1).reshape(
            batch_size,
            self.patch_count,
            self.time_block_count,
            self.patch_len,
        )
        token_mask = token_mask.amax(dim=-1).reshape(batch_size, self.num_tokens, 1)
        return token_mask

    def _broadcast_block_embed(self, block_embed: torch.Tensor) -> torch.Tensor:
        batch_size = block_embed.shape[0]
        expanded = block_embed[:, None, :, :].expand(batch_size, self.patch_count, self.time_block_count, self.d_model)
        return expanded.reshape(batch_size, self.num_tokens, self.d_model)

    def _pool_global_branch(self, sequence_embed: torch.Tensor) -> torch.Tensor:
        batch_size = sequence_embed.shape[0]
        return sequence_embed.reshape(batch_size, self.time_block_count, self.patch_len, self.d_model).mean(dim=2)

    def _pool_patch_branch(self, sequence_embed: torch.Tensor) -> torch.Tensor:
        batch_size = sequence_embed.shape[0]
        patch_embed = sequence_embed.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.patch_count,
            self.time_block_count,
            self.patch_len,
            self.d_model,
        )
        return patch_embed.mean(dim=3).reshape(batch_size, self.num_tokens, self.d_model)

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> dict[str, Any]:
        internal_seq = batch["internal_seq"]
        weather_seq = batch["weather_seq"]
        blast_v2_seq = batch["blast_v2_seq"]
        blast_v3_seq = batch["blast_v3_seq"]
        enc_mask = batch["enc_mask"]

        batch_size = internal_seq.shape[0]
        internal_scalar = self.internal_proj(internal_seq).squeeze(-1) * enc_mask
        internal_tokens = self.patch_embed(internal_scalar.permute(0, 2, 1).reshape(batch_size, -1))
        token_mask = self._build_token_mask(enc_mask)
        internal_tokens = internal_tokens * token_mask

        aux: dict[str, Any] = {}
        weather_tokens = torch.zeros_like(internal_tokens)
        blast_v2_tokens = torch.zeros_like(internal_tokens)
        blast_v3_tokens = torch.zeros_like(internal_tokens)

        if self.use_weather:
            weather_outputs = self.weather_encoder(weather_seq)
            weather_tokens = self._broadcast_block_embed(weather_outputs["weather_block_embed"]) * token_mask
            aux["weather_global_embed"] = weather_outputs["weather_global_embed"]
            aux["weather_attention_weights"] = weather_outputs["attention_weights"]
            aux["weather_debug_shapes"] = weather_outputs["debug_shapes"]

        if self.use_blast_v2:
            blast_v2_block_embed = self._pool_global_branch(self.blast_v2_proj(blast_v2_seq))
            blast_v2_tokens = self._broadcast_block_embed(blast_v2_block_embed) * token_mask

        if self.use_blast_v3 or self.use_eddr:
            blast_v3_tokens = self._pool_patch_branch(self.blast_v3_proj(blast_v3_seq)) * token_mask

        fused = internal_tokens + weather_tokens + blast_v2_tokens + blast_v3_tokens
        fused = self.fusion_norm(fused)

        graph_state: dict[str, Any] | None = None
        if self.use_pggc:
            fused, graph_state = self.pggc(fused, return_debug=return_aux)
            aux["graph_state"] = graph_state

        if self.use_eddr:
            fused, gate_weights = self.eddr(fused, blast_v3_tokens, graph_state or {})
            aux["gate_weights"] = gate_weights

        backbone_out, moe_loss = self.backbone(fused, self._get_backbone_mask(fused.device), self.alpha)
        pred_norm = self.head(
            backbone_out.reshape(batch_size, self.patch_count, self.time_block_count, self.d_model).flatten(start_dim=-2)
        ).permute(0, 2, 1)

        aux["loss_moe"] = moe_loss
        aux["token_mask"] = token_mask
        aux["internal_tokens"] = internal_tokens if return_aux else None
        return {"pred_norm": pred_norm, "aux": aux}
