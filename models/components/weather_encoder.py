"""正式天气分支编码器。
相关模块：models/ms_timefilter.py、scripts/slopemine_v2/run_ms_timefilter_experiments_v1.py
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherEncoder(nn.Module):
    """Linear Projection + Multi-scale Conv1D + Temporal Attention + Time-block Pooling。"""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        patch_len: int,
        kernel_sizes: tuple[int, int, int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)

        self.proj = nn.Linear(self.input_dim, self.d_model)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.fuse = nn.Linear(self.d_model * len(self.kernel_sizes), self.d_model)
        self.attn_hidden = nn.Linear(self.d_model, self.d_model)
        self.attn_score = nn.Linear(self.d_model, 1)

    def forward(self, weather_seq: torch.Tensor) -> dict[str, Any]:
        """返回 block-level 天气嵌入、全局天气嵌入和注意力权重。"""

        if weather_seq.ndim != 3:
            raise ValueError(f"weather_seq 期望为 [B, L, C]，收到 shape={tuple(weather_seq.shape)}")

        batch_size, seq_len, _ = weather_seq.shape
        if seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={seq_len} 不能被 patch_len={self.patch_len} 整除。")

        projected = self.proj(weather_seq)
        conv_input = projected.transpose(1, 2)
        conv_outputs = [F.gelu(conv(conv_input)).transpose(1, 2) for conv in self.convs]
        fused = self.fuse(torch.cat(conv_outputs, dim=-1))

        attn_logits = self.attn_score(torch.tanh(self.attn_hidden(fused))).squeeze(-1)
        attention_weights = torch.softmax(attn_logits, dim=1)
        weather_global_embed = torch.sum(fused * attention_weights.unsqueeze(-1), dim=1)

        block_count = seq_len // self.patch_len
        weather_block_embed = fused.reshape(batch_size, block_count, self.patch_len, self.d_model).mean(dim=2)

        debug_shapes = {
            "weather_seq": tuple(weather_seq.shape),
            "projected": tuple(projected.shape),
            "conv_k3": tuple(conv_outputs[0].shape),
            "conv_k5": tuple(conv_outputs[1].shape),
            "conv_k7": tuple(conv_outputs[2].shape),
            "fused": tuple(fused.shape),
            "weather_block_embed": tuple(weather_block_embed.shape),
            "weather_global_embed": tuple(weather_global_embed.shape),
            "attention_weights": tuple(attention_weights.shape),
        }

        return {
            "weather_block_embed": weather_block_embed,
            "weather_global_embed": weather_global_embed,
            "attention_weights": attention_weights,
            "debug_shapes": debug_shapes,
        }
