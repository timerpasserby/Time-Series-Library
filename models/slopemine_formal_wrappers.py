"""这个模块负责把 slopemine 正式 patch 输入包装到现有 TSLib 风格模型。
相关模型：LSTM、TCN、DLinear、PatchTST、STGCN、TimeFilter
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


def resolve_patch_len(seq_len: int, preferred: int) -> int:
    """选择一个能整除 seq_len 的 patch_len。"""

    if preferred > 0 and seq_len % preferred == 0:
        return preferred
    for candidate in range(min(preferred, seq_len), 1, -1):
        if seq_len % candidate == 0:
            return candidate
    return max(1, seq_len)


def build_model_configs(
    *,
    seq_len: int,
    pred_len: int,
    num_nodes: int,
    model_cfg: dict[str, Any],
) -> SimpleNamespace:
    """构造统一的模型配置对象。"""

    patch_len = resolve_patch_len(int(seq_len), int(model_cfg["patch_len"]))
    stride = resolve_patch_len(int(seq_len), int(model_cfg.get("stride", patch_len)))
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=int(seq_len),
        pred_len=int(pred_len),
        enc_in=int(num_nodes),
        dec_in=int(num_nodes),
        c_out=int(num_nodes),
        d_model=int(model_cfg["d_model"]),
        d_ff=int(model_cfg["d_ff"]),
        e_layers=int(model_cfg["e_layers"]),
        dropout=float(model_cfg["dropout"]),
        n_heads=int(model_cfg["n_heads"]),
        d_conv=int(model_cfg["d_conv"]),
        moving_avg=int(model_cfg["moving_avg"]),
        factor=int(model_cfg["factor"]),
        activation=str(model_cfg["activation"]),
        patch_len=int(patch_len),
        stride=int(stride),
        alpha=float(model_cfg["alpha"]),
        top_p=float(model_cfg["top_p"]),
        pos=True,
        label_len=0,
        num_class=1,
    )


class FormalPatchModel(nn.Module):
    """把 patch-feature 序列压缩成现有骨干模型可接受的一维 patch 序列。"""

    def __init__(
        self,
        *,
        model_name: str,
        feature_dim: int,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        model_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_nodes = int(num_nodes)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

        configs = build_model_configs(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            num_nodes=self.num_nodes,
            model_cfg=model_cfg,
        )
        backbone_module = importlib.import_module(f"models.{model_name}")
        self.backbone = backbone_module.Model(configs)

        self.feature_proj = nn.Linear(int(feature_dim), 1)
        with torch.no_grad():
            self.feature_proj.weight.zero_()
            self.feature_proj.bias.zero_()
            self.feature_proj.weight[0, 0] = 1.0

    def forward(self, x: torch.Tensor, enc_mask: torch.Tensor | None = None) -> torch.Tensor:
        if enc_mask is not None:
            x = x * enc_mask.unsqueeze(-1)

        x_scalar = self.feature_proj(x).squeeze(-1)
        if enc_mask is not None:
            x_scalar = x_scalar * enc_mask

        batch_size = x_scalar.size(0)
        x_mark_enc = torch.zeros(batch_size, self.seq_len, 1, device=x_scalar.device, dtype=x_scalar.dtype)
        x_dec = torch.zeros(batch_size, self.pred_len, self.num_nodes, device=x_scalar.device, dtype=x_scalar.dtype)
        x_mark_dec = torch.zeros(batch_size, self.pred_len, 1, device=x_scalar.device, dtype=x_scalar.dtype)
        return self.backbone(x_scalar, x_mark_enc, x_dec, x_mark_dec)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """按目标掩码计算 MSE。"""

    weight = mask.float()
    denom = weight.sum().clamp_min(eps)
    return (((pred - target) ** 2) * weight).sum() / denom


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""

    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))
