"""这个模块负责 slopemine v2 正式 patch 实验的数据加载、窗口构造与子集掩码。
相关文件：patch_tensor_base_v2_ps10.npz、experiment_split_plan_v1.csv、dataset_manifest_v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EXPERIMENT_FEATURE_SPECS = {
    "A1": ["internal"],
    "A2": ["internal", "weather"],
    "A3": ["internal", "weather", "blast_v2"],
    "A4": ["internal", "weather", "blast_v3"],
}


@dataclass(frozen=True)
class FormalWindowBundle:
    """冻结后的正式实验基础张量。"""

    timestamps: np.ndarray
    patch_ids: np.ndarray
    internal_feature_names: list[str]
    weather_feature_names: list[str]
    blast_v2_feature_names: list[str]
    blast_v3_feature_names: list[str]
    internal: np.ndarray
    weather: np.ndarray
    blast_v2: np.ndarray
    blast_v3: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    input_mask: np.ndarray
    global_missing: np.ndarray


def _as_string_list(values: np.ndarray) -> list[str]:
    return [str(item) for item in values.tolist()]


def broadcast_global(matrix: np.ndarray, patch_count: int) -> np.ndarray:
    """把全局特征广播到 patch 维。"""

    return np.broadcast_to(matrix[:, None, :], (matrix.shape[0], patch_count, matrix.shape[1])).astype(np.float32)


def load_formal_window_bundle(
    dataset_dir: Path,
    *,
    missing_hour_policy: str = "drop_global_missing",
) -> tuple[FormalWindowBundle, pd.DataFrame, dict[str, Any]]:
    """读取正式实验基础张量，并按策略处理 8 个全局缺失小时。"""

    raw_bundle = np.load(dataset_dir / "patch_tensor_base_v2_ps10.npz", allow_pickle=True)
    patch_meta = pd.read_csv(dataset_dir / "patch_meta_v2_ps10_bg_excluded.csv")

    patch_ids = raw_bundle["patch_ids"].astype(str)
    patch_meta = patch_meta.set_index("patch_id").reindex(patch_ids).reset_index()
    if patch_meta["patch_id"].isna().any():
        raise ValueError("patch_meta_v2_ps10_bg_excluded.csv 与 patch_tensor_base_v2_ps10.npz 的 patch 顺序无法对齐。")

    timestamps_all = pd.to_datetime(raw_bundle["timestamps"].astype(str)).to_numpy()
    global_missing = raw_bundle["global_missing"].astype(np.uint8)

    if missing_hour_policy != "drop_global_missing":
        raise ValueError(f"暂不支持的 missing_hour_policy: {missing_hour_policy}")

    keep_mask = global_missing == 0
    dropped_timestamps = pd.to_datetime(timestamps_all[~keep_mask])
    bundle = FormalWindowBundle(
        timestamps=timestamps_all[keep_mask],
        patch_ids=patch_ids,
        internal_feature_names=_as_string_list(raw_bundle["internal_feature_names"]),
        weather_feature_names=_as_string_list(raw_bundle["weather_feature_names"]),
        blast_v2_feature_names=_as_string_list(raw_bundle["blast_v2_feature_names"]),
        blast_v3_feature_names=_as_string_list(raw_bundle["blast_v3_feature_names"]),
        internal=raw_bundle["internal"][keep_mask].astype(np.float32),
        weather=raw_bundle["weather"][keep_mask].astype(np.float32),
        blast_v2=raw_bundle["blast_v2"][keep_mask].astype(np.float32),
        blast_v3=raw_bundle["blast_v3"][keep_mask].astype(np.float32),
        target=raw_bundle["target"][keep_mask].astype(np.float32),
        target_mask=raw_bundle["target_mask"][keep_mask].astype(np.uint8),
        input_mask=raw_bundle["input_mask"][keep_mask].astype(np.uint8),
        global_missing=raw_bundle["global_missing"][keep_mask].astype(np.uint8),
    )
    report = {
        "missing_hour_policy": missing_hour_policy,
        "natural_total_steps": int(len(timestamps_all)),
        "effective_total_steps": int(bundle.target.shape[0]),
        "dropped_global_missing_hours": int((~keep_mask).sum()),
        "dropped_global_missing_timestamps": [timestamp.isoformat() for timestamp in dropped_timestamps.to_pydatetime()],
    }
    return bundle, patch_meta, report


def load_split_plan(split_path: Path) -> pd.DataFrame:
    """读取正式 split 方案。"""

    split_plan = pd.read_csv(split_path, parse_dates=["start_timestamp", "end_timestamp"])
    required = {"train", "val", "test"}
    if set(split_plan["split_name"]) != required:
        raise ValueError("experiment_split_plan_v1.csv 缺少 train/val/test 三段正式切分。")
    split_plan = split_plan.sort_values("effective_start_index").reset_index(drop=True)
    return split_plan


def build_feature_tensor_by_experiment(
    bundle: FormalWindowBundle,
    experiment_key: str,
) -> tuple[np.ndarray, list[str]]:
    """按 A1-A4 定义拼接正式输入特征。"""

    patch_count = bundle.internal.shape[1]
    internal = bundle.internal.astype(np.float32)
    weather = broadcast_global(bundle.weather.astype(np.float32), patch_count)
    blast_v2 = broadcast_global(bundle.blast_v2.astype(np.float32), patch_count)
    blast_v3 = bundle.blast_v3.astype(np.float32)

    internal_names = list(bundle.internal_feature_names)
    weather_names = list(bundle.weather_feature_names)
    blast_v2_names = list(bundle.blast_v2_feature_names)
    blast_v3_names = list(bundle.blast_v3_feature_names)

    if experiment_key == "A1":
        return internal, internal_names
    if experiment_key == "A2":
        return np.concatenate([internal, weather], axis=-1), internal_names + weather_names
    if experiment_key == "A3":
        return np.concatenate([internal, weather, blast_v2], axis=-1), internal_names + weather_names + blast_v2_names
    if experiment_key == "A4":
        return np.concatenate([internal, weather, blast_v3], axis=-1), internal_names + weather_names + blast_v3_names
    raise ValueError(f"未知实验：{experiment_key}")


def resolve_formal_window_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """优先读取冻结后的正式最佳窗口配置。"""

    formal_cfg = config["formal_round1"]
    tuned_path = repo_root / "outputs" / "slopemine_v2" / "tuning_v1" / "best_window_config_v1.json"
    source = "config_default"
    seq_len = int(formal_cfg["default_seq_len"])
    pred_len = int(formal_cfg["default_pred_len"])

    if bool(formal_cfg.get("use_tuned_window", False)) and tuned_path.exists():
        tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
        best_overall = tuned["best_overall"]
        seq_len = int(best_overall["seq_len"])
        pred_len = int(best_overall["pred_len"])
        source = str(tuned_path.relative_to(repo_root))

    return {
        "seq_len": seq_len,
        "pred_len": pred_len,
        "source": source,
    }


def build_formal_window_manifest(
    timestamps: np.ndarray,
    split_plan: pd.DataFrame,
    *,
    seq_len: int,
    pred_len: int,
    missing_hour_policy: str,
) -> pd.DataFrame:
    """基于正式 split 构造样本级窗口清单。"""

    total_steps = len(timestamps)
    num_samples = total_steps - seq_len - pred_len + 1
    if num_samples <= 0:
        raise ValueError(f"无可用窗口：seq_len={seq_len}, pred_len={pred_len}, total_steps={total_steps}")

    sample_start = np.arange(num_samples, dtype=np.int32)
    encoder_start = sample_start
    encoder_end_exclusive = sample_start + seq_len
    decoder_start = encoder_end_exclusive
    decoder_end_exclusive = decoder_start + pred_len

    rows: list[dict[str, Any]] = []
    split_lookup = {str(row.split_name): row for row in split_plan.itertuples(index=False)}
    for split_name, split_row in split_lookup.items():
        split_start = int(split_row.effective_start_index)
        split_end = int(split_row.effective_end_index_exclusive)
        mask = (decoder_start >= split_start) & (decoder_end_exclusive <= split_end)
        selected = np.where(mask)[0]
        for sample_id in selected.tolist():
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "split_name": split_name,
                    "selected_candidate_id": str(split_row.selected_candidate_id),
                    "seq_len": int(seq_len),
                    "pred_len": int(pred_len),
                    "encoder_start_index": int(encoder_start[sample_id]),
                    "encoder_end_index_exclusive": int(encoder_end_exclusive[sample_id]),
                    "decoder_start_index": int(decoder_start[sample_id]),
                    "decoder_end_index_exclusive": int(decoder_end_exclusive[sample_id]),
                    "encoder_start_timestamp": pd.Timestamp(timestamps[encoder_start[sample_id]]),
                    "encoder_end_timestamp": pd.Timestamp(timestamps[encoder_end_exclusive[sample_id] - 1]),
                    "decoder_start_timestamp": pd.Timestamp(timestamps[decoder_start[sample_id]]),
                    "decoder_end_timestamp": pd.Timestamp(timestamps[decoder_end_exclusive[sample_id] - 1]),
                    "encoder_crosses_split_start": int(encoder_start[sample_id] < split_start),
                    "uses_effective_timeline": 1,
                    "missing_hour_policy": missing_hour_policy,
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["split_name", "sample_id"]).reset_index(drop=True)
    return manifest


def compute_subset_hour_flags(bundle: FormalWindowBundle) -> dict[str, np.ndarray]:
    """生成 blast、rain 和 non-event 小时标签。"""

    weather_idx = bundle.weather_feature_names.index("rainfall")
    blast_idx = bundle.blast_v2_feature_names.index("blast_count")
    strength_idx = bundle.blast_v2_feature_names.index("total_charge_kg")

    rain_hours = bundle.weather[:, weather_idx] > 0
    blast_hours = bundle.blast_v2[:, blast_idx] > 0
    non_event_hours = ~(rain_hours | blast_hours)
    blast_strength = bundle.blast_v2[:, strength_idx].astype(np.float32)

    return {
        "blast_hours": blast_hours.astype(np.uint8),
        "rain_hours": rain_hours.astype(np.uint8),
        "non_event_hours": non_event_hours.astype(np.uint8),
        "blast_strength": blast_strength,
    }


def build_sample_subset_masks(
    manifest: pd.DataFrame,
    subset_hour_flags: dict[str, np.ndarray],
    pred_len: int,
) -> dict[str, np.ndarray]:
    """把小时标签映射到样本级 decoder horizon 掩码。"""

    decoder_start = manifest["decoder_start_index"].to_numpy(dtype=np.int32)
    decoder_indices = decoder_start[:, None] + np.arange(pred_len, dtype=np.int32)[None, :]
    masks: dict[str, np.ndarray] = {}
    for subset_name in ["blast_hours", "rain_hours", "non_event_hours"]:
        masks[subset_name] = subset_hour_flags[subset_name][decoder_indices].astype(np.uint8)
    return masks


def compute_feature_scaler(
    feature_array: np.ndarray,
    input_mask: np.ndarray,
    train_end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """仅基于 train 时间段估计特征均值和方差。"""

    feature_dim = feature_array.shape[-1]
    feature_mean = np.zeros(feature_dim, dtype=np.float32)
    feature_std = np.ones(feature_dim, dtype=np.float32)
    train_mask = input_mask[:train_end_index].astype(bool)

    for feature_idx in range(feature_dim):
        values = feature_array[:train_end_index, :, feature_idx][train_mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        feature_mean[feature_idx] = float(values.mean())
        std = float(values.std())
        feature_std[feature_idx] = std if std >= 1e-6 else 1.0

    return feature_mean, feature_std


def compute_target_scaler(
    target_array: np.ndarray,
    target_mask: np.ndarray,
    train_end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """仅基于 train 时间段估计每个 patch 的目标均值和方差。"""

    num_nodes = target_array.shape[1]
    target_mean = np.zeros(num_nodes, dtype=np.float32)
    target_std = np.ones(num_nodes, dtype=np.float32)

    for node_idx in range(num_nodes):
        mask = target_mask[:train_end_index, node_idx].astype(bool)
        values = target_array[:train_end_index, node_idx][mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        target_mean[node_idx] = float(values.mean())
        std = float(values.std())
        target_std[node_idx] = std if std >= 1e-6 else 1.0

    return target_mean, target_std


class SlopeMineFormalDataset(Dataset):
    """按正式 split 返回 encoder/decoder 窗口。"""

    def __init__(
        self,
        *,
        manifest: pd.DataFrame,
        feature_array: np.ndarray,
        target_array: np.ndarray,
        input_mask: np.ndarray,
        target_mask: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.feature_array = feature_array.astype(np.float32)
        self.target_array = target_array.astype(np.float32)
        self.input_mask = input_mask.astype(np.float32)
        self.target_mask = target_mask.astype(np.float32)
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[index]
        encoder_slice = slice(int(row.encoder_start_index), int(row.encoder_end_index_exclusive))
        decoder_slice = slice(int(row.decoder_start_index), int(row.decoder_end_index_exclusive))

        x = self.feature_array[encoder_slice]
        enc_mask = self.input_mask[encoder_slice]
        x = (x - self.feature_mean[None, None, :]) / self.feature_std[None, None, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y_raw = self.target_array[decoder_slice]
        y_mask = self.target_mask[decoder_slice]
        y = (y_raw - self.target_mean[None, :]) / self.target_std[None, :]
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return {
            "sample_id": torch.tensor(int(row.sample_id), dtype=torch.long),
            "x": torch.from_numpy(x),
            "enc_mask": torch.from_numpy(enc_mask.astype(np.float32)),
            "y": torch.from_numpy(y),
            "y_raw": torch.from_numpy(np.nan_to_num(y_raw, nan=0.0).astype(np.float32)),
            "y_mask": torch.from_numpy(y_mask.astype(np.float32)),
        }
