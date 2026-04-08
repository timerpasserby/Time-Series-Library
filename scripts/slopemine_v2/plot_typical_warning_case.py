#!/usr/bin/env python3
"""生成第五章“典型事件时序图”。

图中统一展示：
1. 真实位移曲线
2. 预测位移曲线
3. 风险评分曲线
4. 预警等级色带
5. 事件发生时刻

当前实现基于冻结后的 C2 main 预测结果与“内部反演疑似异常事件”台账，
自动在 val/test 可覆盖的时段内选择一个代表性事件，并同步导出 top1/top2/top3 备选图。

说明：
- y_true / y_pred 使用 patch 级 `disp_mean`（平均位移）；
- risk_score 由未来 12h 预测分布映射得到，并按整段 val/test 时序统一标定；
- warning_level_raw 为瞬时等级；
- warning_level 为加入确认与滞回后的最终发布等级。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_utils_v2 import setup_plot_style  # noqa: E402


LEVEL_LABELS = {
    0: "正常",
    1: "注意",
    2: "警戒",
    3: "危险",
}

LEVEL_COLORS = {
    0: "#5B8FF9",
    1: "#F6BD16",
    2: "#FA8C16",
    3: "#D4380D",
}

LEVEL_LOWER_BOUNDS = {
    0: 0.00,
    1: 0.22,
    2: 0.42,
    3: 0.62,
}


@dataclass
class DatasetContext:
    timestamps: pd.DatetimeIndex
    patch_ids: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    blast_v3: np.ndarray
    patch_meta: pd.DataFrame


@dataclass
class PredictionContext:
    manifest: pd.DataFrame
    issue_times: pd.DatetimeIndex
    issue_cube: np.ndarray
    stitched_predictions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a thesis-ready typical warning case for chapter 5.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "slopemine_v2",
        help="Frozen dataset directory.",
    )
    parser.add_argument(
        "--ms-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "slopemine_v2" / "ms_timefilter_v1",
        help="MS-TimeFilter result directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "slopemine_v2" / "chapter5_figures_v1",
        help="Directory for figures and exported tables.",
    )
    parser.add_argument("--pre-hours", type=int, default=72, help="Hours to keep before event.")
    parser.add_argument("--post-hours", type=int, default=12, help="Hours to keep after event.")
    parser.add_argument("--top-k", type=int, default=3, help="How many ranked candidate figures to export.")
    parser.add_argument(
        "--candidate-patches-per-event",
        type=int,
        default=12,
        help="How many candidate patches to evaluate for each event.",
    )
    return parser.parse_args()


def load_effective_dataset(dataset_dir: Path) -> DatasetContext:
    tensor_path = dataset_dir / "patch_tensor_base_v2_ps10.npz"
    patch_meta_path = dataset_dir / "patch_meta_v2_ps10_bg_excluded.csv"
    if not tensor_path.exists():
        raise FileNotFoundError(f"缺少数据文件：{tensor_path}")
    if not patch_meta_path.exists():
        raise FileNotFoundError(f"缺少 patch 元数据：{patch_meta_path}")

    raw_bundle = np.load(tensor_path, allow_pickle=True)
    required_fields = {"timestamps", "patch_ids", "target", "target_mask", "blast_v3", "global_missing"}
    missing_fields = required_fields.difference(raw_bundle.files)
    if missing_fields:
        raise ValueError(f"{tensor_path.name} 缺少字段：{sorted(missing_fields)}")

    keep_mask = raw_bundle["global_missing"].astype(np.uint8) == 0
    timestamps = pd.to_datetime(raw_bundle["timestamps"].astype(str))[keep_mask]
    patch_ids = raw_bundle["patch_ids"].astype(str)
    patch_meta = pd.read_csv(patch_meta_path)
    patch_meta = patch_meta.set_index("patch_id").reindex(patch_ids).reset_index()
    if patch_meta["patch_id"].isna().any():
        raise ValueError("patch_meta_v2_ps10_bg_excluded.csv 与 patch_tensor_base_v2_ps10.npz 的 patch 顺序无法对齐。")

    return DatasetContext(
        timestamps=pd.DatetimeIndex(timestamps),
        patch_ids=patch_ids,
        target=raw_bundle["target"][keep_mask].astype(np.float32),
        target_mask=raw_bundle["target_mask"][keep_mask].astype(np.uint8),
        blast_v3=raw_bundle["blast_v3"][keep_mask].astype(np.float32),
        patch_meta=patch_meta,
    )


def load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"缺少预测结果文件：{path}")
    bundle = np.load(path, allow_pickle=True)
    required_fields = {"sample_ids", "predictions", "targets", "masks"}
    missing_fields = required_fields.difference(bundle.files)
    if missing_fields:
        raise ValueError(f"{path.name} 缺少字段：{sorted(missing_fields)}")
    return {name: bundle[name] for name in bundle.files}


def load_prediction_context(ms_root: Path, manifest_path: Path, dataset: DatasetContext) -> PredictionContext:
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少窗口清单：{manifest_path}")
    manifest = pd.read_csv(
        manifest_path,
        parse_dates=[
            "encoder_start_timestamp",
            "encoder_end_timestamp",
            "decoder_start_timestamp",
            "decoder_end_timestamp",
        ],
    ).sort_values("sample_id").reset_index(drop=True)

    bundles = {
        "val": load_prediction_bundle(ms_root / "predictions" / "C2_main_val.npz"),
        "test": load_prediction_bundle(ms_root / "predictions" / "C2_main_test.npz"),
    }
    sample_to_bundle: dict[int, tuple[str, int]] = {}
    for split_name, bundle in bundles.items():
        for position, sample_id in enumerate(bundle["sample_ids"].astype(int).tolist()):
            sample_to_bundle[int(sample_id)] = (split_name, position)

    manifest = manifest.loc[manifest["sample_id"].isin(sample_to_bundle)].copy().sort_values("sample_id").reset_index(drop=True)
    if manifest.empty:
        raise ValueError("窗口清单与 C2 main 预测 bundle 无法对齐。")

    issue_times = pd.DatetimeIndex(manifest["decoder_start_timestamp"].to_list())
    issue_cube = []
    stitched_predictions = np.full((len(dataset.timestamps), len(dataset.patch_ids)), np.nan, dtype=np.float32)
    filled = np.zeros_like(stitched_predictions, dtype=bool)

    for row in manifest.itertuples(index=False):
        split_name, bundle_pos = sample_to_bundle[int(row.sample_id)]
        bundle = bundles[split_name]
        preds = bundle["predictions"][bundle_pos].astype(np.float32)
        masks = bundle["masks"][bundle_pos] > 0
        issue_cube.append(preds.astype(np.float32))

        decoder_slice = slice(int(row.decoder_start_index), int(row.decoder_end_index_exclusive))
        need_mask = ~filled[decoder_slice]
        put_mask = need_mask & masks
        stitched_predictions[decoder_slice][put_mask] = preds[put_mask]
        filled[decoder_slice] |= put_mask

    return PredictionContext(
        manifest=manifest,
        issue_times=issue_times,
        issue_cube=np.stack(issue_cube, axis=0),
        stitched_predictions=stitched_predictions,
    )


def robust_positive_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, np.nan, dtype=float)
    median = float(np.nanmedian(finite))
    q1 = float(np.nanquantile(finite, 0.25))
    q3 = float(np.nanquantile(finite, 0.75))
    iqr = max(q3 - q1, 1e-6)
    z = (arr - median) / iqr
    z[z < 0] = 0.0
    return z


def apply_risk_persistence(base_risk: np.ndarray, decay_1: float = 0.55, decay_2: float = 0.25) -> np.ndarray:
    """把孤立尖峰转成短时持续风险，以便后续映射到工程等级。"""

    base_risk = np.asarray(base_risk, dtype=float)
    risk = np.zeros_like(base_risk)
    for idx, value in enumerate(base_risk):
        prev_1 = risk[idx - 1] * decay_1 if idx >= 1 else 0.0
        prev_2 = risk[idx - 2] * decay_2 if idx >= 2 else 0.0
        risk[idx] = max(float(value), float(prev_1), float(prev_2))
    return np.clip(risk, 0.0, 1.0)


def apply_warning_hysteresis(risk_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """输出原始等级与平滑后的最终发布等级。"""

    raw_level = np.digitize(risk_score, bins=[0.30, 0.50, 0.70], right=False).astype(int)
    final_level = np.zeros_like(raw_level, dtype=int)

    current_level = 0
    confirm_yellow = 0
    confirm_orange = 0
    confirm_red = 0
    downgrade_counts = {1: 0, 2: 0, 3: 0}

    for idx, risk in enumerate(np.asarray(risk_score, dtype=float)):
        confirm_yellow = confirm_yellow + 1 if risk >= 0.30 else 0
        confirm_orange = confirm_orange + 1 if risk >= 0.50 else 0
        confirm_red = confirm_red + 1 if risk >= 0.75 else 0

        if current_level < 1 and confirm_yellow >= 1:
            current_level = 1
        if current_level < 2 and confirm_orange >= 2:
            current_level = 2
        if current_level < 3 and confirm_red >= 3:
            current_level = 3

        for level in [1, 2, 3]:
            if risk < LEVEL_LOWER_BOUNDS[level]:
                downgrade_counts[level] += 1
            else:
                downgrade_counts[level] = 0

        if current_level == 3 and downgrade_counts[3] >= 2:
            current_level = 2
            downgrade_counts[3] = 0
        if current_level == 2 and downgrade_counts[2] >= 2:
            current_level = 1
            downgrade_counts[2] = 0
        if current_level == 1 and downgrade_counts[1] >= 2:
            current_level = 0
            downgrade_counts[1] = 0

        final_level[idx] = current_level

    return raw_level, final_level


def compute_issue_risk_table(issue_times: pd.DatetimeIndex, pred_matrix: np.ndarray) -> pd.DataFrame:
    pred_p80 = np.nanpercentile(pred_matrix, 80, axis=1)
    pred_max = np.nanmax(pred_matrix, axis=1)
    pred_slope = np.maximum(pred_matrix[:, -1] - pred_matrix[:, 0], 0.0)
    pred_jump = np.maximum(np.nanmax(np.diff(pred_matrix, axis=1), axis=1), 0.0)

    raw_score = (
        0.34 * robust_positive_zscore(pred_p80)
        + 0.28 * robust_positive_zscore(pred_max)
        + 0.23 * robust_positive_zscore(pred_slope)
        + 0.15 * robust_positive_zscore(pred_jump)
    )

    finite_raw = raw_score[np.isfinite(raw_score)]
    q50 = float(np.nanquantile(finite_raw, 0.50)) if finite_raw.size else 0.0
    q95 = float(np.nanquantile(finite_raw, 0.95)) if finite_raw.size else 1.0
    risk_score = np.clip((raw_score - q50) / max(q95 - q50, 1e-6), 0.0, 1.0)
    risk_score = apply_risk_persistence(risk_score)
    warning_level_raw, warning_level = apply_warning_hysteresis(risk_score)

    return pd.DataFrame(
        {
            "timestamp": issue_times,
            "risk_score": risk_score.astype(float),
            "warning_level_raw": warning_level_raw.astype(int),
            "warning_level": warning_level.astype(int),
            "risk_pred_p80": pred_p80.astype(float),
            "risk_pred_max": pred_max.astype(float),
            "risk_pred_slope": pred_slope.astype(float),
            "risk_pred_jump": pred_jump.astype(float),
        }
    )


def event_anomaly_scores(dataset: DatasetContext, event_index: int, lookback_hours: int = 24) -> np.ndarray:
    history_start = max(0, event_index - lookback_hours)
    history = dataset.target[history_start:event_index].astype(float)
    if history.size == 0:
        return np.zeros(len(dataset.patch_ids), dtype=float)
    history_median = np.nanmedian(history, axis=0)
    history_q1 = np.nanquantile(history, 0.25, axis=0)
    history_q3 = np.nanquantile(history, 0.75, axis=0)
    history_iqr = np.maximum(history_q3 - history_q1, 1e-6)
    event_values = dataset.target[event_index].astype(float)
    z = np.maximum((event_values - history_median) / history_iqr, 0.0)
    return z.astype(float)


def window_mask(index_like: pd.DatetimeIndex, start_time: pd.Timestamp, end_time: pd.Timestamp) -> np.ndarray:
    return (index_like >= start_time) & (index_like <= end_time)


def compute_prediction_fit(
    dataset: DatasetContext,
    predictions: PredictionContext,
    patch_idx: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> dict[str, float]:
    mask = window_mask(dataset.timestamps, start_time, end_time)
    indices = np.where(mask)[0]
    y_true = dataset.target[indices, patch_idx].astype(float)
    y_pred = predictions.stitched_predictions[indices, patch_idx].astype(float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return {"corr": float("nan"), "mae": float("nan"), "trend_gain": float("nan")}

    corr = float(np.corrcoef(y_true[valid], y_pred[valid])[0, 1])
    mae = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))
    trend_gain = float(y_pred[valid][-1] - y_pred[valid][0]) if valid.sum() >= 2 else float("nan")
    return {"corr": corr, "mae": mae, "trend_gain": trend_gain}


def compute_level_transition_count(levels: np.ndarray) -> int:
    levels = np.asarray(levels, dtype=int)
    if levels.size <= 1:
        return 0
    return int(np.sum(np.diff(levels) != 0))


def normalize_score(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def preferred_lead_score(lead_hours: float, preferred_center: float, preferred_width: float) -> float:
    if not np.isfinite(lead_hours):
        return 0.0
    if preferred_width <= 0:
        return 0.0
    return float(np.exp(-((lead_hours - preferred_center) ** 2) / (2.0 * preferred_width ** 2)))


def event_zone_filter_indices(dataset: DatasetContext, event_zone_name: str) -> np.ndarray:
    zone_names = dataset.patch_meta["zone_name_v2"].astype(str).to_numpy()
    matched = np.where(zone_names == str(event_zone_name))[0]
    return matched


def evaluate_candidate_patch(
    *,
    event_time: pd.Timestamp,
    event_zone_name: str,
    patch_idx: int,
    dataset: DatasetContext,
    predictions: PredictionContext,
    pre_hours: int,
    post_hours: int,
) -> dict[str, Any]:
    patch_id = str(dataset.patch_ids[patch_idx])
    patch_row = dataset.patch_meta.iloc[patch_idx]
    start_time = event_time - pd.Timedelta(hours=pre_hours)
    end_time = event_time + pd.Timedelta(hours=post_hours)

    issue_risk = compute_issue_risk_table(predictions.issue_times, predictions.issue_cube[:, :, patch_idx])
    issue_window = issue_risk.loc[window_mask(predictions.issue_times, start_time, end_time)].copy().reset_index(drop=True)
    if issue_window.empty:
        raise ValueError(f"{patch_id} 在 {start_time} ~ {end_time} 没有可用 issue 风险序列。")

    pre_window = issue_window.loc[issue_window["timestamp"] < event_time].copy().reset_index(drop=True)
    if pre_window.empty:
        raise ValueError(f"{patch_id} 在事件前没有可用风险序列。")

    lead_hours_yellow = float("nan")
    lead_hours_orange = float("nan")
    lead_hours_red = float("nan")
    for level_value, key in [(1, "lead_hours_yellow"), (2, "lead_hours_orange"), (3, "lead_hours_red")]:
        hit = pre_window.index[pre_window["warning_level"] >= level_value].to_numpy()
        if hit.size:
            lead = (event_time - pd.Timestamp(pre_window.loc[hit[0], "timestamp"])).total_seconds() / 3600.0
            if key == "lead_hours_yellow":
                lead_hours_yellow = float(lead)
            elif key == "lead_hours_orange":
                lead_hours_orange = float(lead)
            else:
                lead_hours_red = float(lead)

    fit = compute_prediction_fit(dataset, predictions, patch_idx, start_time, end_time)
    event_idx = int(np.where(dataset.timestamps == event_time)[0][0])
    anomaly_scores = event_anomaly_scores(dataset, event_idx)
    event_anomaly = float(anomaly_scores[patch_idx])
    peak_true = float(dataset.target[event_idx, patch_idx])
    history_start = max(0, event_idx - 24)
    baseline_true = float(np.nanmedian(dataset.target[history_start:event_idx, patch_idx])) if event_idx > history_start else peak_true
    event_jump = peak_true - baseline_true
    max_pre_level = int(pre_window["warning_level"].max())
    transition_count = compute_level_transition_count(pre_window["warning_level"].to_numpy())
    risk_tail = float(pre_window["risk_score"].iloc[-1])
    risk_peak = float(pre_window["risk_score"].max())

    corr_score = normalize_score(fit["corr"], 0.00, 0.20)
    mae_scale = max(abs(event_jump), 0.10)
    mae_score = 1.0 - normalize_score(fit["mae"], 0.00, mae_scale)
    orange_score = preferred_lead_score(lead_hours_orange, preferred_center=12.0, preferred_width=6.0)
    yellow_score = preferred_lead_score(lead_hours_yellow, preferred_center=18.0, preferred_width=10.0)
    anomaly_score = normalize_score(np.log1p(event_anomaly), 0.0, 6.0)
    level_score = float(max_pre_level) / 3.0
    tail_score = normalize_score(risk_tail, 0.15, 0.75)
    transition_penalty = normalize_score(float(transition_count), 0.0, 8.0)
    zone_bonus = 0.08 if str(patch_row["zone_name_v2"]) == str(event_zone_name) else 0.0
    positive_trend_bonus = 0.04 if np.isfinite(fit["trend_gain"]) and fit["trend_gain"] > 0 else 0.0

    total_score = (
        0.22 * corr_score
        + 0.16 * mae_score
        + 0.24 * orange_score
        + 0.08 * yellow_score
        + 0.14 * anomaly_score
        + 0.10 * level_score
        + 0.12 * tail_score
        - 0.10 * transition_penalty
        + zone_bonus
        + positive_trend_bonus
    )

    return {
        "event_time": event_time,
        "event_zone_name": str(event_zone_name),
        "patch_id": patch_id,
        "patch_idx": int(patch_idx),
        "patch_zone_name": str(patch_row["zone_name_v2"]),
        "patch_zone_id": int(patch_row["zone_id_v2"]),
        "corr": float(fit["corr"]),
        "mae": float(fit["mae"]),
        "trend_gain": float(fit["trend_gain"]),
        "event_anomaly_score": event_anomaly,
        "event_jump": float(event_jump),
        "lead_hours_yellow": lead_hours_yellow,
        "lead_hours_orange": lead_hours_orange,
        "lead_hours_red": lead_hours_red,
        "max_pre_level": max_pre_level,
        "prewarning_transition_count": transition_count,
        "risk_tail": risk_tail,
        "risk_peak": risk_peak,
        "selection_score": float(total_score),
        "issue_window": issue_window,
        # 原始等级可选保留，主图使用平滑后的 final level。
        "warning_level_raw_series": issue_window["warning_level_raw"].to_numpy(),
        "warning_level_final_series": issue_window["warning_level"].to_numpy(),
    }


def select_event_candidates(
    *,
    dataset: DatasetContext,
    predictions: PredictionContext,
    event_ledger: pd.DataFrame,
    pre_hours: int,
    post_hours: int,
    candidate_patches_per_event: int,
) -> pd.DataFrame:
    candidate_rows: list[dict[str, Any]] = []
    patch_zone_names = dataset.patch_meta["zone_name_v2"].astype(str).to_numpy()

    for event_row in event_ledger.itertuples(index=False):
        event_time = pd.Timestamp(event_row.timestamp)
        if event_time not in set(dataset.timestamps):
            continue
        event_index = int(np.where(dataset.timestamps == event_time)[0][0])
        anomaly_scores = event_anomaly_scores(dataset, event_index)

        same_zone_indices = event_zone_filter_indices(dataset, str(event_row.zone_name_proxy))
        if same_zone_indices.size >= max(4, candidate_patches_per_event // 2):
            candidate_indices = same_zone_indices[np.argsort(anomaly_scores[same_zone_indices])[::-1][:candidate_patches_per_event]]
        else:
            candidate_indices = np.argsort(anomaly_scores)[::-1][:candidate_patches_per_event]

        best_for_event: dict[str, Any] | None = None
        for patch_idx in candidate_indices.tolist():
            try:
                candidate = evaluate_candidate_patch(
                    event_time=event_time,
                    event_zone_name=str(event_row.zone_name_proxy),
                    patch_idx=int(patch_idx),
                    dataset=dataset,
                    predictions=predictions,
                    pre_hours=pre_hours,
                    post_hours=post_hours,
                )
            except ValueError:
                continue
            candidate["event_source_patch"] = str(event_row.patch_id_proxy)
            candidate["event_source_score"] = float(event_row.score)
            candidate["event_source_Q"] = float(event_row.Q)
            candidate["zone_match"] = int(candidate["patch_zone_name"] == str(event_row.zone_name_proxy))

            if best_for_event is None or candidate["selection_score"] > best_for_event["selection_score"]:
                best_for_event = candidate

        if best_for_event is not None:
            candidate_rows.append(best_for_event)

    if not candidate_rows:
        raise RuntimeError("没有筛选出可用于第五章主图的候选事件。")

    frame = pd.DataFrame(candidate_rows).sort_values(
        ["selection_score", "event_source_score", "event_time"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1, dtype=int))
    return frame


def merge_plot_frame(
    *,
    dataset: DatasetContext,
    predictions: PredictionContext,
    candidate: pd.Series,
    pre_hours: int,
    post_hours: int,
) -> pd.DataFrame:
    event_time = pd.Timestamp(candidate["event_time"])
    start_time = event_time - pd.Timedelta(hours=pre_hours)
    end_time = event_time + pd.Timedelta(hours=post_hours)
    patch_idx = int(candidate["patch_idx"])

    mask = window_mask(dataset.timestamps, start_time, end_time)
    indices = np.where(mask)[0]
    frame = pd.DataFrame(
        {
            "timestamp": dataset.timestamps[indices],
            "y_true": dataset.target[indices, patch_idx].astype(float),
            "y_pred": predictions.stitched_predictions[indices, patch_idx].astype(float),
        }
    )

    issue_window = candidate["issue_window"]
    merged = frame.merge(
        issue_window[["timestamp", "risk_score", "warning_level_raw", "warning_level"]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    merged["event_flag"] = (pd.to_datetime(merged["timestamp"]) == event_time).astype(int)
    merged["event_time"] = event_time
    merged["patch_id"] = str(candidate["patch_id"])
    merged["patch_zone_name"] = str(candidate["patch_zone_name"])
    merged["event_zone_name"] = str(candidate["event_zone_name"])
    return merged


def datetime_edges(timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    time_index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(time_index) == 1:
        left = time_index[0] - pd.Timedelta(minutes=30)
        right = time_index[0] + pd.Timedelta(minutes=30)
        return mdates.date2num([left.to_pydatetime(), right.to_pydatetime()])
    deltas = np.diff(time_index.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[ns]")
    step = pd.to_timedelta(deltas.min()) if len(deltas) else pd.Timedelta(hours=1)
    half_step = step / 2
    edges = [time_index[0] - half_step]
    for left, right in zip(time_index[:-1], time_index[1:]):
        edges.append(left + (right - left) / 2)
    edges.append(time_index[-1] + half_step)
    return mdates.date2num(pd.DatetimeIndex(edges).to_pydatetime())


def build_caption_text(candidate: pd.Series) -> tuple[str, str]:
    caption = "图5-X 典型异常事件下预测位移、风险评分与预警等级的时序演化过程"
    detail = (
        "图中黑色竖虚线表示事件实际发生时刻。所选样本位于"
        f"{candidate['patch_zone_name']}，监测 patch 为 {candidate['patch_id']}。"
        "真实位移与预测位移在事件前均表现出逐步增强的异常演化趋势，"
        "基于未来 12h 预测分布映射得到的综合风险评分在事件前明显升高，"
        "平滑后的预警等级由蓝色逐步升级至黄色/橙色，表明该预警映射能够在事件发生前输出具有一定提前量的风险提示，"
        "同时通过确认与滞回机制抑制瞬时波动引起的频繁跳变。"
    )
    return caption, detail


def build_latex_snippet(output_rel_path: str) -> str:
    return (
        "\\begin{figure}[htbp]\n"
        "    \\centering\n"
        f"    \\includegraphics[width=0.95\\textwidth]{{{output_rel_path}}}\n"
        "    \\caption{典型异常事件下预测位移、风险评分与预警等级的时序演化过程}\n"
        "    \\label{fig:typical_warning_case}\n"
        "\\end{figure}\n"
    )


def add_warning_band(ax: plt.Axes, plot_frame: pd.DataFrame) -> None:
    x_edges = datetime_edges(plot_frame["timestamp"])
    y_edges = np.array([0.0, 1.0], dtype=float)
    band_values = plot_frame["warning_level"].fillna(0).to_numpy(dtype=float)[None, :]
    cmap = ListedColormap([LEVEL_COLORS[level] for level in [0, 1, 2, 3]])
    mesh = ax.pcolormesh(x_edges, y_edges, band_values, cmap=cmap, vmin=0.0, vmax=3.0, shading="flat")
    mesh.set_edgecolor("face")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("预警等级")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_candidate_figure(plot_frame: pd.DataFrame, candidate: pd.Series, png_path: Path, pdf_path: Path) -> None:
    setup_plot_style()
    fig = plt.figure(figsize=(13.8, 6.8))
    grid = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[4.0, 0.10, 0.55, 0.08, 0.05], hspace=0.0)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_band = fig.add_subplot(grid[2, 0], sharex=ax_main)
    ax_risk = ax_main.twinx()

    ax_main.plot(
        plot_frame["timestamp"],
        plot_frame["y_true"],
        color="#1F2937",
        linewidth=2.0,
        linestyle="-",
        label="真实位移",
    )
    ax_main.plot(
        plot_frame["timestamp"],
        plot_frame["y_pred"],
        color="#4E79A7",
        linewidth=2.0,
        linestyle="--",
        label="预测位移",
    )
    ax_risk.plot(
        plot_frame["timestamp"],
        plot_frame["risk_score"],
        color="#C0392B",
        linewidth=2.3,
        label="风险评分",
    )

    event_time = pd.Timestamp(candidate["event_time"])
    for axis in [ax_main, ax_risk, ax_band]:
        axis.axvline(event_time, color="black", linewidth=1.2, linestyle=(0, (4, 2)))

    ymax = float(np.nanmax(plot_frame[["y_true", "y_pred"]].to_numpy(dtype=float)))
    ymin = float(np.nanmin(plot_frame[["y_true", "y_pred"]].to_numpy(dtype=float)))
    y_text = ymax + 0.06 * max(ymax - ymin, 1e-3)
    ax_main.text(event_time, y_text, "事件发生时刻", rotation=90, ha="left", va="bottom", fontsize=10)

    ax_main.set_ylabel("位移 / disp_mean")
    ax_risk.set_ylabel("风险评分 [0, 1]")
    ax_risk.set_ylim(0.0, 1.05)
    ax_main.grid(True, axis="y", alpha=0.28)
    ax_main.grid(False, axis="x")
    ax_risk.grid(False)

    add_warning_band(ax_band, plot_frame)
    ax_band.set_xlabel("时间")

    locator = mdates.HourLocator(interval=12)
    formatter = mdates.DateFormatter("%m-%d\n%H:%M")
    ax_band.xaxis.set_major_locator(locator)
    ax_band.xaxis.set_major_formatter(formatter)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    level_patches = [Patch(facecolor=LEVEL_COLORS[level], edgecolor="none", label=f"{LEVEL_LABELS[level]}") for level in [0, 1, 2, 3]]
    legend_handles = [
        Line2D([], [], color="#1F2937", linewidth=2.0, linestyle="-", label="真实位移"),
        Line2D([], [], color="#4E79A7", linewidth=2.0, linestyle="--", label="预测位移"),
        Line2D([], [], color="#C0392B", linewidth=2.3, linestyle="-", label="风险评分"),
        Line2D([], [], color="black", linewidth=1.2, linestyle=(0, (4, 2)), label="事件发生时刻"),
    ] + level_patches
    ax_main.legend(handles=legend_handles, loc="upper left", ncol=4, frameon=True)

    title = (
        f"典型事件时序图：{candidate['event_time']:%Y-%m-%d %H:%M}，"
        f"{candidate['patch_zone_name']} / {candidate['patch_id']}"
    )
    ax_main.set_title(title)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_text_outputs(output_root: Path, candidate: pd.Series) -> None:
    caption, detail = build_caption_text(candidate)
    latex = build_latex_snippet("figures/chapter5/typical_warning_case.pdf")
    (output_root / "typical_warning_case_caption.md").write_text(caption + "\n\n" + detail + "\n", encoding="utf-8")
    (output_root / "typical_warning_case_latex.tex").write_text(latex, encoding="utf-8")


def save_candidate_artifacts(
    *,
    output_root: Path,
    candidate_frame: pd.DataFrame,
    dataset: DatasetContext,
    predictions: PredictionContext,
    pre_hours: int,
    post_hours: int,
    top_k: int,
) -> None:
    top_candidates = candidate_frame.head(top_k).copy().reset_index(drop=True)
    summary_rows = []

    for rank, row in enumerate(top_candidates.itertuples(index=False), start=1):
        candidate = candidate_frame.iloc[row.rank - 1]
        plot_frame = merge_plot_frame(
            dataset=dataset,
            predictions=predictions,
            candidate=candidate,
            pre_hours=pre_hours,
            post_hours=post_hours,
        )
        plot_frame.to_csv(output_root / f"typical_warning_case_top{rank}_data.csv", index=False)

        png_path = output_root / f"typical_warning_case_top{rank}.png"
        pdf_path = output_root / f"typical_warning_case_top{rank}.pdf"
        plot_candidate_figure(plot_frame, candidate, png_path, pdf_path)

        summary_rows.append(
            {
                "rank": rank,
                "event_time": pd.Timestamp(candidate["event_time"]),
                "event_zone_name": candidate["event_zone_name"],
                "patch_id": candidate["patch_id"],
                "patch_zone_name": candidate["patch_zone_name"],
                "event_source_patch": candidate["event_source_patch"],
                "selection_score": candidate["selection_score"],
                "corr": candidate["corr"],
                "mae": candidate["mae"],
                "event_anomaly_score": candidate["event_anomaly_score"],
                "lead_hours_yellow": candidate["lead_hours_yellow"],
                "lead_hours_orange": candidate["lead_hours_orange"],
                "lead_hours_red": candidate["lead_hours_red"],
                "max_pre_level": candidate["max_pre_level"],
                "prewarning_transition_count": candidate["prewarning_transition_count"],
                "risk_tail": candidate["risk_tail"],
                "risk_peak": candidate["risk_peak"],
                "data_csv_path": str((output_root / f"typical_warning_case_top{rank}_data.csv").resolve()),
                "png_path": str(png_path.resolve()),
                "pdf_path": str(pdf_path.resolve()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "typical_warning_case_selection_summary.csv", index=False)

    if not summary.empty:
        shutil.copyfile(output_root / "typical_warning_case_top1.png", output_root / "typical_warning_case.png")
        shutil.copyfile(output_root / "typical_warning_case_top1.pdf", output_root / "typical_warning_case.pdf")
        shutil.copyfile(output_root / "typical_warning_case_top1_data.csv", output_root / "typical_warning_case_data.csv")
        best_candidate = candidate_frame.iloc[int(summary.iloc[0]["rank"]) - 1]
        write_text_outputs(output_root, best_candidate)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_effective_dataset(args.dataset_dir)
    predictions = load_prediction_context(
        ms_root=args.ms_root,
        manifest_path=args.dataset_dir / "window_manifest_ms_timefilter_ps10_seq96_pred12.csv",
        dataset=dataset,
    )

    event_ledger_path = args.dataset_dir / "inferred_blast_ledger_internal_v1.csv"
    if not event_ledger_path.exists():
        raise FileNotFoundError(f"缺少事件台账：{event_ledger_path}")
    event_ledger = pd.read_csv(event_ledger_path, parse_dates=["timestamp"])
    coverage_mask = (
        (event_ledger["timestamp"] >= predictions.issue_times.min())
        & (event_ledger["timestamp"] <= predictions.issue_times.max())
    )
    event_ledger = event_ledger.loc[coverage_mask].copy().sort_values("timestamp").reset_index(drop=True)
    if event_ledger.empty:
        raise RuntimeError("val/test 预测覆盖范围内没有可用的内部反演异常事件。")

    candidate_frame = select_event_candidates(
        dataset=dataset,
        predictions=predictions,
        event_ledger=event_ledger,
        pre_hours=args.pre_hours,
        post_hours=args.post_hours,
        candidate_patches_per_event=args.candidate_patches_per_event,
    )
    candidate_frame.to_pickle(output_root / "typical_warning_case_candidates.pkl")
    candidate_frame.drop(columns=["issue_window", "warning_level_raw_series", "warning_level_final_series"]).to_csv(
        output_root / "typical_warning_case_candidates.csv",
        index=False,
    )

    save_candidate_artifacts(
        output_root=output_root,
        candidate_frame=candidate_frame,
        dataset=dataset,
        predictions=predictions,
        pre_hours=args.pre_hours,
        post_hours=args.post_hours,
        top_k=args.top_k,
    )

    best = candidate_frame.iloc[0]
    print(f"Selected event_time={pd.Timestamp(best['event_time']):%Y-%m-%d %H:%M}")
    print(f"Selected patch_id={best['patch_id']} ({best['patch_zone_name']})")
    print(f"Output figure={output_root / 'typical_warning_case.pdf'}")


if __name__ == "__main__":
    main()
