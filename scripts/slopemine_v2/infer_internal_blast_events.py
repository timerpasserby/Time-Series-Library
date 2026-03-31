#!/usr/bin/env python3
"""这个脚本负责基于边坡内部时序异常反演疑似爆破事件台账。
相关文件：config_v2.toml、point_meta_v2.csv、patch_meta_v2_ps10.csv、patch_series_v2_ps10.csv
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 兼容 Python 3.10
    import tomli as tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font")

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_utils_v2 import set_patch_axes, setup_plot_style


ACTIVE_ZONE_KEYS = ["toe", "middle_slope", "platform", "crest"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Infer internal blast ledger from slope time-series only.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/slopemine_v2/config_v2.toml"),
        help="Path to the TOML config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """读取并解析配置文件。"""

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    repo_root = config_path.resolve().parents[2]
    config["repo_root"] = repo_root

    for key in ["dataset_dir", "output_dir"]:
        raw_value = Path(config["paths"][key])
        config["paths"][key] = (repo_root / raw_value).resolve() if not raw_value.is_absolute() else raw_value.resolve()

    config["paths"]["raw_csv"] = Path(config["paths"]["raw_csv"]).expanduser().resolve()
    return config


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """创建输出目录。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_dir = Path(config["paths"]["output_dir"]) / "inferred_blast"
    figure_dir = output_dir / "figures"
    heatmap_dir = output_dir / "selected_time_patch_heatmaps"

    for path in [dataset_dir, output_dir, figure_dir, heatmap_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "figure_dir": figure_dir,
        "heatmap_dir": heatmap_dir,
    }


def parse_zone_assignment_rule(rule_path: Path) -> dict[str, dict[str, Any]]:
    """从 markdown 规则文件中解析 zone 名称和 polygon 顶点。"""

    content = rule_path.read_text(encoding="utf-8").splitlines()
    zone_rows: dict[str, dict[str, Any]] = {}
    current_zone_id: int | None = None

    heading_pattern = re.compile(r"^### zone_id_v2=(\d+)\s+(.+)$")
    key_pattern = re.compile(r"^- zone_key_v2：`([^`]+)`$")
    polygon_pattern = re.compile(r"^- polygon：(.+)$")
    point_pattern = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")

    for line in content:
        heading_match = heading_pattern.match(line.strip())
        if heading_match:
            current_zone_id = int(heading_match.group(1))
            zone_rows[str(current_zone_id)] = {
                "zone_id_v2": current_zone_id,
                "zone_name_v2": heading_match.group(2).strip(),
            }
            continue

        if current_zone_id is None:
            continue

        key_match = key_pattern.match(line.strip())
        if key_match:
            zone_rows[str(current_zone_id)]["zone_key_v2"] = key_match.group(1)
            continue

        polygon_match = polygon_pattern.match(line.strip())
        if polygon_match:
            polygon = [
                [float(x_value), float(y_value)]
                for x_value, y_value in point_pattern.findall(polygon_match.group(1))
            ]
            zone_rows[str(current_zone_id)]["polygon"] = polygon

    if len(zone_rows) < 5:
        raise ValueError("zone_assignment_rule.md 解析失败，未识别到完整的 5 个 zone。")
    return zone_rows


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """读取内部反演所需的数据。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    rule_path = dataset_dir / "zone_assignment_rule.md"

    point_meta = pd.read_csv(dataset_dir / "point_meta_v2.csv")
    patch_meta = pd.read_csv(dataset_dir / "patch_meta_v2_ps10.csv")
    patch_series = pd.read_csv(dataset_dir / "patch_series_v2_ps10.csv", parse_dates=["timestamp"])
    zone_rule = parse_zone_assignment_rule(rule_path)

    required_patch_columns = {
        "patch_id",
        "zone_id_v2",
        "zone_key_v2",
        "zone_name_v2",
        "cell_x_min",
        "cell_x_max",
        "cell_y_min",
        "cell_y_max",
        "centroid_x",
        "centroid_y",
    }
    missing_columns = required_patch_columns.difference(patch_meta.columns)
    if missing_columns:
        raise ValueError(f"patch_meta_v2_ps10.csv 缺少必要列：{sorted(missing_columns)}")

    return {
        "point_meta": point_meta,
        "patch_meta": patch_meta,
        "patch_series": patch_series,
        "zone_rule": zone_rule,
        "rule_path": rule_path,
    }


def robust_scale(series: pd.Series, clip_lower: float | None = None) -> pd.Series:
    """使用 median/IQR 计算稳健 z 分数。"""

    values = series.astype(float)
    median_value = float(values.median())
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    iqr = q75 - q25
    if not math.isfinite(iqr) or iqr < 1e-6:
        fallback = float(values.std(ddof=0))
        scale = fallback if math.isfinite(fallback) and fallback >= 1e-6 else 1.0
    else:
        scale = iqr
    scaled = (values - median_value) / scale
    if clip_lower is not None:
        scaled = scaled.clip(lower=clip_lower)
    return scaled


def prepare_patch_frame(patch_meta: pd.DataFrame) -> pd.DataFrame:
    """整理 patch 元数据并预计算代理编号分位。"""

    frame = patch_meta.copy()
    frame["contains_x_min"] = frame["cell_x_min"].astype(float)
    frame["contains_x_max"] = frame["cell_x_max"].astype(float)
    frame["contains_y_min"] = frame["cell_y_min"].astype(float)
    frame["contains_y_max"] = frame["cell_y_max"].astype(float)

    active_frame = frame.loc[frame["zone_key_v2"].isin(ACTIVE_ZONE_KEYS)].copy()
    global_y_edges = np.quantile(active_frame["centroid_y"], [1 / 3, 2 / 3])

    def assign_band(value: float, edges: np.ndarray, prefix: str) -> str:
        if value <= edges[0]:
            return f"{prefix}01"
        if value <= edges[1]:
            return f"{prefix}02"
        return f"{prefix}03"

    frame["proxy_bench_id"] = frame["centroid_y"].map(lambda value: assign_band(float(value), global_y_edges, "B"))

    zone_area_map: dict[str, tuple[float, float]] = {}
    for zone_key, zone_frame in active_frame.groupby("zone_key_v2"):
        zone_area_map[zone_key] = tuple(np.quantile(zone_frame["centroid_x"], [1 / 3, 2 / 3]))

    frame["proxy_blast_area_id"] = frame.apply(
        lambda row: assign_band(float(row["centroid_x"]), np.asarray(zone_area_map.get(row["zone_key_v2"], zone_area_map["toe"])), "A")
        if row["zone_key_v2"] in zone_area_map
        else "A00",
        axis=1,
    )
    return frame


def build_hourly_signal_table(
    patch_series: pd.DataFrame,
    patch_meta: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """构造用于候选时刻筛选的小时级异常表。"""

    score_weights = config["internal_blast_inference"]["score_weights"]
    merged = patch_series.merge(
        patch_meta[["patch_id", "zone_key_v2", "zone_name_v2"]],
        on="patch_id",
        how="left",
        validate="many_to_one",
    )
    active = merged.loc[merged["zone_key_v2"].isin(ACTIVE_ZONE_KEYS)].copy()
    active["abs_vel_mean"] = active["vel_mean"].abs()
    active["abs_acc_mean"] = active["acc_mean"].abs()

    hourly = (
        active.groupby("timestamp", as_index=False)
        .agg(
            disp_max_p95=("disp_max", lambda values: float(np.quantile(values, 0.95))),
            abs_vel_mean_p95=("abs_vel_mean", lambda values: float(np.quantile(values, 0.95))),
            abs_acc_mean_p95=("abs_acc_mean", lambda values: float(np.quantile(values, 0.95))),
            active_ratio_mean=("active_ratio", "mean"),
            valid_ratio_mean=("valid_ratio", "mean"),
            active_patch_count=("patch_id", "count"),
            missing_patch_count=("is_global_missing", "sum"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    hourly["disp_score"] = robust_scale(hourly["disp_max_p95"], clip_lower=0.0)
    hourly["vel_score"] = robust_scale(hourly["abs_vel_mean_p95"], clip_lower=0.0)
    hourly["acc_score"] = robust_scale(hourly["abs_acc_mean_p95"], clip_lower=0.0)
    hourly["active_score"] = robust_scale(hourly["active_ratio_mean"], clip_lower=0.0)
    hourly["score"] = (
        float(score_weights["disp"]) * hourly["disp_score"]
        + float(score_weights["vel"]) * hourly["vel_score"]
        + float(score_weights["acc"]) * hourly["acc_score"]
        + float(score_weights["active"]) * hourly["active_score"]
    )

    peak_half_window = int(config["internal_blast_inference"]["local_peak_half_window"])
    rolling_max = (
        hourly["score"]
        .rolling(window=peak_half_window * 2 + 1, center=True, min_periods=1)
        .max()
    )
    hourly["is_local_peak"] = hourly["score"].ge(rolling_max - 1e-9)
    return hourly


def apply_time_nms(hourly_candidates: pd.DataFrame, min_separation_hours: int) -> pd.DataFrame:
    """按最小时间间隔做非极大值抑制。"""

    selected_indices: list[int] = []
    selected_times: list[pd.Timestamp] = []
    for row in hourly_candidates.sort_values(["score", "timestamp"], ascending=[False, True]).itertuples():
        timestamp = pd.Timestamp(row.timestamp)
        if any(abs((timestamp - chosen).total_seconds()) < min_separation_hours * 3600 for chosen in selected_times):
            continue
        selected_indices.append(int(row.Index))
        selected_times.append(timestamp)

    return hourly_candidates.loc[selected_indices].sort_values("timestamp").reset_index(drop=True)


def select_event_hours(hourly: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    """按阈值自适应选出最终事件时刻。"""

    inference_cfg = config["internal_blast_inference"]
    min_events = int(inference_cfg["min_events"])
    max_events = int(inference_cfg["max_events"])
    target_events = int(inference_cfg["target_events"])
    min_separation = int(inference_cfg["min_event_separation_hours"])
    start_threshold = float(inference_cfg["score_threshold_start"])
    relax_step = float(inference_cfg["score_threshold_relax_step"])
    floor_threshold = float(inference_cfg["score_threshold_floor"])

    peak_frame = hourly.loc[hourly["is_local_peak"]].copy()
    peak_frame = peak_frame.sort_values(["score", "timestamp"], ascending=[False, True]).reset_index(drop=True)

    best_selection: pd.DataFrame | None = None
    best_threshold = start_threshold
    threshold = start_threshold

    while threshold >= floor_threshold - 1e-9:
        candidate_frame = peak_frame.loc[peak_frame["score"] >= threshold].copy()
        if candidate_frame.empty:
            threshold -= relax_step
            continue
        selection = apply_time_nms(candidate_frame, min_separation)
        if best_selection is None:
            best_selection = selection
            best_threshold = threshold
        else:
            current_gap = abs(len(selection) - target_events)
            best_gap = abs(len(best_selection) - target_events)
            if (min_events <= len(selection) <= max_events and not (min_events <= len(best_selection) <= max_events)) or (
                min_events <= len(selection) <= max_events
                and min_events <= len(best_selection) <= max_events
                and (current_gap < best_gap or (current_gap == best_gap and threshold > best_threshold))
            ):
                best_selection = selection
                best_threshold = threshold
        if min_events <= len(selection) <= max_events:
            if len(selection) == target_events:
                break
        threshold -= relax_step

    if best_selection is None or best_selection.empty:
        raise ValueError("内部反演失败：没有找到任何候选事件时刻。")

    if len(best_selection) > max_events:
        best_selection = best_selection.sort_values(["score", "timestamp"], ascending=[False, True]).head(max_events)
        best_selection = best_selection.sort_values("timestamp").reset_index(drop=True)
    elif len(best_selection) < min_events:
        fallback = apply_time_nms(hourly.sort_values(["score", "timestamp"], ascending=[False, True]), min_separation)
        best_selection = fallback.head(min_events).sort_values("timestamp").reset_index(drop=True)
        best_threshold = float(best_selection["score"].min())

    return best_selection.reset_index(drop=True), best_threshold


def locate_patch_for_coordinate(x_value: float, y_value: float, patch_meta: pd.DataFrame) -> pd.Series | None:
    """根据坐标定位所属 patch；若落空则返回 None。"""

    mask = (
        patch_meta["zone_key_v2"].isin(ACTIVE_ZONE_KEYS)
        & patch_meta["contains_x_min"].le(x_value)
        & patch_meta["contains_x_max"].ge(x_value)
        & patch_meta["contains_y_min"].le(y_value)
        & patch_meta["contains_y_max"].ge(y_value)
    )
    matches = patch_meta.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_event_row(
    *,
    event_index: int,
    timestamp: pd.Timestamp,
    patch_slice: pd.DataFrame,
    patch_meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """根据单个时刻的 patch 响应拟合事件坐标与属性。"""

    score_weights = config["internal_blast_inference"]["score_weights"]
    prior_weights = config["internal_blast_inference"]["zone_prior_weights"]
    support_ratio = float(config["internal_blast_inference"]["support_patch_ratio"])
    toe_only_ratio = float(config["internal_blast_inference"]["toe_only_support_ratio"])

    frame = patch_slice.merge(
        patch_meta[
            [
                "patch_id",
                "zone_id_v2",
                "zone_key_v2",
                "zone_name_v2",
                "centroid_x",
                "centroid_y",
                "contains_x_min",
                "contains_x_max",
                "contains_y_min",
                "contains_y_max",
                "proxy_bench_id",
                "proxy_blast_area_id",
            ]
        ],
        on="patch_id",
        how="left",
        validate="one_to_one",
    ).copy()
    frame = frame.loc[frame["zone_key_v2"].isin(ACTIVE_ZONE_KEYS)].reset_index(drop=True)
    frame["abs_vel_mean"] = frame["vel_mean"].abs()
    frame["abs_acc_mean"] = frame["acc_mean"].abs()
    frame["disp_score"] = robust_scale(frame["disp_max"], clip_lower=0.0)
    frame["vel_score"] = robust_scale(frame["abs_vel_mean"], clip_lower=0.0)
    frame["acc_score"] = robust_scale(frame["abs_acc_mean"], clip_lower=0.0)
    frame["active_score"] = robust_scale(frame["active_ratio"], clip_lower=0.0)
    frame["raw_response_score"] = (
        float(score_weights["disp"]) * frame["disp_score"]
        + float(score_weights["vel"]) * frame["vel_score"]
        + float(score_weights["acc"]) * frame["acc_score"]
        + float(score_weights["active"]) * frame["active_score"]
    )
    frame["zone_prior"] = frame["zone_key_v2"].map(lambda key: float(prior_weights.get(key, 0.0)))
    frame["weighted_response_score"] = frame["raw_response_score"] * frame["zone_prior"]

    positive_frame = frame.loc[frame["weighted_response_score"] > 0].copy()
    if positive_frame.empty:
        positive_frame = frame.nlargest(1, "weighted_response_score").copy()

    support_count = max(1, int(math.ceil(len(positive_frame) * support_ratio)))
    support_frame = positive_frame.nlargest(support_count, "weighted_response_score").copy()
    toe_support_ratio = float((support_frame["zone_key_v2"] == "toe").mean())

    if toe_support_ratio >= toe_only_ratio and (support_frame["zone_key_v2"] == "toe").any():
        coord_frame = support_frame.loc[support_frame["zone_key_v2"] == "toe"].copy()
        coordinate_strategy = "toe_only"
    else:
        coord_frame = support_frame.loc[support_frame["zone_key_v2"].isin(["toe", "middle_slope"])].copy()
        coordinate_strategy = "toe_middle_joint"
        if coord_frame.empty:
            coord_frame = support_frame.copy()
            coordinate_strategy = "support_fallback"

    weights = coord_frame["weighted_response_score"].to_numpy(dtype=float)
    if np.allclose(weights.sum(), 0.0):
        weights = np.ones(len(coord_frame), dtype=float)
    x_value = float(np.average(coord_frame["centroid_x"], weights=weights))
    y_value = float(np.average(coord_frame["centroid_y"], weights=weights))

    located_patch = locate_patch_for_coordinate(x_value, y_value, patch_meta)
    snapped_to_patch = 0
    if located_patch is None:
        coord_xy = coord_frame[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        distance = np.sqrt(np.square(coord_xy[:, 0] - x_value) + np.square(coord_xy[:, 1] - y_value))
        nearest_idx = int(np.argmin(distance))
        nearest_patch_id = str(coord_frame.iloc[nearest_idx]["patch_id"])
        located_patch = patch_meta.loc[patch_meta["patch_id"] == nearest_patch_id].iloc[0]
        x_value = float(located_patch["centroid_x"])
        y_value = float(located_patch["centroid_y"])
        snapped_to_patch = 1

    dominant_zone = str(support_frame["zone_key_v2"].value_counts().idxmax())

    event_row = {
        "blast_id": f"internal_inferred_{timestamp:%Y%m%d_%H%M}_{event_index:02d}",
        "timestamp": timestamp,
        "x_b": x_value,
        "y_b": y_value,
        "zone_id_proxy": int(located_patch["zone_id_v2"]),
        "zone_name_proxy": str(located_patch["zone_name_v2"]),
        "patch_id_proxy": str(located_patch["patch_id"]),
        "coordinate_source": "internal_inferred",
        "is_real_coordinate": 0,
        "proxy_bench_id": str(located_patch["proxy_bench_id"]),
        "proxy_blast_area_id": str(located_patch["proxy_blast_area_id"]),
        "support_patch_count": int(len(support_frame)),
        "coord_patch_count": int(len(coord_frame)),
        "toe_support_ratio": toe_support_ratio,
        "dominant_zone_key": dominant_zone,
        "coordinate_strategy": coordinate_strategy,
        "snapped_to_patch": snapped_to_patch,
        "disp_max_event": float(frame["disp_max"].max()),
        "abs_vel_event": float(frame["abs_vel_mean"].max()),
        "abs_acc_event": float(frame["abs_acc_mean"].max()),
        "active_ratio_event": float(frame["active_ratio"].mean()),
    }
    return event_row, frame


def map_qe_values(event_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """把事件分数映射为 Qe。"""

    qe_min = float(config["internal_blast_inference"]["qe_min"])
    qe_max = float(config["internal_blast_inference"]["qe_max"])
    scores = event_table["score"].astype(float)
    median_value = float(scores.median())
    q25 = float(scores.quantile(0.25))
    q75 = float(scores.quantile(0.75))
    iqr = q75 - q25
    scale = iqr if iqr >= 1e-6 else max(float(scores.std(ddof=0)), 1.0)
    robust = (scores - median_value) / scale
    robust = robust.clip(lower=-1.5, upper=2.5)
    normalized = (robust - robust.min()) / max(robust.max() - robust.min(), 1e-6)
    event_table["Q"] = (qe_min + normalized * (qe_max - qe_min)).round(6)
    return event_table


def build_event_ledger(
    hourly: pd.DataFrame,
    selected_hours: pd.DataFrame,
    patch_series: pd.DataFrame,
    patch_meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.DataFrame]]:
    """构建正式推断爆破台账。"""

    hourly_lookup = hourly.set_index("timestamp")
    event_rows: list[dict[str, Any]] = []
    heatmap_frames: dict[pd.Timestamp, pd.DataFrame] = {}

    for event_index, timestamp in enumerate(selected_hours["timestamp"], start=1):
        patch_slice = patch_series.loc[patch_series["timestamp"] == timestamp].copy()
        event_row, heatmap_frame = build_event_row(
            event_index=event_index,
            timestamp=pd.Timestamp(timestamp),
            patch_slice=patch_slice,
            patch_meta=patch_meta,
            config=config,
        )
        hourly_row = hourly_lookup.loc[pd.Timestamp(timestamp)]
        event_row.update(
            {
                "score": float(hourly_row["score"]),
                "disp_score_hour": float(hourly_row["disp_score"]),
                "vel_score_hour": float(hourly_row["vel_score"]),
                "acc_score_hour": float(hourly_row["acc_score"]),
                "active_score_hour": float(hourly_row["active_score"]),
                "disp_max_p95_hour": float(hourly_row["disp_max_p95"]),
                "abs_vel_mean_p95_hour": float(hourly_row["abs_vel_mean_p95"]),
                "abs_acc_mean_p95_hour": float(hourly_row["abs_acc_mean_p95"]),
                "active_ratio_mean_hour": float(hourly_row["active_ratio_mean"]),
            }
        )
        event_rows.append(event_row)
        heatmap_frames[pd.Timestamp(timestamp)] = heatmap_frame

    event_table = pd.DataFrame(event_rows).sort_values("timestamp").reset_index(drop=True)
    event_table = map_qe_values(event_table, config)
    ordered_columns = [
        "blast_id",
        "timestamp",
        "x_b",
        "y_b",
        "Q",
        "zone_id_proxy",
        "zone_name_proxy",
        "patch_id_proxy",
        "coordinate_source",
        "is_real_coordinate",
        "proxy_bench_id",
        "proxy_blast_area_id",
        "score",
        "support_patch_count",
        "coord_patch_count",
        "toe_support_ratio",
        "dominant_zone_key",
        "coordinate_strategy",
        "snapped_to_patch",
        "disp_max_event",
        "abs_vel_event",
        "abs_acc_event",
        "active_ratio_event",
        "disp_max_p95_hour",
        "abs_vel_mean_p95_hour",
        "abs_acc_mean_p95_hour",
        "active_ratio_mean_hour",
        "disp_score_hour",
        "vel_score_hour",
        "acc_score_hour",
        "active_score_hour",
    ]
    return event_table[ordered_columns], heatmap_frames


def summarize_monotonicity(event_table: pd.DataFrame) -> dict[str, float]:
    """计算 Q 与内部异常指标的秩相关。"""

    return {
        "disp_max_event_spearman": float(event_table["Q"].corr(event_table["disp_max_event"], method="spearman")),
        "abs_vel_event_spearman": float(event_table["Q"].corr(event_table["abs_vel_event"], method="spearman")),
        "abs_acc_event_spearman": float(event_table["Q"].corr(event_table["abs_acc_event"], method="spearman")),
    }


def plot_patch_response_heatmap(
    *,
    patch_meta: pd.DataFrame,
    response_frame: pd.DataFrame,
    event_row: pd.Series,
    zone_rule: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """绘制单个代表性时刻的 patch 响应热力图。"""

    zone_color_lookup = {
        zone_row["zone_key_v2"]: zone_row["zone_id_v2"]
        for zone_row in zone_rule.values()
    }
    active = response_frame.loc[response_frame["zone_key_v2"].isin(ACTIVE_ZONE_KEYS)].copy()
    positive = active.loc[active["weighted_response_score"] > 0].copy()

    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    ax.set_facecolor("#F5F5F5")

    for zone_row in zone_rule.values():
        polygon = zone_row["polygon"]
        x_values = [vertex[0] for vertex in polygon] + [polygon[0][0]]
        y_values = [vertex[1] for vertex in polygon] + [polygon[0][1]]
        ax.plot(
            x_values,
            y_values,
            color="#8A8A8A",
            linewidth=0.8,
            linestyle=(0, (4, 2)),
            alpha=0.8,
            zorder=1,
        )

    if not positive.empty:
        vmax = float(np.quantile(positive["weighted_response_score"], 0.99))
        vmax = max(vmax, float(positive["weighted_response_score"].max()), 1e-6)
        norm = colors.Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap("YlOrRd")
        for row in positive.itertuples(index=False):
            rect = Rectangle(
                (float(row.contains_x_min), float(row.contains_y_min)),
                float(row.contains_x_max - row.contains_x_min),
                float(row.contains_y_max - row.contains_y_min),
                facecolor=cmap(norm(float(row.weighted_response_score))),
                edgecolor="#FFFFFF",
                linewidth=0.35,
                alpha=0.98,
                zorder=2,
            )
            ax.add_patch(rect)
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.82, pad=0.02)
        colorbar.set_label("patch 响应分数", rotation=90)

    ax.scatter(
        [float(event_row["x_b"])],
        [float(event_row["y_b"])],
        s=90,
        c="#111111",
        edgecolors="white",
        linewidths=0.8,
        marker="*",
        zorder=4,
    )
    ax.set_title(
        f"{pd.Timestamp(event_row['timestamp']):%Y-%m-%d %H:%M} | "
        f"Qe={float(event_row['Q']):.1f} | {event_row['zone_name_proxy']}"
    )
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    set_patch_axes(ax, patch_meta)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_internal_spatial_distribution(
    *,
    point_meta: pd.DataFrame,
    event_table: pd.DataFrame,
    out_path: Path,
    config: dict[str, Any],
) -> None:
    """绘制内部推断事件的空间分布图。"""

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.scatter(
        point_meta["grid_x"],
        point_meta["grid_y"],
        s=3,
        c="#D8D8D8",
        alpha=0.28,
        linewidths=0.0,
        rasterized=True,
        zorder=1,
    )

    q_min = float(event_table["Q"].min())
    q_max = float(event_table["Q"].max())
    size_min = float(config["internal_blast_inference"]["plot_event_size_min"])
    size_max = float(config["internal_blast_inference"]["plot_event_size_max"])
    if math.isclose(q_min, q_max):
        sizes = np.full(len(event_table), (size_min + size_max) / 2.0)
    else:
        sizes = size_min + (event_table["Q"].to_numpy(dtype=float) - q_min) / (q_max - q_min) * (size_max - size_min)

    zone_colors = {
        "toe": "#F4A261",
        "middle_slope": "#2A9D8F",
        "platform": "#E9C46A",
        "crest": "#457B9D",
    }
    colors_list = [zone_colors.get(zone_key, "#444444") for zone_key in event_table["dominant_zone_key"]]
    ax.scatter(
        event_table["x_b"],
        event_table["y_b"],
        s=sizes,
        c=colors_list,
        edgecolors="#111111",
        linewidths=0.6,
        alpha=0.95,
        zorder=3,
    )
    for row in event_table.itertuples(index=False):
        ax.text(float(row.x_b) + 1.2, float(row.y_b) + 1.2, str(row.blast_id).split("_")[-1], fontsize=7.5, color="#333333")

    x_min = float(point_meta["grid_x"].min()) - 8.0
    x_max = float(point_meta["grid_x"].max()) + 8.0
    y_min = float(point_meta["grid_y"].min()) - 8.0
    y_max = float(point_meta["grid_y"].max()) + 8.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.set_title("内部反演疑似爆破事件空间分布图")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_zone_overlay(
    *,
    point_meta: pd.DataFrame,
    event_table: pd.DataFrame,
    zone_rule: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """绘制工程分区叠加的最终事件图。"""

    zone_colors = {
        "stable_background": "#D9D9D9",
        "toe": "#F4A261",
        "middle_slope": "#2A9D8F",
        "platform": "#E9C46A",
        "crest": "#457B9D",
    }

    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    for zone_key, zone_points in point_meta.groupby("zone_key_v2"):
        ax.scatter(
            zone_points["grid_x"],
            zone_points["grid_y"],
            s=4,
            c=zone_colors.get(zone_key, "#999999"),
            alpha=0.45 if zone_key == "stable_background" else 0.68,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )

    for zone_row in zone_rule.values():
        polygon = zone_row["polygon"]
        x_values = [vertex[0] for vertex in polygon] + [polygon[0][0]]
        y_values = [vertex[1] for vertex in polygon] + [polygon[0][1]]
        ax.plot(
            x_values,
            y_values,
            color="#666666",
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            alpha=0.85,
            zorder=2,
        )

    ax.scatter(
        event_table["x_b"],
        event_table["y_b"],
        s=92,
        c="#111111",
        edgecolors="white",
        linewidths=0.8,
        marker="*",
        zorder=4,
    )
    x_min = float(point_meta["grid_x"].min()) - 8.0
    x_max = float(point_meta["grid_x"].max()) + 8.0
    y_min = float(point_meta["grid_y"].min()) - 8.0
    y_max = float(point_meta["grid_y"].max()) + 8.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.set_title("内部反演事件与工程分区叠加图")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_candidate_ranking(
    hourly: pd.DataFrame,
    selected_hours: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """输出候选小时排名表。"""

    ranking = hourly.sort_values(["score", "timestamp"], ascending=[False, True]).reset_index(drop=True).copy()
    selected_map = {
        pd.Timestamp(row.timestamp): int(position)
        for position, row in enumerate(selected_hours.sort_values("timestamp").itertuples(index=False), start=1)
    }
    ranking["selected_rank"] = ranking["timestamp"].map(selected_map).fillna(0).astype(int)
    ranking["is_selected"] = ranking["selected_rank"].gt(0).astype(int)
    ranking["selection_threshold"] = threshold
    return ranking


def build_diagnostic_report(
    *,
    output_path: Path,
    event_table: pd.DataFrame,
    candidate_ranking: pd.DataFrame,
    threshold: float,
    monotonicity: dict[str, float],
    heatmap_dir: Path,
    figure_dir: Path,
    config_path: Path,
    rule_path: Path,
) -> None:
    """输出 markdown 诊断报告。"""

    top_candidates = candidate_ranking.head(10)
    selected_rows = event_table.sort_values("score", ascending=False).head(10)
    boundary_rows = candidate_ranking.loc[candidate_ranking["is_selected"] == 0].head(8)

    toe_ratio = float((event_table["zone_name_proxy"] == "坡脚区").mean())
    fallback_count = int((event_table["zone_name_proxy"] != "坡脚区").sum())

    direct_support = "是"
    monotonic_ok = all(value >= 0.65 for value in monotonicity.values())

    lines = [
        "# internal_blast_diagnostic_report_v1",
        "",
        "## 方法边界",
        "",
        "- 本次结果只基于边坡内部位移/速度/加速度/活跃度时序异常反演，不读取旧爆破小时表、真实爆破台账或代理爆破台账。",
        f"- 使用配置文件：`{config_path}`",
        f"- 使用分区规则：`{rule_path}`",
        f"- 最终事件数：{len(event_table)}，阈值={threshold:.2f}，最小事件间隔=6 小时。",
        "- `Q` 字段表示 `Qe` 等效强度，不代表真实装药量。",
        "",
        "## 必答结论",
        "",
        f"- 这些事件是否由边坡内部异常直接支撑，而不是旧爆破表迁移出来的：{direct_support}。脚本只读取 `patch_series_v2_ps10.csv`、`point_meta_v2.csv`、`patch_meta_v2_ps10.csv` 和冻结分区规则。",
        f"- 事件是否主要落在 toe 区，以及有多少比例因证据不足而回退到其他区：toe 占比={toe_ratio:.2%}，非 toe 回退事件数={fallback_count}。",
        f"- `Qe` 与位移峰值、速度峰值、加速度峰值之间是否保持单调一致：{'是' if monotonic_ok else '部分成立'}，Spearman 相关分别为 disp={monotonicity['disp_max_event_spearman']:.3f}、vel={monotonicity['abs_vel_event_spearman']:.3f}、acc={monotonicity['abs_acc_event_spearman']:.3f}。",
        "",
        "## 候选小时总排名（Top 10）",
        "",
        "| rank | timestamp | score | disp_p95 | |vel|_p95 | |acc|_p95 | active_ratio | local_peak | selected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top_candidates.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {pd.Timestamp(row.timestamp):%Y-%m-%d %H:%M} | {float(row.score):.4f} | {float(row.disp_max_p95):.4f} | {float(row.abs_vel_mean_p95):.4f} | {float(row.abs_acc_mean_p95):.4f} | {float(row.active_ratio_mean):.4f} | {int(row.is_local_peak)} | {int(row.is_selected)} |"
        )

    lines.extend(
        [
            "",
            "## 最终入选事件（Top 10 by score）",
            "",
            "| blast_id | timestamp | zone | patch_id_proxy | Qe | support_patch_count | toe_support_ratio | strategy |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in selected_rows.itertuples(index=False):
        lines.append(
            f"| {row.blast_id} | {pd.Timestamp(row.timestamp):%Y-%m-%d %H:%M} | {row.zone_name_proxy} | {row.patch_id_proxy} | {float(row.Q):.2f} | {int(row.support_patch_count)} | {float(row.toe_support_ratio):.2%} | {row.coordinate_strategy} |"
        )

    lines.extend(
        [
            "",
            "## 边界样本（高分但未入选）",
            "",
            "| timestamp | score | disp_p95 | |vel|_p95 | |acc|_p95 | reason |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in boundary_rows.itertuples(index=False):
        reason = "被更高分事件在 6 小时窗口内压制" if int(row.is_local_peak) == 1 else "不是局部峰值"
        lines.append(
            f"| {pd.Timestamp(row.timestamp):%Y-%m-%d %H:%M} | {float(row.score):.4f} | {float(row.disp_max_p95):.4f} | {float(row.abs_vel_mean_p95):.4f} | {float(row.abs_acc_mean_p95):.4f} | {reason} |"
        )

    lines.extend(
        [
            "",
            "## 图件输出",
            "",
            f"- 事件空间分布图：`{figure_dir / 'internal_blast_spatial_distribution.png'}`",
            f"- 分区叠加图：`{figure_dir / 'internal_blast_zone_overlay.png'}`",
            f"- 代表性时刻热力图目录：`{heatmap_dir}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """执行内部爆破事件反演。"""

    args = parse_args()
    config = load_config(args.config)
    setup_plot_style()
    paths = ensure_output_dirs(config)
    inputs = load_inputs(config)

    patch_meta = prepare_patch_frame(inputs["patch_meta"])
    point_meta = inputs["point_meta"]
    patch_series = inputs["patch_series"]
    zone_rule = inputs["zone_rule"]

    hourly = build_hourly_signal_table(
        patch_series=patch_series,
        patch_meta=patch_meta,
        config=config,
    )
    selected_hours, threshold = select_event_hours(hourly, config)
    event_table, heatmap_frames = build_event_ledger(
        hourly=hourly,
        selected_hours=selected_hours,
        patch_series=patch_series,
        patch_meta=patch_meta,
        config=config,
    )
    monotonicity = summarize_monotonicity(event_table)
    candidate_ranking = build_candidate_ranking(hourly, selected_hours, threshold)

    dataset_dir = paths["dataset_dir"]
    output_dir = paths["output_dir"]
    figure_dir = paths["figure_dir"]
    heatmap_dir = paths["heatmap_dir"]

    ledger_path = dataset_dir / "inferred_blast_ledger_internal_v1.csv"
    candidate_path = output_dir / "internal_blast_candidate_hour_ranking.csv"
    report_path = output_dir / "internal_blast_diagnostic_report_v1.md"

    event_table.to_csv(ledger_path, index=False)
    candidate_ranking.to_csv(candidate_path, index=False)

    plot_internal_spatial_distribution(
        point_meta=point_meta,
        event_table=event_table,
        out_path=figure_dir / "internal_blast_spatial_distribution.png",
        config=config,
    )
    plot_zone_overlay(
        point_meta=point_meta,
        event_table=event_table,
        zone_rule=zone_rule,
        out_path=figure_dir / "internal_blast_zone_overlay.png",
    )

    representative_events = event_table.sort_values("score", ascending=False).head(3)
    for row in representative_events.itertuples(index=False):
        response_frame = heatmap_frames[pd.Timestamp(row.timestamp)]
        plot_patch_response_heatmap(
            patch_meta=patch_meta,
            response_frame=response_frame,
            event_row=pd.Series(row._asdict()),
            zone_rule=zone_rule,
            out_path=heatmap_dir / f"internal_event_patch_response_{pd.Timestamp(row.timestamp):%Y%m%d_%H%M}.png",
        )

    build_diagnostic_report(
        output_path=report_path,
        event_table=event_table,
        candidate_ranking=candidate_ranking,
        threshold=threshold,
        monotonicity=monotonicity,
        heatmap_dir=heatmap_dir,
        figure_dir=figure_dir,
        config_path=args.config.resolve(),
        rule_path=inputs["rule_path"],
    )

    print(ledger_path)
    print(candidate_path)
    print(report_path)
    print(figure_dir / "internal_blast_spatial_distribution.png")
    print(figure_dir / "internal_blast_zone_overlay.png")
    print(heatmap_dir)


if __name__ == "__main__":
    main()
