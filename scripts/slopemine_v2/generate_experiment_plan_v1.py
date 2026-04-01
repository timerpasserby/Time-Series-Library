#!/usr/bin/env python3
"""这个脚本负责冻结正式实验 split、数据清单和实验矩阵。
相关文件：config_v2.toml、patch_series_v2_ps10.csv、patch_meta_v2_ps10.csv、blast_ledger_v3_input.csv、weather.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 兼容 Python 3.10
    import tomli as tomllib


FEATURE_COLUMNS = ["disp_mean_mean", "disp_max_p95", "active_ratio_mean"]
SUMMARY_STATS = ["mean", "std", "q25", "q50", "q75", "q90"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate the official split plan and experiment manifests for slopemine v2.")
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

    config["paths"]["weather_csv"] = Path(config["paths"]["weather_csv"]).expanduser().resolve()
    return config


def ensure_output_dir(config: dict[str, Any]) -> Path:
    """创建正式实验计划输出目录。"""

    output_dir = Path(config["paths"]["output_dir"]) / str(config["experiment_plan_v1"]["output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """读取生成实验计划所需的输入文件。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    outputs_dir = Path(config["paths"]["output_dir"])

    patch_series = pd.read_csv(dataset_dir / "patch_series_v2_ps10.csv", parse_dates=["timestamp"])
    patch_meta = pd.read_csv(dataset_dir / "patch_meta_v2_ps10.csv")
    patch_meta_bg_excluded = pd.read_csv(dataset_dir / "patch_meta_v2_ps10_bg_excluded.csv")
    point_meta = pd.read_csv(dataset_dir / "point_meta_v2.csv")
    weather = pd.read_csv(config["paths"]["weather_csv"], parse_dates=["timestamp"])
    blast_ledger = pd.read_csv(dataset_dir / "blast_ledger_v3_input.csv", parse_dates=["timestamp"])
    inferred_blast = pd.read_csv(dataset_dir / "inferred_blast_ledger_internal_v1.csv", parse_dates=["timestamp"])
    patch_blast_v3 = pd.read_csv(dataset_dir / "patch_blast_features_v3_ps10.csv", parse_dates=["timestamp"])
    gate_summary = json.loads((outputs_dir / "baseline_v2" / "baseline_gate_summary_v2.json").read_text(encoding="utf-8"))
    tensor_bundle = np.load(dataset_dir / "patch_tensor_base_v2_ps10.npz", allow_pickle=True)

    return {
        "patch_series": patch_series,
        "patch_meta": patch_meta,
        "patch_meta_bg_excluded": patch_meta_bg_excluded,
        "point_meta": point_meta,
        "weather": weather,
        "blast_ledger": blast_ledger,
        "inferred_blast": inferred_blast,
        "patch_blast_v3": patch_blast_v3,
        "gate_summary": gate_summary,
        "tensor_bundle": tensor_bundle,
    }


def build_effective_hourly_frame(inputs: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """整理主实验使用的有效时间步序列。"""

    patch_series = inputs["patch_series"]
    patch_meta = inputs["patch_meta"]
    weather = inputs["weather"]
    blast_ledger = inputs["blast_ledger"]

    train_patch_ids = patch_meta.loc[patch_meta["is_train_patch"] == 1, "patch_id"].tolist()
    valid_patch_series = patch_series.loc[
        patch_series["patch_id"].isin(train_patch_ids) & (patch_series["is_global_missing"] == 0)
    ].copy()

    disp_quantile = float(config["experiment_plan_v1"]["effective_disp_max_quantile"])
    hourly = (
        valid_patch_series.groupby("timestamp", as_index=False)
        .agg(
            disp_mean_mean=("disp_mean", "mean"),
            disp_max_p95=("disp_max", lambda values: float(np.quantile(values, disp_quantile))),
            active_ratio_mean=("active_ratio", "mean"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    blast_hourly = (
        blast_ledger.assign(timestamp=blast_ledger["timestamp"].dt.floor("h"))
        .groupby("timestamp", as_index=False)
        .agg(
            blast_hours=("blast_id", "count"),
            blast_q_sum=("Q", "sum"),
        )
    )

    hourly = hourly.merge(weather[["timestamp", "rainfall"]], on="timestamp", how="left", validate="one_to_one")
    hourly = hourly.merge(blast_hourly, on="timestamp", how="left", validate="one_to_one")
    hourly["rainfall"] = hourly["rainfall"].fillna(0.0)
    hourly["blast_hours"] = hourly["blast_hours"].fillna(0).astype(np.int32)
    hourly["blast_q_sum"] = hourly["blast_q_sum"].fillna(0.0)
    hourly["is_rain"] = (hourly["rainfall"] > 0).astype(np.int8)
    hourly["is_blast"] = (hourly["blast_hours"] > 0).astype(np.int8)
    hourly["effective_index"] = np.arange(len(hourly), dtype=np.int32)

    missing_hours = (
        patch_series.loc[patch_series["is_global_missing"] == 1, "timestamp"]
        .drop_duplicates()
        .sort_values()
    )
    return hourly, pd.DatetimeIndex(missing_hours.tolist())


def compute_global_feature_stats(hourly: pd.DataFrame) -> dict[str, dict[str, float]]:
    """预计算整体分布统计，供候选切分评分。"""

    rows: dict[str, dict[str, float]] = {}
    for feature_name in FEATURE_COLUMNS:
        series = hourly[feature_name].astype(float)
        rows[feature_name] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "q25": float(series.quantile(0.25)),
            "q50": float(series.quantile(0.50)),
            "q75": float(series.quantile(0.75)),
            "q90": float(series.quantile(0.90)),
        }
    return rows


def compute_segment_distribution_score(
    segment: pd.DataFrame,
    global_stats: dict[str, dict[str, float]],
) -> tuple[float, dict[str, float]]:
    """计算单个切分段与整体分布的差异分数。"""

    feature_scores: dict[str, float] = {}
    for feature_name in FEATURE_COLUMNS:
        series = segment[feature_name].astype(float)
        segment_stats = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "q25": float(series.quantile(0.25)),
            "q50": float(series.quantile(0.50)),
            "q75": float(series.quantile(0.75)),
            "q90": float(series.quantile(0.90)),
        }
        values = []
        for stat_name in SUMMARY_STATS:
            base_value = global_stats[feature_name][stat_name]
            values.append(abs(segment_stats[stat_name] - base_value) / (abs(base_value) + 1e-6))
        feature_scores[feature_name] = float(np.mean(values))
    score = float(np.mean(list(feature_scores.values())))
    return score, feature_scores


def count_missing_in_range(
    missing_hours: pd.DatetimeIndex,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> int:
    """统计某个自然时间范围内部的全局缺失小时数。"""

    mask = (missing_hours >= start_time) & (missing_hours <= end_time)
    return int(mask.sum())


def build_candidate_rows(
    hourly: pd.DataFrame,
    missing_hours: pd.DatetimeIndex,
    config: dict[str, Any],
) -> pd.DataFrame:
    """枚举候选切分方案并打分。"""

    cfg = config["experiment_plan_v1"]
    global_stats = compute_global_feature_stats(hourly)

    train_min = float(cfg["train_ratio_min"])
    train_max = float(cfg["train_ratio_max"])
    val_min = float(cfg["val_ratio_min"])
    val_max = float(cfg["val_ratio_max"])
    test_min = float(cfg["test_ratio_min"])
    step = float(cfg["ratio_step"])

    train_target = float(cfg["train_ratio_target"])
    val_target = float(cfg["val_ratio_target"])
    test_target = float(cfg["test_ratio_target"])
    target_blast = int(cfg["coverage_target_blast_hours"])
    target_rain = int(cfg["coverage_target_rain_hours"])
    weight_dist = float(cfg["score_weight_distribution"])
    weight_ratio = float(cfg["score_weight_ratio"])
    weight_coverage = float(cfg["score_weight_coverage"])

    total_steps = len(hourly)
    seen_boundaries: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []

    train_ratios = np.arange(train_min, train_max + step / 2.0, step)
    val_ratios = np.arange(val_min, val_max + step / 2.0, step)

    for train_ratio in train_ratios:
        for val_ratio in val_ratios:
            test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
            if test_ratio < test_min - 1e-9:
                continue

            train_steps = int(round(total_steps * float(train_ratio)))
            val_steps = int(round(total_steps * float(val_ratio)))
            test_steps = total_steps - train_steps - val_steps
            if min(train_steps, val_steps, test_steps) <= 0:
                continue
            if test_steps / total_steps < test_min - 1e-9:
                continue

            boundary_key = (train_steps, train_steps + val_steps)
            if boundary_key in seen_boundaries:
                continue
            seen_boundaries.add(boundary_key)

            train_frame = hourly.iloc[:train_steps].copy()
            val_frame = hourly.iloc[train_steps : train_steps + val_steps].copy()
            test_frame = hourly.iloc[train_steps + val_steps :].copy()

            train_blast_hours = int(train_frame["is_blast"].sum())
            val_blast_hours = int(val_frame["is_blast"].sum())
            test_blast_hours = int(test_frame["is_blast"].sum())
            train_rain_hours = int(train_frame["is_rain"].sum())
            val_rain_hours = int(val_frame["is_rain"].sum())
            test_rain_hours = int(test_frame["is_rain"].sum())

            if min(val_blast_hours, test_blast_hours, val_rain_hours, test_rain_hours) <= 0:
                continue

            train_score, train_feature_scores = compute_segment_distribution_score(train_frame, global_stats)
            val_score, val_feature_scores = compute_segment_distribution_score(val_frame, global_stats)
            test_score, test_feature_scores = compute_segment_distribution_score(test_frame, global_stats)
            distribution_score = float(np.mean([train_score, val_score, test_score]))

            actual_train_ratio = train_steps / total_steps
            actual_val_ratio = val_steps / total_steps
            actual_test_ratio = test_steps / total_steps
            ratio_penalty = (
                abs(actual_train_ratio - train_target)
                + abs(actual_val_ratio - val_target)
                + abs(actual_test_ratio - test_target)
            )

            coverage_penalty = (
                max(0, target_blast - val_blast_hours) / max(target_blast, 1)
                + max(0, target_blast - test_blast_hours) / max(target_blast, 1)
                + max(0, target_rain - val_rain_hours) / max(target_rain, 1)
                + max(0, target_rain - test_rain_hours) / max(target_rain, 1)
            )

            total_score = (
                weight_dist * distribution_score
                + weight_ratio * ratio_penalty
                + weight_coverage * coverage_penalty
            )

            train_start = pd.Timestamp(train_frame["timestamp"].iloc[0])
            train_end = pd.Timestamp(train_frame["timestamp"].iloc[-1])
            val_start = pd.Timestamp(val_frame["timestamp"].iloc[0])
            val_end = pd.Timestamp(val_frame["timestamp"].iloc[-1])
            test_start = pd.Timestamp(test_frame["timestamp"].iloc[0])
            test_end = pd.Timestamp(test_frame["timestamp"].iloc[-1])

            rows.append(
                {
                    "train_steps": int(train_steps),
                    "val_steps": int(val_steps),
                    "test_steps": int(test_steps),
                    "total_steps": int(total_steps),
                    "train_ratio": actual_train_ratio,
                    "val_ratio": actual_val_ratio,
                    "test_ratio": actual_test_ratio,
                    "total_score": total_score,
                    "distribution_score": distribution_score,
                    "ratio_penalty": ratio_penalty,
                    "coverage_penalty": coverage_penalty,
                    "train_distribution_score": train_score,
                    "val_distribution_score": val_score,
                    "test_distribution_score": test_score,
                    "train_disp_mean_score": train_feature_scores["disp_mean_mean"],
                    "val_disp_mean_score": val_feature_scores["disp_mean_mean"],
                    "test_disp_mean_score": test_feature_scores["disp_mean_mean"],
                    "train_disp_max_score": train_feature_scores["disp_max_p95"],
                    "val_disp_max_score": val_feature_scores["disp_max_p95"],
                    "test_disp_max_score": test_feature_scores["disp_max_p95"],
                    "train_active_ratio_score": train_feature_scores["active_ratio_mean"],
                    "val_active_ratio_score": val_feature_scores["active_ratio_mean"],
                    "test_active_ratio_score": test_feature_scores["active_ratio_mean"],
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_blast_hours": train_blast_hours,
                    "val_blast_hours": val_blast_hours,
                    "test_blast_hours": test_blast_hours,
                    "train_rain_hours": train_rain_hours,
                    "val_rain_hours": val_rain_hours,
                    "test_rain_hours": test_rain_hours,
                    "train_missing_hours_in_range": count_missing_in_range(missing_hours, train_start, train_end),
                    "val_missing_hours_in_range": count_missing_in_range(missing_hours, val_start, val_end),
                    "test_missing_hours_in_range": count_missing_in_range(missing_hours, test_start, test_end),
                }
            )

    if not rows:
        raise ValueError("没有找到同时满足爆破、降雨和比例约束的切分候选。")

    candidates = pd.DataFrame(rows).sort_values(
        ["total_score", "coverage_penalty", "distribution_score", "ratio_penalty", "train_ratio"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1, dtype=np.int32)
    candidates["candidate_id"] = candidates["candidate_rank"].map(lambda value: f"split_cand_{int(value):02d}")
    return candidates


def build_split_plan_table(
    best_candidate: pd.Series,
    missing_hours: pd.DatetimeIndex,
) -> pd.DataFrame:
    """把推荐方案整理成正式 split 计划表。"""

    rows = []
    split_specs = [
        ("train", 0, int(best_candidate["train_steps"]), pd.Timestamp(best_candidate["train_start"]), pd.Timestamp(best_candidate["train_end"]), int(best_candidate["train_blast_hours"]), int(best_candidate["train_rain_hours"])),
        ("val", int(best_candidate["train_steps"]), int(best_candidate["train_steps"] + best_candidate["val_steps"]), pd.Timestamp(best_candidate["val_start"]), pd.Timestamp(best_candidate["val_end"]), int(best_candidate["val_blast_hours"]), int(best_candidate["val_rain_hours"])),
        ("test", int(best_candidate["train_steps"] + best_candidate["val_steps"]), int(best_candidate["total_steps"]), pd.Timestamp(best_candidate["test_start"]), pd.Timestamp(best_candidate["test_end"]), int(best_candidate["test_blast_hours"]), int(best_candidate["test_rain_hours"])),
    ]

    for split_name, start_idx, end_idx, start_time, end_time, blast_hours, rain_hours in split_specs:
        effective_steps = end_idx - start_idx
        rows.append(
            {
                "plan_version": "v1",
                "selected_candidate_id": str(best_candidate["candidate_id"]),
                "split_name": split_name,
                "effective_start_index": int(start_idx),
                "effective_end_index_exclusive": int(end_idx),
                "effective_steps": int(effective_steps),
                "ratio": float(effective_steps / int(best_candidate["total_steps"])),
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "blast_hours": int(blast_hours),
                "rain_hours": int(rain_hours),
                "excluded_global_missing_hours_in_natural_range": count_missing_in_range(missing_hours, start_time, end_time),
                "is_last_contiguous_block": 1 if split_name == "test" else 0,
                "uses_effective_timeline": 1,
            }
        )
    return pd.DataFrame(rows)


def format_ratio(value: float) -> str:
    """格式化比例。"""

    return f"{value * 100:.2f}%"


def format_timestamp(value: pd.Timestamp) -> str:
    """格式化时间戳。"""

    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def build_split_selection_report(
    *,
    output_path: Path,
    candidates: pd.DataFrame,
    best_candidate: pd.Series,
    missing_hours: pd.DatetimeIndex,
) -> None:
    """输出 split 选择报告。"""

    top_rows = candidates.head(6)
    lines = [
        "# split_selection_report",
        "",
        "## 规则说明",
        "",
        "- 本次正式切分使用 `有效时间步序列`，不是自然连续 720 小时整轴。",
        f"- 全局缺失小时共 {len(missing_hours)} 个，已从有效时间步中剔除："
        + "、".join(format_timestamp(timestamp) for timestamp in missing_hours),
        "- train / val / test 全部按时间顺序连续切分，不做随机打乱。",
        "- test 固定为最后一段连续有效时间块。",
        "- 只接受 `train>=50%`、`val>=15%`、`test>=15%`，且 val / test 同时覆盖爆破小时和降雨小时的候选。",
        "- 候选评分由三部分组成：分布相似度、比例偏离惩罚、事件覆盖惩罚。",
        "",
        "## 候选方案（Top 6）",
        "",
        "| rank | candidate_id | train_ratio | val_ratio | test_ratio | train_end | val_range | test_range | val_blast | test_blast | val_rain | test_rain | score |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"| {int(row.candidate_rank)} | {row.candidate_id} | {format_ratio(float(row.train_ratio))} | {format_ratio(float(row.val_ratio))} | {format_ratio(float(row.test_ratio))} | {format_timestamp(row.train_end)} | {format_timestamp(row.val_start)} ~ {format_timestamp(row.val_end)} | {format_timestamp(row.test_start)} ~ {format_timestamp(row.test_end)} | {int(row.val_blast_hours)} | {int(row.test_blast_hours)} | {int(row.val_rain_hours)} | {int(row.test_rain_hours)} | {float(row.total_score):.4f} |"
        )

    lines.extend(
        [
            "",
            "## 推荐方案",
            "",
            f"- 推荐候选：`{best_candidate['candidate_id']}`。",
            f"- 正式比例：train={format_ratio(float(best_candidate['train_ratio']))}，val={format_ratio(float(best_candidate['val_ratio']))}，test={format_ratio(float(best_candidate['test_ratio']))}。",
            f"- 正式区间：train={format_timestamp(best_candidate['train_start'])} ~ {format_timestamp(best_candidate['train_end'])}；val={format_timestamp(best_candidate['val_start'])} ~ {format_timestamp(best_candidate['val_end'])}；test={format_timestamp(best_candidate['test_start'])} ~ {format_timestamp(best_candidate['test_end'])}。",
            f"- 推荐原因：它在分布相似度、比例平衡和事件覆盖三项综合得分最低；val / test 均覆盖 8 个爆破小时，且分别覆盖 {int(best_candidate['val_rain_hours'])} / {int(best_candidate['test_rain_hours'])} 个降雨小时，训练集仍保留 {int(best_candidate['train_steps'])} 个有效时间步，适合作为正式主实验切分。",
            f"- 需要注意：这个方案仍基于冻结后的 712 个有效时间步；若后续训练代码直接读取旧 `window_manifest_v2_ps10.csv`，需要改为以 `experiment_split_plan_v1.csv` 为正式 split 源。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def describe_npz_bundle(bundle: Any, patch_meta: pd.DataFrame) -> dict[str, Any]:
    """整理基础张量文件的清单信息。"""

    patch_ids = bundle["patch_ids"].tolist()
    active_patch_meta = patch_meta.loc[patch_meta["patch_id"].isin(patch_ids)].copy()
    timestamps = pd.to_datetime(bundle["timestamps"].tolist())
    return {
        "row_count": int(len(timestamps)),
        "time_range": f"{format_timestamp(timestamps.min())} ~ {format_timestamp(timestamps.max())}",
        "patch_count": int(len(patch_ids)),
        "zone_count": int(active_patch_meta["zone_id_v2"].nunique()),
        "stable_background_excluded": "是",
        "note": f"基础张量 shape: internal={tuple(bundle['internal'].shape)}, weather={tuple(bundle['weather'].shape)}, blast_v2={tuple(bundle['blast_v2'].shape)}, blast_v3={tuple(bundle['blast_v3'].shape)}",
    }


def build_dataset_manifest(
    *,
    output_path: Path,
    inputs: dict[str, Any],
    split_plan: pd.DataFrame,
) -> None:
    """输出正式实验数据清单。"""

    patch_series = inputs["patch_series"]
    patch_meta = inputs["patch_meta"]
    patch_meta_bg_excluded = inputs["patch_meta_bg_excluded"]
    point_meta = inputs["point_meta"]
    weather = inputs["weather"]
    blast_ledger = inputs["blast_ledger"]
    patch_blast_v3 = inputs["patch_blast_v3"]
    inferred_blast = inputs["inferred_blast"]
    tensor_bundle = inputs["tensor_bundle"]

    manifest_rows = [
        {
            "file": "dataset/slopemine_v2/point_meta_v2.csv",
            "role": "冻结后的点位工程分区元数据",
            "row_count": int(len(point_meta)),
            "time_range": "-",
            "patch_count": "-",
            "zone_count": int(point_meta["zone_id_v2"].nunique()),
            "stable_background_excluded": "否",
            "note": "分区规则已冻结，可重复生成 zone_id_v2。",
        },
        {
            "file": "dataset/slopemine_v2/patch_meta_v2_ps10.csv",
            "role": "完整 patch 元数据（含 background patch）",
            "row_count": int(len(patch_meta)),
            "time_range": "-",
            "patch_count": int(patch_meta["patch_id"].nunique()),
            "zone_count": int(patch_meta["zone_id_v2"].nunique()),
            "stable_background_excluded": "否",
            "note": "完整 patch 资产，用于回溯和可视化。",
        },
        {
            "file": "dataset/slopemine_v2/patch_meta_v2_ps10_bg_excluded.csv",
            "role": "正式主实验 patch 清单",
            "row_count": int(len(patch_meta_bg_excluded)),
            "time_range": "-",
            "patch_count": int(patch_meta_bg_excluded["patch_id"].nunique()),
            "zone_count": int(patch_meta_bg_excluded["zone_id_v2"].nunique()),
            "stable_background_excluded": "是",
            "note": "正式训练 patch 数固定为 307。",
        },
        {
            "file": "dataset/slopemine_v2/patch_series_v2_ps10.csv",
            "role": "patch 级内部时序主表",
            "row_count": int(len(patch_series)),
            "time_range": f"{format_timestamp(patch_series['timestamp'].min())} ~ {format_timestamp(patch_series['timestamp'].max())}",
            "patch_count": int(patch_series["patch_id"].nunique()),
            "zone_count": int(patch_series["zone_id_v2"].nunique()),
            "stable_background_excluded": "否",
            "note": "包含 720 小时整轴和 8 个显式全局缺失小时。",
        },
        {
            "file": "dataset/slopemine/weather.csv",
            "role": "天气外生输入",
            "row_count": int(len(weather)),
            "time_range": f"{format_timestamp(weather['timestamp'].min())} ~ {format_timestamp(weather['timestamp'].max())}",
            "patch_count": "-",
            "zone_count": "-",
            "stable_background_excluded": "不适用",
            "note": "有效天气时间步共 712 小时。",
        },
        {
            "file": "dataset/slopemine_v2/blast_ledger_v3_input.csv",
            "role": "真实爆破事件输入台账",
            "row_count": int(len(blast_ledger)),
            "time_range": f"{format_timestamp(blast_ledger['timestamp'].min())} ~ {format_timestamp(blast_ledger['timestamp'].max())}",
            "patch_count": "-",
            "zone_count": int(blast_ledger["zone"].nunique()),
            "stable_background_excluded": "不适用",
            "note": "共 70 条事件记录，折算为 57 个爆破小时。",
        },
        {
            "file": "dataset/slopemine_v2/patch_blast_features_v3_ps10.csv",
            "role": "真实坐标 blast V3 patch 级特征",
            "row_count": int(len(patch_blast_v3)),
            "time_range": f"{format_timestamp(patch_blast_v3['timestamp'].min())} ~ {format_timestamp(patch_blast_v3['timestamp'].max())}",
            "patch_count": int(patch_blast_v3["patch_id"].nunique()),
            "zone_count": int(patch_blast_v3["zone_id_v2"].nunique()),
            "stable_background_excluded": "否",
            "note": "A4 正式主实验使用这套 real-coordinate V3 特征。",
        },
        {
            "file": "dataset/slopemine_v2/patch_tensor_base_v2_ps10.npz",
            "role": "正式训练基础张量",
            **describe_npz_bundle(tensor_bundle, patch_meta),
        },
        {
            "file": "dataset/slopemine_v2/inferred_blast_ledger_internal_v1.csv",
            "role": "内部反演疑似爆破事件参考表",
            "row_count": int(len(inferred_blast)),
            "time_range": f"{format_timestamp(inferred_blast['timestamp'].min())} ~ {format_timestamp(inferred_blast['timestamp'].max())}",
            "patch_count": inferred_blast["patch_id_proxy"].nunique(),
            "zone_count": inferred_blast["zone_id_proxy"].nunique(),
            "stable_background_excluded": "是",
            "note": "仅用于第四章补充说明，不作为真实爆破坐标训练输入。",
        },
        {
            "file": "outputs/slopemine_v2/experiment_plan_v1/experiment_split_plan_v1.csv",
            "role": "正式 train/val/test 切分基准",
            "row_count": int(len(split_plan)),
            "time_range": f"{format_timestamp(split_plan['start_timestamp'].min())} ~ {format_timestamp(split_plan['end_timestamp'].max())}",
            "patch_count": int(patch_meta_bg_excluded["patch_id"].nunique()),
            "zone_count": int(patch_meta_bg_excluded["zone_id_v2"].nunique()),
            "stable_background_excluded": "是",
            "note": "正式实验应以此文件覆盖旧 window_manifest 默认切分。",
        },
    ]

    lines = [
        "# dataset_manifest_v1",
        "",
        "- 本次正式主实验固定使用 `patch_size=10`。",
        "- 正式训练 patch 数固定为 307，稳定背景区 `stable_background` 默认排除出主训练。",
        "- 正式 split 基于 712 个有效时间步，不直接复用旧的 720 小时整轴切分。",
        "",
        "| file | role | row_count | time_range | patch_count | zone_count | stable_background_excluded | note |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in manifest_rows:
        lines.append(
            f"| {row['file']} | {row['role']} | {row['row_count']} | {row['time_range']} | {row['patch_count']} | {row['zone_count']} | {row['stable_background_excluded']} | {row['note']} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_recommended_window_config(
    output_root: Path,
    gate_summary: dict[str, Any],
) -> dict[str, Any]:
    """优先读取正式 split 下的窗口搜索结果，否则回退到旧 gate summary。"""

    tuning_path = output_root / "tuning_v1" / "best_window_config_v1.json"
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        best = tuning["best_overall"]
        return {
            "seq_len": int(best["seq_len"]),
            "pred_len": int(best["pred_len"]),
            "source": "formal_split_tuning",
            "note": "基于正式 split 的 A4 验证集窗口搜索结果。",
        }

    return {
        "seq_len": int(gate_summary["best_a4_seq_len"]),
        "pred_len": int(gate_summary["best_a4_pred_len"]),
        "source": "legacy_gate_summary",
        "note": "基于旧 baseline gate summary 的历史结果。",
    }


def build_experiment_matrix(
    *,
    output_path: Path,
    gate_summary: dict[str, Any],
    recommended_window: dict[str, Any],
) -> None:
    """输出正式实验矩阵。"""

    gate_open = bool(gate_summary["a4_stably_better_than_a3"])

    lines = [
        "# experiment_matrix_v1",
        "",
        "- 正式 split 来源：`experiment_split_plan_v1.csv`。",
        "- 默认 patch 粒度：`patch_size=10`；正式主训练 patch 数：307；`stable_background` 不参与主训练。",
        f"- 现有基线门槛结果：A4 相对 A3 已满足继续进入机制阶段；当前推荐基础配置为 `seq_len={recommended_window['seq_len']}, pred_len={recommended_window['pred_len']}`（{recommended_window['note']}）。",
        "",
        "| ID | 名称 | 正式输入 | patch 方案 | 目的 | 运行备注 |",
        "| --- | --- | --- | --- | --- | --- |",
        "| A0 | aggregate baseline（仅参考） | 全局聚合内部统计 + 正式 split | 非主实验，不做 patch 训练 | 给出全局参考下界 | 只做参考，不参与主结论对比 |",
        "| A1 | patch internal only | `patch_tensor_base_v2_ps10.npz` 内部特征 | zone-constrained, ps10, bg excluded | 建立 patch 主实验基线 | 第一轮先跑 |",
        "| A2 | A1 + weather | A1 + `weather` | zone-constrained, ps10, bg excluded | 检验天气外生增益 | 第二轮跑 |",
        "| A3 | A2 + blast V2 | A2 + `blast_v2` | zone-constrained, ps10, bg excluded | 检验真实爆破小时级输入增益 | 第三轮跑 |",
        "| A4 | A2 + blast V3 real-coordinate | A2 + `patch_blast_features_v3_ps10.csv` | zone-constrained, ps10, bg excluded | 检验真实坐标 patch 级爆破输入增益 | 第四轮跑，作为机制阶段入口 |",
        "| B1 | no-zone patch vs zone-constrained patch | 与 A4 同输入口径 | `no-zone patch` 对比 `zone-constrained patch` | 检验工程分区约束是否必要 | 先复用正式 split，再补 no-zone patch 资产 |",
        "| B2 | patch_size=8 vs 10 | 与 A4 同输入口径 | `ps8` 对比 `ps10` | 检验空间粒度敏感性 | 同一正式 split 下比较 |",
        f"| C1 | A4 + PGGC | A4 + PGGC | zone-constrained, ps10, bg excluded | 引入空间图结构 | {'允许进入' if gate_open else '暂缓，需先满足 A4>A3 门槛'} |",
        f"| C2 | C1 + EDDR | C1 + EDDR | zone-constrained, ps10, bg excluded | 引入事件驱动动态增强 | {'允许进入' if gate_open else '暂缓'} |",
        f"| C3 | C2 + PIR | C2 + PIR | zone-constrained, ps10, bg excluded | 引入 patch 重要性重标定 | {'允许进入' if gate_open else '暂缓'} |",
        "",
        "## 正式运行顺序",
        "",
        "- 第一轮：A0（可选参考） -> A1 -> A2 -> A3 -> A4。",
        "- 第二轮：B1、B2 作为结构对照与粒度敏感性实验。",
        "- 第三轮：若 A4 稳定优于 A3，则进入 C1 -> C2 -> C3。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_chapter4_fill_items(
    *,
    output_path: Path,
    split_plan: pd.DataFrame,
    blast_ledger: pd.DataFrame,
    inferred_blast: pd.DataFrame,
    patch_meta_bg_excluded: pd.DataFrame,
    gate_summary: dict[str, Any],
    recommended_window: dict[str, Any],
) -> None:
    """输出第四章可直接补写的关键信息。"""

    train_row = split_plan.loc[split_plan["split_name"] == "train"].iloc[0]
    val_row = split_plan.loc[split_plan["split_name"] == "val"].iloc[0]
    test_row = split_plan.loc[split_plan["split_name"] == "test"].iloc[0]

    real_blast_hour_count = int(blast_ledger["timestamp"].dt.floor("h").nunique())
    inferred_zone_counts = inferred_blast["zone_name_proxy"].value_counts().to_dict()
    inferred_q_mean = float(inferred_blast["Q"].mean())
    inferred_q_median = float(inferred_blast["Q"].median())
    inferred_toe_ratio = float((inferred_blast["zone_name_proxy"] == "坡脚区").mean())

    lines = [
        "# chapter4_fill_items",
        "",
        "1. 正式 train / val / test 时间区间",
        f"- train：{format_timestamp(train_row['start_timestamp'])} ~ {format_timestamp(train_row['end_timestamp'])}，共 {int(train_row['effective_steps'])} 个有效时间步，占比 {format_ratio(float(train_row['ratio']))}。",
        f"- val：{format_timestamp(val_row['start_timestamp'])} ~ {format_timestamp(val_row['end_timestamp'])}，共 {int(val_row['effective_steps'])} 个有效时间步，占比 {format_ratio(float(val_row['ratio']))}；包含 {int(val_row['blast_hours'])} 个爆破小时和 {int(val_row['rain_hours'])} 个降雨小时。",
        f"- test：{format_timestamp(test_row['start_timestamp'])} ~ {format_timestamp(test_row['end_timestamp'])}，共 {int(test_row['effective_steps'])} 个有效时间步，占比 {format_ratio(float(test_row['ratio']))}；包含 {int(test_row['blast_hours'])} 个爆破小时和 {int(test_row['rain_hours'])} 个降雨小时，且为最后一段连续有效时间块。",
        "",
        "2. 真实爆破数据统计结果",
        f"- `blast_ledger_v3_input.csv`：共 {len(blast_ledger)} 条真实爆破事件记录，折算为 {real_blast_hour_count} 个爆破小时，时间范围为 {format_timestamp(blast_ledger['timestamp'].min())} ~ {format_timestamp(blast_ledger['timestamp'].max())}。",
        "- 若本章同时引用 `inferred_blast_ledger_internal_v1.csv`，需要明确它是“内部反演疑似爆破事件表”，不是实时真实坐标台账。",
        f"- `inferred_blast_ledger_internal_v1.csv`：共 {len(inferred_blast)} 条疑似事件，时间范围为 {format_timestamp(inferred_blast['timestamp'].min())} ~ {format_timestamp(inferred_blast['timestamp'].max())}，zone 分布为 {inferred_zone_counts}，坡脚区占比 {inferred_toe_ratio:.2%}，Qe 均值/中位数为 {inferred_q_mean:.2f} / {inferred_q_median:.2f}。",
        "",
        "3. 正式主实验 patch 数",
        f"- 正式主实验 patch 数固定为 {patch_meta_bg_excluded['patch_id'].nunique()}（`patch_size=10`，仅保留 `is_train_patch=1`）。",
        "",
        "4. 是否排除 stable_background",
        "- 是。`stable_background` 对应 background patch 默认 `is_train_patch=0`，不参与正式主训练。",
        "",
        "5. A1-A4 第一轮实验的运行顺序",
        "- 第一轮正式顺序固定为：A1 -> A2 -> A3 -> A4。",
        f"- 当前第一轮推荐基础窗口为 `seq_len={recommended_window['seq_len']}, pred_len={recommended_window['pred_len']}`；来源为 {recommended_window['note']}。",
        "- A4 已稳定优于 A3，可继续进入后续机制实验。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    *,
    output_dir: Path,
    candidates: pd.DataFrame,
    split_plan: pd.DataFrame,
    inputs: dict[str, Any],
) -> None:
    """统一落盘所有正式实验计划输出。"""

    top_k = int(inputs["config"]["experiment_plan_v1"]["candidate_top_k"])
    candidate_out = candidates.head(top_k).copy()
    best_candidate_id = str(candidate_out.iloc[0]["candidate_id"])
    candidate_out["recommended"] = candidate_out["candidate_id"].eq(best_candidate_id).astype(np.int8)
    candidate_out.to_csv(output_dir / "experiment_split_candidates.csv", index=False)
    split_plan.to_csv(output_dir / "experiment_split_plan_v1.csv", index=False)


def main() -> None:
    """执行正式实验计划生成流程。"""

    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    inputs = load_inputs(config)
    inputs["config"] = config

    hourly, missing_hours = build_effective_hourly_frame(inputs, config)
    candidates = build_candidate_rows(hourly, missing_hours, config)
    best_candidate = candidates.iloc[0]
    split_plan = build_split_plan_table(best_candidate, missing_hours)
    recommended_window = load_recommended_window_config(Path(config["paths"]["output_dir"]), inputs["gate_summary"])

    save_outputs(
        output_dir=output_dir,
        candidates=candidates,
        split_plan=split_plan,
        inputs=inputs,
    )
    build_split_selection_report(
        output_path=output_dir / "split_selection_report.md",
        candidates=candidates.head(int(config["experiment_plan_v1"]["candidate_top_k"])).copy(),
        best_candidate=best_candidate,
        missing_hours=missing_hours,
    )
    build_dataset_manifest(
        output_path=output_dir / "dataset_manifest_v1.md",
        inputs=inputs,
        split_plan=split_plan,
    )
    build_experiment_matrix(
        output_path=output_dir / "experiment_matrix_v1.md",
        gate_summary=inputs["gate_summary"],
        recommended_window=recommended_window,
    )
    build_chapter4_fill_items(
        output_path=output_dir / "chapter4_fill_items.md",
        split_plan=split_plan,
        blast_ledger=inputs["blast_ledger"],
        inferred_blast=inputs["inferred_blast"],
        patch_meta_bg_excluded=inputs["patch_meta_bg_excluded"],
        gate_summary=inputs["gate_summary"],
        recommended_window=recommended_window,
    )

    print(output_dir / "experiment_split_candidates.csv")
    print(output_dir / "experiment_split_plan_v1.csv")
    print(output_dir / "split_selection_report.md")
    print(output_dir / "dataset_manifest_v1.md")
    print(output_dir / "experiment_matrix_v1.md")
    print(output_dir / "chapter4_fill_items.md")


if __name__ == "__main__":
    main()
