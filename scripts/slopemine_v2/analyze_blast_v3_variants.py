#!/usr/bin/env python3
"""这个脚本负责分析 blast V3 的局部事件敏感变体。
相关文件：config_v2.toml、blast_ledger_v3_input.csv、proxy_blast_ledger_v1.csv、patch_series_v2_ps10.csv
"""

from __future__ import annotations

import argparse
import math
import os
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

from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_utils_v2 import setup_plot_style


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Analyze local event-sensitive blast V3 variants.")
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
    config["paths"]["blast_hourly_csv"] = Path(config["paths"]["blast_hourly_csv"]).expanduser().resolve()
    return config


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """创建 blast V3 变体分析输出目录。"""

    base_dir = Path(config["paths"]["output_dir"]) / "blast_v3_variants"
    heatmap_dir = base_dir / "heatmaps"
    for path in [base_dir, heatmap_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": base_dir,
        "heatmap_dir": heatmap_dir,
    }


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """读取分析所需的输入数据。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    patch_meta = pd.read_csv(dataset_dir / "patch_meta_v2_ps10.csv")
    patch_series = pd.read_csv(dataset_dir / "patch_series_v2_ps10.csv", parse_dates=["timestamp"])
    current_ledger = pd.read_csv(dataset_dir / "blast_ledger_v3_input.csv", parse_dates=["timestamp"])
    proxy_ledger = pd.read_csv(dataset_dir / "proxy_blast_ledger_v1.csv", parse_dates=["timestamp"])
    current_saved = pd.read_csv(dataset_dir / "patch_blast_features_v3_ps10.csv", parse_dates=["timestamp"])
    proxy_saved = pd.read_csv(dataset_dir / "patch_blast_features_v3_ps10_proxy.csv", parse_dates=["timestamp"])
    blast_hourly = pd.read_csv(config["paths"]["blast_hourly_csv"], parse_dates=["timestamp"])
    return {
        "patch_meta": patch_meta,
        "patch_series": patch_series,
        "current_ledger": current_ledger,
        "proxy_ledger": proxy_ledger,
        "current_saved": current_saved,
        "proxy_saved": proxy_saved,
        "blast_hourly": blast_hourly,
    }


def set_patch_axes(ax: plt.Axes, patch_meta: pd.DataFrame, padding: float = 6.0) -> None:
    """把坐标轴锁定到 patch 的真实空间范围。"""

    x_min = float(patch_meta["cell_x_min"].min()) - padding
    x_max = float(patch_meta["cell_x_max"].max()) + padding
    y_min = float(patch_meta["cell_y_min"].min()) - padding
    y_max = float(patch_meta["cell_y_max"].max()) + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def pick_representative_hours(blast_hourly: pd.DataFrame, count: int) -> list[pd.Timestamp]:
    """挑选代表性的爆破小时。"""

    candidate = blast_hourly.loc[blast_hourly["blast_count"].fillna(0) > 0].copy()
    candidate = candidate.sort_values(
        ["total_charge_kg", "max_ppv_est_mm_s", "blast_count"],
        ascending=False,
    )
    return sorted(candidate["timestamp"].head(count).tolist())


def compute_base_sum_matrix(
    ledger: pd.DataFrame,
    patch_meta: pd.DataFrame,
    full_hours: pd.DatetimeIndex,
    tau_hours: float,
) -> np.ndarray:
    """按原始连续累积型公式计算 patch-hour 爆破影响矩阵。"""

    matrix = np.zeros((len(full_hours), len(patch_meta)), dtype=np.float32)
    patch_coords = patch_meta[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    hour_values = full_hours.to_numpy(dtype="datetime64[ns]")

    for event_row in ledger.itertuples(index=False):
        event_time = np.datetime64(pd.Timestamp(event_row.timestamp).to_datetime64())
        delta_hours = ((hour_values - event_time) / np.timedelta64(1, "h")).astype(np.float32)
        valid = delta_hours >= 0
        if not np.any(valid):
            continue

        distance = np.hypot(patch_coords[:, 0] - float(event_row.x_b), patch_coords[:, 1] - float(event_row.y_b))
        sigma = max(float(getattr(event_row, "sigma", getattr(event_row, "sigma_proxy", 35.0))), 1.0)
        spatial = float(event_row.Q) * np.exp(-np.square(distance) / (2.0 * sigma * sigma))
        time_decay = np.zeros(len(full_hours), dtype=np.float32)
        time_decay[valid] = np.exp(-delta_hours[valid] / tau_hours)
        matrix += np.outer(time_decay, spatial).astype(np.float32)

    return matrix


def compute_truncated_matrix(
    ledger: pd.DataFrame,
    patch_meta: pd.DataFrame,
    full_hours: pd.DatetimeIndex,
    tau_hours: float,
    spatial_cutoff: float,
    temporal_cutoff: float,
) -> np.ndarray:
    """计算带空间和时间双截断的累积影响矩阵。"""

    matrix = np.zeros((len(full_hours), len(patch_meta)), dtype=np.float32)
    patch_coords = patch_meta[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    hour_values = full_hours.to_numpy(dtype="datetime64[ns]")

    for event_row in ledger.itertuples(index=False):
        event_time = np.datetime64(pd.Timestamp(event_row.timestamp).to_datetime64())
        delta_hours = ((hour_values - event_time) / np.timedelta64(1, "h")).astype(np.float32)
        valid = (delta_hours >= 0) & (delta_hours <= temporal_cutoff)
        if not np.any(valid):
            continue

        distance = np.hypot(patch_coords[:, 0] - float(event_row.x_b), patch_coords[:, 1] - float(event_row.y_b))
        spatial_mask = distance <= spatial_cutoff
        if not np.any(spatial_mask):
            continue

        sigma = max(float(getattr(event_row, "sigma", getattr(event_row, "sigma_proxy", 35.0))), 1.0)
        spatial = np.zeros(len(patch_meta), dtype=np.float32)
        spatial[spatial_mask] = float(event_row.Q) * np.exp(-np.square(distance[spatial_mask]) / (2.0 * sigma * sigma))
        time_decay = np.zeros(len(full_hours), dtype=np.float32)
        time_decay[valid] = np.exp(-delta_hours[valid] / tau_hours)
        matrix += np.outer(time_decay, spatial).astype(np.float32)

    return matrix


def compute_max_local_matrix(
    ledger: pd.DataFrame,
    patch_meta: pd.DataFrame,
    full_hours: pd.DatetimeIndex,
    tau_hours: float,
) -> np.ndarray:
    """计算同一 patch-hour 只取最大单事件影响的矩阵。"""

    matrix = np.zeros((len(full_hours), len(patch_meta)), dtype=np.float32)
    patch_coords = patch_meta[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    hour_values = full_hours.to_numpy(dtype="datetime64[ns]")

    for event_row in ledger.itertuples(index=False):
        event_time = np.datetime64(pd.Timestamp(event_row.timestamp).to_datetime64())
        delta_hours = ((hour_values - event_time) / np.timedelta64(1, "h")).astype(np.float32)
        valid = delta_hours >= 0
        if not np.any(valid):
            continue

        distance = np.hypot(patch_coords[:, 0] - float(event_row.x_b), patch_coords[:, 1] - float(event_row.y_b))
        sigma = max(float(getattr(event_row, "sigma", getattr(event_row, "sigma_proxy", 35.0))), 1.0)
        spatial = float(event_row.Q) * np.exp(-np.square(distance) / (2.0 * sigma * sigma))
        time_decay = np.zeros(len(full_hours), dtype=np.float32)
        time_decay[valid] = np.exp(-delta_hours[valid] / tau_hours)
        contribution = np.outer(time_decay, spatial).astype(np.float32)
        matrix = np.maximum(matrix, contribution)

    return matrix


def compute_local_contrast_matrices(
    base_matrix: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 local contrast 的 median/IQR 和 zscore 版本。"""

    median = np.median(base_matrix, axis=1, keepdims=True)
    q25 = np.quantile(base_matrix, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(base_matrix, 0.75, axis=1, keepdims=True)
    iqr = q75 - q25
    median_iqr = (base_matrix - median) / (iqr + eps)

    mean = np.mean(base_matrix, axis=1, keepdims=True)
    std = np.std(base_matrix, axis=1, keepdims=True)
    zscore = (base_matrix - mean) / (std + eps)
    return median_iqr.astype(np.float32), zscore.astype(np.float32)


def matrix_to_frame(
    matrix: np.ndarray,
    full_hours: pd.DatetimeIndex,
    patch_meta: pd.DataFrame,
) -> pd.DataFrame:
    """把矩阵转换成 patch-hour 表。"""

    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(full_hours.to_numpy(), len(patch_meta)),
            "patch_id": np.tile(patch_meta["patch_id"].to_numpy(), len(full_hours)),
            "variant_value": matrix.reshape(-1).astype(np.float32),
        }
    )
    return frame.merge(
        patch_meta[["patch_id", "zone_id_v2", "zone_label_v2"]],
        on="patch_id",
        how="left",
        validate="many_to_one",
    )


def compute_lag_curve(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    max_lag: int,
) -> list[float]:
    """计算 blast 特征领先目标序列的滞后相关曲线。"""

    correlations = []
    for lag in range(max_lag + 1):
        left = feature_matrix[:-lag or None].reshape(-1)
        right = target_matrix[lag:].reshape(-1)
        mask = np.isfinite(left) & np.isfinite(right)
        if int(mask.sum()) < 10:
            correlations.append(np.nan)
            continue
        correlations.append(float(np.corrcoef(left[mask], right[mask])[0, 1]))
    return correlations


def summarize_variant(
    *,
    source: str,
    variant_name: str,
    variant_family: str,
    frame: pd.DataFrame,
    feature_matrix: np.ndarray,
    disp_max_matrix: np.ndarray,
    active_ratio_matrix: np.ndarray,
    blast_hours: pd.DatetimeIndex,
    max_lag: int,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """汇总单个变体的分布与相关性指标。"""

    values = frame["variant_value"].to_numpy(dtype=float)
    blast_values = frame.loc[frame["timestamp"].isin(blast_hours), "variant_value"].to_numpy(dtype=float)
    nonblast_values = frame.loc[~frame["timestamp"].isin(blast_hours), "variant_value"].to_numpy(dtype=float)

    disp_curve = compute_lag_curve(feature_matrix, disp_max_matrix, max_lag)
    active_curve = compute_lag_curve(feature_matrix, active_ratio_matrix, max_lag)
    disp_best_lag = int(np.nanargmax(np.abs(disp_curve))) if np.isfinite(disp_curve).any() else -1
    active_best_lag = int(np.nanargmax(np.abs(active_curve))) if np.isfinite(active_curve).any() else -1

    row = {
        "source": source,
        "variant_name": variant_name,
        "variant_family": variant_family,
        "row_count": int(len(values)),
        "nonzero_patch_ratio": float(np.mean(np.abs(values) > 1e-12)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "blast_hour_nonzero_ratio": float(np.mean(np.abs(blast_values) > 1e-12)) if len(blast_values) else np.nan,
        "nonblast_hour_nonzero_ratio": float(np.mean(np.abs(nonblast_values) > 1e-12)) if len(nonblast_values) else np.nan,
        "blast_hour_p50": float(np.quantile(blast_values, 0.50)) if len(blast_values) else np.nan,
        "blast_hour_p90": float(np.quantile(blast_values, 0.90)) if len(blast_values) else np.nan,
        "nonblast_hour_p50": float(np.quantile(nonblast_values, 0.50)) if len(nonblast_values) else np.nan,
        "nonblast_hour_p90": float(np.quantile(nonblast_values, 0.90)) if len(nonblast_values) else np.nan,
        "disp_max_best_lag": disp_best_lag,
        "disp_max_best_corr": float(disp_curve[disp_best_lag]) if disp_best_lag >= 0 else np.nan,
        "active_ratio_best_lag": active_best_lag,
        "active_ratio_best_corr": float(active_curve[active_best_lag]) if active_best_lag >= 0 else np.nan,
    }
    if extra_meta:
        row.update(extra_meta)
    return row


def plot_variant_heatmaps(
    *,
    frame: pd.DataFrame,
    patch_meta: pd.DataFrame,
    timestamps: list[pd.Timestamp],
    variant_label: str,
    out_dir: Path,
) -> list[Path]:
    """绘制单个变体的代表性时刻热力图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    signed_variant = (frame["variant_value"].min() < 0) and (frame["variant_value"].max() > 0)

    for timestamp in timestamps:
        current = frame.loc[frame["timestamp"] == timestamp, ["patch_id", "variant_value"]].copy()
        merged = patch_meta.merge(current, on="patch_id", how="left", validate="one_to_one")
        if merged["variant_value"].isna().all():
            continue

        for style_key, style_label in [("raw", "原始"), ("p99", "P99"), ("log1p", "log1p")]:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_facecolor("#EFEFEF")

            value_array = merged["variant_value"].fillna(0.0).to_numpy(dtype=float)
            if signed_variant:
                if style_key == "raw":
                    display_values = value_array.copy()
                elif style_key == "p99":
                    clip_value = float(np.quantile(np.abs(value_array), 0.99))
                    display_values = np.clip(value_array, -clip_value, clip_value)
                else:
                    display_values = np.sign(value_array) * np.log1p(np.abs(value_array))
                max_abs = max(float(np.max(np.abs(display_values))), 1e-9)
                norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
                cmap = plt.get_cmap("RdBu_r")
            else:
                positive_values = np.where(value_array > 0, value_array, np.nan)
                if style_key == "raw":
                    display_values = value_array.copy()
                    vmax = float(np.nanmax(positive_values)) if np.isfinite(positive_values).any() else 1.0
                elif style_key == "p99":
                    clip_value = float(np.nanquantile(positive_values, 0.99)) if np.isfinite(positive_values).any() else 1.0
                    display_values = np.clip(value_array, a_min=0.0, a_max=clip_value)
                    vmax = clip_value
                else:
                    display_values = np.log1p(np.clip(value_array, a_min=0.0, a_max=None))
                    vmax = float(np.nanmax(display_values)) if np.isfinite(display_values).any() else 1.0
                norm = plt.Normalize(vmin=0.0, vmax=max(vmax, 1e-9))
                cmap = plt.get_cmap("inferno")

            draw_frame = merged.copy()
            draw_frame["display_value"] = display_values
            if not signed_variant:
                draw_frame = draw_frame.loc[draw_frame["display_value"] > 0].copy()
            else:
                draw_frame = draw_frame.loc[np.abs(draw_frame["display_value"]) > 1e-12].copy()

            for row in draw_frame.itertuples(index=False):
                rect = Rectangle(
                    (row.cell_x_min, row.cell_y_min),
                    getattr(row, "cell_width", row.patch_size),
                    getattr(row, "cell_height", row.patch_size),
                    facecolor=cmap(norm(row.display_value)),
                    edgecolor="#777777",
                    linewidth=0.40,
                )
                ax.add_patch(rect)

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar_label = "variant_value"
            if signed_variant and style_key == "log1p":
                colorbar_label = "signed_log1p(variant_value)"
            plt.colorbar(sm, ax=ax, label=colorbar_label)
            ax.set_title(f"{variant_label} {style_label}：{timestamp:%Y-%m-%d %H:%M}")
            ax.set_xlabel("grid_x")
            ax.set_ylabel("grid_y")
            set_patch_axes(ax, patch_meta)

            output_path = out_dir / f"{variant_label}_{timestamp:%Y%m%d_%H%M}_{style_key}.png"
            fig.savefig(output_path, dpi=240)
            plt.close(fig)
            output_paths.append(output_path)

    return output_paths


def build_recommendation(compare: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按事件驱动特性给每个变体打分并选出推荐版本。"""

    candidates = compare.loc[~compare["variant_family"].isin(["reference"])].copy()
    target_ratio = float(config["blast_v3_variant_analysis"]["recommendation_nonzero_target"])
    candidates["event_separation_score"] = (
        (candidates["blast_hour_nonzero_ratio"] - candidates["nonblast_hour_nonzero_ratio"]).fillna(0.0)
        + np.log1p(candidates["blast_hour_p90"].fillna(0.0).clip(lower=0.0))
        - np.log1p(candidates["nonblast_hour_p90"].fillna(0.0).abs())
    )
    candidates["sparsity_score"] = -np.abs(candidates["nonblast_hour_nonzero_ratio"] - target_ratio)
    candidates["correlation_score"] = (
        candidates["disp_max_best_corr"].abs().fillna(0.0) + candidates["active_ratio_best_corr"].abs().fillna(0.0)
    )
    candidates["total_score"] = (
        1.8 * candidates["event_separation_score"]
        + 1.0 * candidates["sparsity_score"]
        + 1.4 * candidates["correlation_score"]
    )

    aggregated = (
        candidates.groupby(["variant_name", "variant_family"], as_index=False)
        .agg(
            source_count=("source", "nunique"),
            total_score_mean=("total_score", "mean"),
            nonzero_patch_ratio_mean=("nonzero_patch_ratio", "mean"),
            nonblast_hour_nonzero_ratio_mean=("nonblast_hour_nonzero_ratio", "mean"),
            blast_hour_p90_mean=("blast_hour_p90", "mean"),
            nonblast_hour_p90_mean=("nonblast_hour_p90", "mean"),
            disp_max_best_corr_mean=("disp_max_best_corr", lambda values: float(np.nanmean(np.abs(values)))),
            active_ratio_best_corr_mean=("active_ratio_best_corr", lambda values: float(np.nanmean(np.abs(values)))),
        )
        .sort_values("total_score_mean", ascending=False)
        .reset_index(drop=True)
    )
    best_row = aggregated.iloc[0].to_dict()
    return aggregated, best_row


def write_report(
    *,
    compare: pd.DataFrame,
    recommendation_table: pd.DataFrame,
    best_row: dict[str, Any],
    representative_hours: list[pd.Timestamp],
    heatmap_dir: Path,
    out_path: Path,
) -> None:
    """输出 blast V3 变体诊断报告。"""

    table_lines = [
        "| source | variant_name | variant_family | row_count | nonzero_patch_ratio | p50 | p90 | p99 | max | blast_hour_nonzero_ratio | nonblast_hour_nonzero_ratio | blast_hour_p90 | nonblast_hour_p90 | disp_max_best_lag | disp_max_best_corr | active_ratio_best_lag | active_ratio_best_corr |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in compare.itertuples(index=False):
        table_lines.append(
            f"| {row.source} | {row.variant_name} | {row.variant_family} | {int(row.row_count)} | {float(row.nonzero_patch_ratio):.6f} | {float(row.p50):.6f} | {float(row.p90):.6f} | {float(row.p99):.6f} | {float(row.max):.6f} | {float(row.blast_hour_nonzero_ratio):.6f} | {float(row.nonblast_hour_nonzero_ratio):.6f} | {float(row.blast_hour_p90):.6f} | {float(row.nonblast_hour_p90):.6f} | {int(row.disp_max_best_lag)} | {float(row.disp_max_best_corr):.6f} | {int(row.active_ratio_best_lag)} | {float(row.active_ratio_best_corr):.6f} |"
        )

    lines = [
        "# blast_v3_variant_diagnostic_report",
        "",
        "## 结论",
        "- `current_v3 / proxy_v3` 原始连续累积型特征在绝大多数 patch-hour 上都非零，确实不适合作为直接事件驱动输入。",
        f"- 综合稀疏性、blast/non-blast 分离度和与 `disp_max / active_ratio` 的 lag-correlation，本轮推荐版本是 `{best_row['variant_name']}`。",
        f"- 推荐热力图目录：`{heatmap_dir}`。",
        "",
        "## 代表性时刻",
        *[f"- {timestamp:%Y-%m-%d %H:%M}" for timestamp in representative_hours],
        "",
        "## 推荐版本摘要",
        f"- 版本：`{best_row['variant_name']}`",
        f"- 平均总分：{float(best_row['total_score_mean']):.6f}",
        f"- 非 blast 小时非零比例均值：{float(best_row['nonblast_hour_nonzero_ratio_mean']):.6f}",
        f"- blast 小时 p90 均值：{float(best_row['blast_hour_p90_mean']):.6f}",
        f"- `disp_max` 最优相关绝对值均值：{float(best_row['disp_max_best_corr_mean']):.6f}",
        f"- `active_ratio` 最优相关绝对值均值：{float(best_row['active_ratio_best_corr_mean']):.6f}",
        "",
        "## 评分前十",
    ]

    for row in recommendation_table.head(10).itertuples(index=False):
        lines.append(
            f"- {row.variant_name}：score={float(row.total_score_mean):.6f}，nonblast_nonzero={float(row.nonblast_hour_nonzero_ratio_mean):.6f}，blast_p90={float(row.blast_hour_p90_mean):.6f}"
        )

    lines.extend(["", "## 详细对比表", "", *table_lines, ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """执行 blast V3 局部事件敏感变体分析。"""

    args = parse_args()
    config = load_config(args.config)
    setup_plot_style()
    outputs = ensure_output_dirs(config)
    inputs = load_inputs(config)

    patch_meta = inputs["patch_meta"]
    patch_series = inputs["patch_series"]
    blast_hourly = inputs["blast_hourly"]
    full_hours = pd.DatetimeIndex(sorted(patch_series["timestamp"].drop_duplicates()))
    representative_hours = pick_representative_hours(
        blast_hourly,
        int(config["blast_v3_variant_analysis"]["representative_heatmap_count"]),
    )
    blast_hours = pd.DatetimeIndex(
        sorted(blast_hourly.loc[blast_hourly["blast_count"].fillna(0) > 0, "timestamp"].drop_duplicates())
    )

    disp_max_matrix = (
        patch_series.pivot(index="timestamp", columns="patch_id", values="disp_max")
        .reindex(index=full_hours, columns=patch_meta["patch_id"])
        .to_numpy(dtype=np.float32)
    )
    active_ratio_matrix = (
        patch_series.pivot(index="timestamp", columns="patch_id", values="active_ratio")
        .reindex(index=full_hours, columns=patch_meta["patch_id"])
        .to_numpy(dtype=np.float32)
    )

    tau_main = float(config["processing"]["blast_decay_tau_hours"])
    eps = float(config["blast_v3_variant_analysis"]["contrast_eps"])
    max_lag = int(config["blast_v3_variant_analysis"]["max_lag_hours"])

    ledgers = {
        "current": inputs["current_ledger"],
        "proxy": inputs["proxy_ledger"],
    }
    saved_reference = {
        "current": inputs["current_saved"],
        "proxy": inputs["proxy_saved"],
    }

    compare_rows: list[dict[str, Any]] = []
    heatmap_root = outputs["heatmap_dir"]

    for source_name, ledger in ledgers.items():
        source_heatmap_root = heatmap_root / source_name
        source_heatmap_root.mkdir(parents=True, exist_ok=True)

        reference_frame = saved_reference[source_name].rename(columns={"blast_patch_decay": "variant_value"}).copy()
        reference_matrix = (
            reference_frame.pivot(index="timestamp", columns="patch_id", values="variant_value")
            .reindex(index=full_hours, columns=patch_meta["patch_id"])
            .to_numpy(dtype=np.float32)
        )
        compare_rows.append(
            summarize_variant(
                source=source_name,
                variant_name="V3_reference",
                variant_family="reference",
                frame=reference_frame[["timestamp", "patch_id", "variant_value", "zone_id_v2", "zone_label_v2"]],
                feature_matrix=reference_matrix,
                disp_max_matrix=disp_max_matrix,
                active_ratio_matrix=active_ratio_matrix,
                blast_hours=blast_hours,
                max_lag=max_lag,
            )
        )
        plot_variant_heatmaps(
            frame=reference_frame[["timestamp", "patch_id", "variant_value", "zone_id_v2", "zone_label_v2"]],
            patch_meta=patch_meta,
            timestamps=representative_hours,
            variant_label="V3_reference",
            out_dir=source_heatmap_root / "V3_reference",
        )

        base_sum_matrix = compute_base_sum_matrix(ledger, patch_meta, full_hours, tau_main)

        for spatial_cutoff in config["blast_v3_variant_analysis"]["spatial_cutoffs"]:
            for temporal_cutoff in config["blast_v3_variant_analysis"]["temporal_cutoffs"]:
                matrix = compute_truncated_matrix(
                    ledger=ledger,
                    patch_meta=patch_meta,
                    full_hours=full_hours,
                    tau_hours=tau_main,
                    spatial_cutoff=float(spatial_cutoff),
                    temporal_cutoff=float(temporal_cutoff),
                )
                variant_name = f"V3a_truncated_Rc{int(spatial_cutoff)}_Hc{int(temporal_cutoff)}"
                frame = matrix_to_frame(matrix, full_hours, patch_meta)
                compare_rows.append(
                    summarize_variant(
                        source=source_name,
                        variant_name=variant_name,
                        variant_family="V3a_truncated",
                        frame=frame,
                        feature_matrix=matrix,
                        disp_max_matrix=disp_max_matrix,
                        active_ratio_matrix=active_ratio_matrix,
                        blast_hours=blast_hours,
                        max_lag=max_lag,
                        extra_meta={"spatial_cutoff": float(spatial_cutoff), "temporal_cutoff": float(temporal_cutoff)},
                    )
                )
                plot_variant_heatmaps(
                    frame=frame,
                    patch_meta=patch_meta,
                    timestamps=representative_hours,
                    variant_label=variant_name,
                    out_dir=source_heatmap_root / variant_name,
                )

        max_local_matrix = compute_max_local_matrix(ledger, patch_meta, full_hours, tau_main)
        max_local_frame = matrix_to_frame(max_local_matrix, full_hours, patch_meta)
        compare_rows.append(
            summarize_variant(
                source=source_name,
                variant_name="V3b_max_local",
                variant_family="V3b_max_local",
                frame=max_local_frame,
                feature_matrix=max_local_matrix,
                disp_max_matrix=disp_max_matrix,
                active_ratio_matrix=active_ratio_matrix,
                blast_hours=blast_hours,
                max_lag=max_lag,
            )
        )
        plot_variant_heatmaps(
            frame=max_local_frame,
            patch_meta=patch_meta,
            timestamps=representative_hours,
            variant_label="V3b_max_local",
            out_dir=source_heatmap_root / "V3b_max_local",
        )

        median_iqr_matrix, zscore_matrix = compute_local_contrast_matrices(base_sum_matrix, eps)
        median_iqr_frame = matrix_to_frame(median_iqr_matrix, full_hours, patch_meta)
        zscore_frame = matrix_to_frame(zscore_matrix, full_hours, patch_meta)
        compare_rows.append(
            summarize_variant(
                source=source_name,
                variant_name="V3c_local_contrast_median_iqr",
                variant_family="V3c_local_contrast",
                frame=median_iqr_frame,
                feature_matrix=median_iqr_matrix,
                disp_max_matrix=disp_max_matrix,
                active_ratio_matrix=active_ratio_matrix,
                blast_hours=blast_hours,
                max_lag=max_lag,
                extra_meta={"normalization": "median_iqr"},
            )
        )
        compare_rows.append(
            summarize_variant(
                source=source_name,
                variant_name="V3c_local_contrast_zscore",
                variant_family="V3c_local_contrast",
                frame=zscore_frame,
                feature_matrix=zscore_matrix,
                disp_max_matrix=disp_max_matrix,
                active_ratio_matrix=active_ratio_matrix,
                blast_hours=blast_hours,
                max_lag=max_lag,
                extra_meta={"normalization": "zscore"},
            )
        )
        plot_variant_heatmaps(
            frame=median_iqr_frame,
            patch_meta=patch_meta,
            timestamps=representative_hours,
            variant_label="V3c_local_contrast_median_iqr",
            out_dir=source_heatmap_root / "V3c_local_contrast_median_iqr",
        )
        plot_variant_heatmaps(
            frame=zscore_frame,
            patch_meta=patch_meta,
            timestamps=representative_hours,
            variant_label="V3c_local_contrast_zscore",
            out_dir=source_heatmap_root / "V3c_local_contrast_zscore",
        )

    compare = pd.DataFrame(compare_rows).sort_values(["source", "variant_family", "variant_name"]).reset_index(drop=True)
    compare.to_csv(outputs["base_dir"] / "blast_v3_variant_compare.csv", index=False)

    recommendation_table, best_row = build_recommendation(compare, config)
    recommendation_table.to_csv(outputs["base_dir"] / "blast_v3_variant_recommendation_table.csv", index=False)
    write_report(
        compare=compare,
        recommendation_table=recommendation_table,
        best_row=best_row,
        representative_hours=representative_hours,
        heatmap_dir=heatmap_root,
        out_path=outputs["base_dir"] / "blast_v3_variant_diagnostic_report.md",
    )

    print(outputs["base_dir"] / "blast_v3_variant_compare.csv")
    print(outputs["base_dir"] / "blast_v3_variant_diagnostic_report.md")
    print(outputs["base_dir"] / "blast_v3_variant_recommendation_table.csv")
    print(heatmap_root)


if __name__ == "__main__":
    main()
