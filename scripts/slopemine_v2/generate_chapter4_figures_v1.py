#!/usr/bin/env python3
"""生成第四章后半部分论文图件。
相关文件：results_main_all.csv、results_subsets_all.csv、c2_vs_lstm_event_windows.md、patch_tensor_base_v2_ps10.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from data_provider.slopemine_formal import (  # noqa: E402
        build_formal_window_manifest,
        build_sample_subset_masks,
        compute_subset_hour_flags,
        load_formal_window_bundle,
        load_split_plan,
    )
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class FrozenFormalWindowBundle:
        timestamps: np.ndarray
        patch_ids: np.ndarray
        weather_feature_names: list[str]
        blast_v2_feature_names: list[str]
        blast_v3_feature_names: list[str]
        weather: np.ndarray
        blast_v2: np.ndarray
        blast_v3: np.ndarray
        target: np.ndarray
        global_missing: np.ndarray


    def _as_string_list(values: np.ndarray) -> list[str]:
        return [str(item) for item in values.tolist()]


    def load_formal_window_bundle(
        dataset_dir: Path,
        *,
        missing_hour_policy: str = "drop_global_missing",
    ) -> tuple[FrozenFormalWindowBundle, pd.DataFrame, dict[str, Any]]:
        raw_bundle = np.load(dataset_dir / "patch_tensor_base_v2_ps10.npz", allow_pickle=True)
        patch_meta = pd.read_csv(dataset_dir / "patch_meta_v2_ps10_bg_excluded.csv")
        patch_ids = raw_bundle["patch_ids"].astype(str)
        patch_meta = patch_meta.set_index("patch_id").reindex(patch_ids).reset_index()
        if patch_meta["patch_id"].isna().any():
            raise ValueError("patch_meta_v2_ps10_bg_excluded.csv 与 patch_tensor_base_v2_ps10.npz 的 patch 顺序无法对齐。")

        if missing_hour_policy != "drop_global_missing":
            raise ValueError(f"暂不支持的 missing_hour_policy: {missing_hour_policy}")

        timestamps_all = pd.to_datetime(raw_bundle["timestamps"].astype(str)).to_numpy()
        global_missing = raw_bundle["global_missing"].astype(np.uint8)
        keep_mask = global_missing == 0
        dropped_timestamps = pd.to_datetime(timestamps_all[~keep_mask])
        bundle = FrozenFormalWindowBundle(
            timestamps=timestamps_all[keep_mask],
            patch_ids=patch_ids,
            weather_feature_names=_as_string_list(raw_bundle["weather_feature_names"]),
            blast_v2_feature_names=_as_string_list(raw_bundle["blast_v2_feature_names"]),
            blast_v3_feature_names=_as_string_list(raw_bundle["blast_v3_feature_names"]),
            weather=raw_bundle["weather"][keep_mask].astype(np.float32),
            blast_v2=raw_bundle["blast_v2"][keep_mask].astype(np.float32),
            blast_v3=raw_bundle["blast_v3"][keep_mask].astype(np.float32),
            target=raw_bundle["target"][keep_mask].astype(np.float32),
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
        split_plan = pd.read_csv(split_path, parse_dates=["start_timestamp", "end_timestamp"])
        split_plan = split_plan.sort_values("effective_start_index").reset_index(drop=True)
        return split_plan


    def build_formal_window_manifest(
        timestamps: np.ndarray,
        split_plan: pd.DataFrame,
        *,
        seq_len: int,
        pred_len: int,
        missing_hour_policy: str,
    ) -> pd.DataFrame:
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
        for split_row in split_plan.itertuples(index=False):
            split_name = str(split_row.split_name)
            split_start = int(split_row.effective_start_index)
            split_end = int(split_row.effective_end_index_exclusive)
            mask = (decoder_start >= split_start) & (decoder_end_exclusive <= split_end)
            for sample_id in np.where(mask)[0].tolist():
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

        return pd.DataFrame(rows).sort_values(["split_name", "sample_id"]).reset_index(drop=True)


    def compute_subset_hour_flags(bundle: FrozenFormalWindowBundle) -> dict[str, np.ndarray]:
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
        decoder_start = manifest["decoder_start_index"].to_numpy(dtype=np.int32)
        decoder_indices = decoder_start[:, None] + np.arange(pred_len, dtype=np.int32)[None, :]
        return {
            subset_name: subset_hour_flags[subset_name][decoder_indices].astype(np.uint8)
            for subset_name in ["blast_hours", "rain_hours", "non_event_hours"]
        }

from plot_utils_v2 import set_patch_axes, setup_plot_style  # noqa: E402


MODEL_LABELS = {
    "baseline_raw_timefilter_a4_96": "原始 TimeFilter\n(原始骨干)",
    "C1": "C1\n(PGGC)",
    "C2": "C2 main\n(PGGC+EDDR)",
    "C3": "C3 main\n(PGGC+EDDR+PIR)",
    "baseline_lstm_a4_96": "LSTM\n(最强参考基线)",
}

MODEL_COLORS = {
    "baseline_raw_timefilter_a4_96": "#9AA5B1",
    "C1": "#4D908E",
    "C2": "#E76F51",
    "C3": "#577590",
    "baseline_lstm_a4_96": "#2A9D8F",
}

SUBSET_ORDER = ["blast_hours", "rain_hours", "non_event_hours"]
SUBSET_LABELS = {
    "blast_hours": "爆破时段",
    "rain_hours": "降雨时段",
    "non_event_hours": "非事件时段",
}

PHYSICAL_CURVE_COLORS = {
    "truth": "#111111",
    "c2": "#2C7FB8",
    "c3": "#D73027",
    "blast": "#2F2F2F",
}
SPATIAL_CONTEXT_COLOR = "#D6D6D6"
SPATIAL_CMAP = "YlOrRd"
THESIS_PATCH_BY_SAMPLE = {
    532: "ps10_v2_p155",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chapter 4 result figures from frozen slopemine outputs.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root.",
    )
    parser.add_argument(
        "--physical-sample-id",
        type=int,
        default=532,
        help="Representative sample id for the thesis-facing physical response curve figure.",
    )
    parser.add_argument(
        "--spatial-model",
        type=str,
        default="C3",
        choices=["A2", "A3", "A4", "C1", "C2", "C3"],
        help="Experiment id whose test patch MAE is used in the thesis-facing spatial heatmap.",
    )
    parser.add_argument(
        "--include-unmatched-context",
        dest="include_unmatched_context",
        action="store_true",
        help="Render unmatched edge/background points as grey context in the spatial heatmap.",
    )
    parser.add_argument(
        "--exclude-unmatched-context",
        dest="include_unmatched_context",
        action="store_false",
        help="Hide unmatched edge/background points from the spatial heatmap.",
    )
    parser.set_defaults(include_unmatched_context=True)
    return parser.parse_args()


def ensure_output_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_artifact_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    text = str(path)
    remote_prefix = "/root/autodl-tmp/Time-Series-Library/"
    if text.startswith(remote_prefix):
        local_path = repo_root / text[len(remote_prefix):]
        if local_path.exists():
            return local_path
    return path


def save_dual(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def compute_metrics(target: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask > 0
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "mse": float("nan")}
    error = (pred - target) * mask
    abs_error = np.abs(error)
    denom = float(mask.sum())
    mse = float((error**2).sum() / denom)
    return {
        "mae": float(abs_error.sum() / denom),
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
    }


def build_context(repo_root: Path) -> dict[str, Any]:
    ms_root = repo_root / "outputs" / "slopemine_v2" / "ms_timefilter_v1"
    output_root = repo_root / "outputs" / "slopemine_v2" / "chapter4_figures_v1"
    ensure_output_root(output_root)

    dataset_dir = repo_root / "dataset" / "slopemine_v2"
    split_plan = load_split_plan(repo_root / "outputs" / "slopemine_v2" / "experiment_plan_v1" / "experiment_split_plan_v1.csv")
    bundle, patch_meta, missing_report = load_formal_window_bundle(dataset_dir, missing_hour_policy="drop_global_missing")
    manifest_path = dataset_dir / "window_manifest_ms_timefilter_ps10_seq96_pred12.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(
            manifest_path,
            parse_dates=[
                "encoder_start_timestamp",
                "encoder_end_timestamp",
                "decoder_start_timestamp",
                "decoder_end_timestamp",
            ],
        )
    else:
        manifest = build_formal_window_manifest(
            bundle.timestamps,
            split_plan,
            seq_len=96,
            pred_len=12,
            missing_hour_policy="drop_global_missing",
        )
    subset_flags = compute_subset_hour_flags(bundle)
    subset_masks = build_sample_subset_masks(manifest, subset_flags, pred_len=12)
    point_meta = pd.read_csv(dataset_dir / "point_meta_v2.csv")
    zone_rule = json.loads((dataset_dir / "engineering_zone_polygons_v2.json").read_text(encoding="utf-8"))

    return {
        "repo_root": repo_root,
        "ms_root": ms_root,
        "output_root": output_root,
        "dataset_dir": dataset_dir,
        "split_plan": split_plan,
        "bundle": bundle,
        "patch_meta": patch_meta,
        "manifest": manifest,
        "subset_flags": subset_flags,
        "subset_masks": subset_masks,
        "point_meta": point_meta,
        "zone_rule": zone_rule,
        "missing_report": missing_report,
    }


def read_results(ms_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_main = pd.read_csv(ms_root / "results_main_all.csv")
    results_subsets = pd.read_csv(ms_root / "results_subsets_all.csv")
    return results_main, results_subsets


def get_main_row(results_main: pd.DataFrame, experiment_id: str, branch_name: str) -> pd.Series:
    return results_main.loc[
        (results_main["experiment_id"] == experiment_id) & (results_main["branch_name"] == branch_name)
    ].iloc[0]


def get_subset_value(
    results_subsets: pd.DataFrame,
    experiment_id: str,
    branch_name: str,
    split_name: str,
    subset_name: str,
    metric: str = "mae",
) -> float:
    row = results_subsets.loc[
        (results_subsets["experiment_id"] == experiment_id)
        & (results_subsets["branch_name"] == branch_name)
        & (results_subsets["split"] == split_name)
        & (results_subsets["subset"] == subset_name)
    ].iloc[0]
    return float(row[metric])


def fig_main_performance_compare(
    results_main: pd.DataFrame,
    results_subsets: pd.DataFrame,
    output_root: Path,
) -> list[dict[str, str]]:
    order = [
        ("baseline_raw_timefilter_a4_96", "reference"),
        ("C1", "main"),
        ("C2", "main"),
        ("C3", "main"),
        ("baseline_lstm_a4_96", "reference"),
    ]
    labels = [MODEL_LABELS[key] for key, _ in order]
    test_mae = [float(get_main_row(results_main, key, branch)["test_mae"]) for key, branch in order]
    blast_mae = [get_subset_value(results_subsets, key, branch, "test", "blast_hours") for key, branch in order]
    colors = [MODEL_COLORS[key] for key, _ in order]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), squeeze=False)
    axes = axes[0]
    for axis, values, title in [
        (axes[0], test_mae, "总体测试误差"),
        (axes[1], blast_mae, "爆破时段误差"),
    ]:
        x = np.arange(len(labels))
        bars = axis.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel("MAE")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for idx, bar in enumerate(bars):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{values[idx]:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        c2_idx = [key for key, _ in order].index("C2")
        bars[c2_idx].set_linewidth(2.0)
        bars[c2_idx].set_edgecolor("#8C1C13")

    fig.suptitle("主模型与参考模型总体性能对比", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = output_root / "fig_ch4_main_performance_compare.png"
    pdf_path = output_root / "fig_ch4_main_performance_compare.pdf"
    save_dual(fig, png_path, pdf_path)
    return [{
        "filename": png_path.name + " / " + pdf_path.name,
        "data_source": "results_main_all.csv + results_subsets_all.csv",
        "figure_meaning": "比较原始 TimeFilter、C1、C2、C3 与 LSTM 在 test_MAE 与 blast_test_MAE 上的总体表现，突出 C2 main 作为当前主模型。",
        "section": "第四章 4.4 主结果对比",
    }]


def fig_subset_compare(
    results_subsets: pd.DataFrame,
    output_root: Path,
) -> list[dict[str, str]]:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), squeeze=False)
    axes = axes[0]
    for axis, split_name in zip(axes, ["val", "test"]):
        x = np.arange(len(SUBSET_ORDER))
        width = 0.34
        c2_values = [get_subset_value(results_subsets, "C2", "main", split_name, subset) for subset in SUBSET_ORDER]
        lstm_values = [get_subset_value(results_subsets, "baseline_lstm_a4_96", "reference", split_name, subset) for subset in SUBSET_ORDER]

        bars_c2 = axis.bar(x - width / 2, c2_values, width=width, color=MODEL_COLORS["C2"], label="C2 main", edgecolor="#333333", linewidth=0.8)
        bars_lstm = axis.bar(x + width / 2, lstm_values, width=width, color=MODEL_COLORS["baseline_lstm_a4_96"], label="LSTM", edgecolor="#333333", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels([SUBSET_LABELS[name] for name in SUBSET_ORDER])
        axis.set_ylabel("MAE")
        axis.set_title(f"{split_name.upper()} 子集误差")
        axis.grid(axis="y", alpha=0.25)

        for idx, (c2_value, lstm_value) in enumerate(zip(c2_values, lstm_values)):
            delta = c2_value - lstm_value
            axis.text(
                x[idx],
                max(c2_value, lstm_value),
                f"Δ={delta:+.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#8C1C13" if delta > 0 else "#1B7F3B",
            )
            axis.text(
                bars_c2[idx].get_x() + bars_c2[idx].get_width() / 2,
                c2_value,
                f"{c2_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            axis.text(
                bars_lstm[idx].get_x() + bars_lstm[idx].get_width() / 2,
                lstm_value,
                f"{lstm_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.text(
        0.5,
        0.01,
        "注：C2 并未全面优于 LSTM，但 PGGC+EDDR 为复杂工况提供了结构扩展空间。",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    png_path = output_root / "fig_ch4_subset_compare_c2_vs_lstm.png"
    pdf_path = output_root / "fig_ch4_subset_compare_c2_vs_lstm.pdf"
    save_dual(fig, png_path, pdf_path)
    return [{
        "filename": png_path.name + " / " + pdf_path.name,
        "data_source": "results_subsets_all.csv + c2_vs_lstm_subset_compare.csv",
        "figure_meaning": "比较 C2 与 LSTM 在 val/test 的爆破、降雨、非事件子集上的 MAE，并标注差值。",
        "section": "第四章 4.4 子集误差分析",
    }]


def parse_event_windows(md_path: Path) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$"
    )
    for line in md_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        rows.append(
            {
                "sample_id": int(match.group(1)),
                "patch_id": str(match.group(2)).strip(),
                "decoder_start": pd.Timestamp(match.group(3).strip()),
                "decoder_end": pd.Timestamp(match.group(4).strip()),
                "blast_score": float(match.group(5)),
            }
        )
    return rows


def event_patch_index(bundle, patch_id: str) -> int:
    patch_ids = bundle.patch_ids.astype(str).tolist()
    return int(patch_ids.index(str(patch_id)))


def sample_lookup(bundle_npz: dict[str, np.ndarray]) -> dict[int, int]:
    return {int(sample_id): idx for idx, sample_id in enumerate(bundle_npz["sample_ids"].astype(int).tolist())}


def compute_smoothness(prediction: np.ndarray) -> float:
    finite = np.asarray(prediction, dtype=float)[np.isfinite(prediction)]
    if finite.size < 3:
        return float("nan")
    return float(np.mean(np.abs(np.diff(finite, n=2))))


def set_point_axes(ax: plt.Axes, point_meta: pd.DataFrame, padding: float = 8.0) -> None:
    x_min = float(point_meta["grid_x"].min()) - padding
    x_max = float(point_meta["grid_x"].max()) + padding
    y_min = float(point_meta["grid_y"].min()) - padding
    y_max = float(point_meta["grid_y"].max()) + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def get_prediction_bundle_for_result(
    *,
    context: dict[str, Any],
    results_main: pd.DataFrame,
    experiment_id: str,
    branch_name: str,
) -> dict[str, np.ndarray]:
    row = get_main_row(results_main, experiment_id, branch_name)
    return load_prediction_bundle(resolve_artifact_path(context["repo_root"], row["prediction_test_path"]))


def strongest_blast_patch_idx(context: dict[str, Any], decoder_slice: slice) -> int:
    return int(np.argmax(context["bundle"].blast_v3[decoder_slice, :, 0].sum(axis=0)))


def build_point_patch_mapping(context: dict[str, Any]) -> pd.DataFrame:
    cached = context.get("_point_patch_mapping")
    if cached is not None:
        return cached.copy()

    point_meta = context["point_meta"]
    patch_meta = context["patch_meta"][["patch_id", "zone_id_v2", "cell_x_min", "cell_x_max", "cell_y_min", "cell_y_max"]]
    merged = point_meta.merge(patch_meta, on="zone_id_v2", how="left")
    matched = merged.loc[
        (merged["grid_x"] >= merged["cell_x_min"])
        & (merged["grid_x"] <= merged["cell_x_max"])
        & (merged["grid_y"] >= merged["cell_y_min"])
        & (merged["grid_y"] <= merged["cell_y_max"])
    ].copy()

    match_counts = matched.groupby("point_id").size()
    multi_match = match_counts.loc[match_counts > 1]
    if not multi_match.empty:
        raise RuntimeError(f"发现 {len(multi_match)} 个点位命中了多个 patch，无法构造唯一散点热图。")

    matched_unique = matched.drop_duplicates("point_id").loc[:, ["point_id", "patch_id"]]
    point_frame = point_meta.merge(matched_unique, on="point_id", how="left")
    point_frame["matched_flag"] = point_frame["patch_id"].notna().astype(np.uint8)

    matched_count = int(point_frame["matched_flag"].sum())
    unmatched_count = int((point_frame["matched_flag"] == 0).sum())
    context["_point_patch_mapping"] = point_frame.copy()
    context["_point_patch_stats"] = {
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "multi_match_count": int(len(multi_match)),
    }
    return point_frame


def plot_event_axis(
    axis: plt.Axes,
    *,
    history_timestamps: pd.DatetimeIndex,
    history: np.ndarray,
    future_timestamps: pd.DatetimeIndex,
    truth: np.ndarray,
    c2_pred: np.ndarray,
    lstm_pred: np.ndarray,
    blast_mask: np.ndarray,
    title: str,
) -> None:
    axis.plot(history_timestamps, history, color="#7F8C8D", linewidth=1.9, label="历史真值")
    axis.plot(future_timestamps, truth, color="#1D3557", linewidth=2.2, marker="o", markersize=3.8, label="Ground Truth")
    axis.plot(future_timestamps, c2_pred, color=MODEL_COLORS["C2"], linewidth=2.0, marker="s", markersize=2.8, label="C2")
    axis.plot(
        future_timestamps,
        lstm_pred,
        color=MODEL_COLORS["baseline_lstm_a4_96"],
        linewidth=2.0,
        marker="^",
        markersize=2.8,
        label="LSTM",
    )
    axis.axvline(history_timestamps[-1], color="#B0B0B0", linestyle="--", linewidth=1.0)
    for ts in future_timestamps[np.asarray(blast_mask, dtype=bool)]:
        axis.axvline(pd.Timestamp(ts), color="#D62828", linestyle=":", linewidth=1.2, alpha=0.85)
    axis.set_title(title)
    axis.set_ylabel("disp_target")
    axis.grid(alpha=0.25)
    axis.tick_params(axis="x", rotation=20)


def build_stitched_prediction(
    *,
    bundle_npz: dict[str, np.ndarray],
    lookup: dict[int, int],
    manifest_test: pd.DataFrame,
    anchor_position: int,
    patch_idx: int,
    anchor_decoder_start: int,
    display_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_len = int(bundle_npz["predictions"].shape[1])
    stitched = np.full(display_len, np.nan, dtype=float)
    filled = np.zeros(display_len, dtype=bool)

    for row in manifest_test.iloc[anchor_position:].itertuples(index=False):
        sample_id = int(row.sample_id)
        decoder_start_index = int(row.decoder_start_index)
        if decoder_start_index >= anchor_decoder_start + display_len:
            break
        sample_pos = lookup.get(sample_id)
        if sample_pos is None:
            continue
        sample_pred = bundle_npz["predictions"][sample_pos, :, patch_idx]
        sample_truth = bundle_npz["targets"][sample_pos, :, patch_idx]
        for step in range(pred_len):
            offset = decoder_start_index + step - anchor_decoder_start
            if offset < 0:
                continue
            if offset >= display_len:
                break
            if not filled[offset]:
                stitched[offset] = float(sample_pred[step])
                filled[offset] = True

    if not np.any(filled):
        raise RuntimeError("未能基于冻结窗口拼接出连续预测序列。")
    last_filled = int(np.where(filled)[0].max()) + 1
    return stitched[:last_filled], filled[:last_filled]


def build_event_plot_series(
    *,
    context: dict[str, Any],
    manifest_test: pd.DataFrame,
    row: dict[str, Any],
    c2_bundle: dict[str, np.ndarray],
    lstm_bundle: dict[str, np.ndarray],
    c2_lookup: dict[int, int],
    lstm_lookup: dict[int, int],
    display_len: int,
) -> dict[str, Any]:
    timestamps = pd.to_datetime(context["bundle"].timestamps)
    sample_id = int(row["sample_id"])
    patch_id = str(row["patch_id"])
    patch_idx = event_patch_index(context["bundle"], patch_id)
    anchor_position = int(manifest_test.index[manifest_test["sample_id"] == sample_id][0])
    manifest_row = manifest_test.iloc[anchor_position]
    encoder_slice = slice(int(manifest_row["encoder_start_index"]), int(manifest_row["encoder_end_index_exclusive"]))
    anchor_decoder_start = int(manifest_row["decoder_start_index"])
    max_len = min(display_len, len(timestamps) - anchor_decoder_start)

    c2_pred, c2_filled = build_stitched_prediction(
        bundle_npz=c2_bundle,
        lookup=c2_lookup,
        manifest_test=manifest_test,
        anchor_position=anchor_position,
        patch_idx=patch_idx,
        anchor_decoder_start=anchor_decoder_start,
        display_len=max_len,
    )
    lstm_pred, lstm_filled = build_stitched_prediction(
        bundle_npz=lstm_bundle,
        lookup=lstm_lookup,
        manifest_test=manifest_test,
        anchor_position=anchor_position,
        patch_idx=patch_idx,
        anchor_decoder_start=anchor_decoder_start,
        display_len=max_len,
    )
    usable_len = min(len(c2_pred), len(lstm_pred), int(np.where(c2_filled & lstm_filled)[0].max()) + 1)
    future_indices = np.arange(anchor_decoder_start, anchor_decoder_start + usable_len, dtype=int)

    return {
        "sample_id": sample_id,
        "patch_id": patch_id,
        "patch_idx": patch_idx,
        "history_timestamps": timestamps[encoder_slice],
        "history": context["bundle"].target[encoder_slice, patch_idx].astype(float),
        "future_timestamps": timestamps[future_indices],
        "truth": context["bundle"].target[future_indices, patch_idx].astype(float),
        "c2_pred": c2_pred[:usable_len],
        "lstm_pred": lstm_pred[:usable_len],
        "blast_mask": context["subset_flags"]["blast_hours"][future_indices].astype(bool),
        "blast_score": float(row["blast_score"]),
    }


def render_event_windows_figure(
    *,
    output_root: Path,
    event_series: list[dict[str, Any]],
    stem: str,
    future_len_label: str,
    note_text: str | None = None,
) -> None:
    fig, axes = plt.subplots(len(event_series), 1, figsize=(14.2, 5.1 * len(event_series)), squeeze=False)
    axes = axes[:, 0]
    for axis, series in zip(axes, event_series):
        title = (
            f"sample={series['sample_id']}, patch={series['patch_id']}, {future_len_label}, "
            f"{pd.Timestamp(series['future_timestamps'][0]).strftime('%Y-%m-%d %H:%M')} ~ "
            f"{pd.Timestamp(series['future_timestamps'][-1]).strftime('%Y-%m-%d %H:%M')}"
        )
        plot_event_axis(
            axis,
            history_timestamps=series["history_timestamps"],
            history=series["history"],
            future_timestamps=series["future_timestamps"],
            truth=series["truth"],
            c2_pred=series["c2_pred"],
            lstm_pred=series["lstm_pred"],
            blast_mask=series["blast_mask"],
            title=title,
        )
    axes[-1].set_xlabel("时间")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    if note_text:
        fig.text(0.5, 0.01, note_text, ha="center", fontsize=10)
        rect = [0, 0.04, 1, 0.95]
    else:
        rect = [0, 0, 1, 0.95]
    fig.tight_layout(rect=rect)
    png_path = output_root / f"{stem}.png"
    pdf_path = output_root / f"{stem}.pdf"
    save_dual(fig, png_path, pdf_path)


def fig_event_windows(
    *,
    context: dict[str, Any],
    results_main: pd.DataFrame,
) -> list[dict[str, str]]:
    md_path = context["ms_root"] / "c2_vs_lstm_event_windows.md"
    event_rows = parse_event_windows(md_path)
    if len(event_rows) < 2:
        raise FileNotFoundError(f"无法从 {md_path} 解析出 2 个事件窗口。")

    c2_row = get_main_row(results_main, "C2", "main")
    lstm_row = get_main_row(results_main, "baseline_lstm_a4_96", "reference")
    c2_bundle = load_prediction_bundle(resolve_artifact_path(context["repo_root"], c2_row["prediction_test_path"]))
    lstm_bundle = load_prediction_bundle(resolve_artifact_path(context["repo_root"], lstm_row["prediction_test_path"]))
    c2_lookup = sample_lookup(c2_bundle)
    lstm_lookup = sample_lookup(lstm_bundle)
    manifest_test = context["manifest"].loc[context["manifest"]["split_name"] == "test"].sort_values("sample_id").reset_index(drop=True)
    selected_rows = event_rows[:2]

    short_series = [
        build_event_plot_series(
            context=context,
            manifest_test=manifest_test,
            row=row,
            c2_bundle=c2_bundle,
            lstm_bundle=lstm_bundle,
            c2_lookup=c2_lookup,
            lstm_lookup=lstm_lookup,
            display_len=12,
        )
        for row in selected_rows
    ]
    render_event_windows_figure(
        output_root=context["output_root"],
        event_series=short_series,
        stem="fig_ch4_event_windows_c2_vs_lstm_short12h",
        future_len_label="12h 单窗预测",
    )

    long_series = [
        build_event_plot_series(
            context=context,
            manifest_test=manifest_test,
            row=row,
            c2_bundle=c2_bundle,
            lstm_bundle=lstm_bundle,
            c2_lookup=c2_lookup,
            lstm_lookup=lstm_lookup,
            display_len=48,
        )
        for row in selected_rows
    ]
    render_event_windows_figure(
        output_root=context["output_root"],
        event_series=long_series,
        stem="fig_ch4_event_windows_c2_vs_lstm",
        future_len_label="48h 连续拼接展示",
        note_text="注：C2 并未整体超过 LSTM；该图用于展示更长时段内的响应形态与误差演化；连续曲线由冻结的 12-step 预测结果按“最早窗口预测”规则拼接得到。",
    )
    return [
        {
            "filename": "fig_ch4_event_windows_c2_vs_lstm.png / fig_ch4_event_windows_c2_vs_lstm.pdf",
            "data_source": "c2_vs_lstm_event_windows.md + prediction bundles + formal 96/12 manifest",
            "figure_meaning": "展示两个典型爆破事件在 48h 有效时间步上的连续预测曲线；重叠时刻采用“最早窗口预测”规则拼接，并标记 blast event 时刻。",
            "section": "第四章 4.5 典型事件窗口分析",
        },
        {
            "filename": "fig_ch4_event_windows_c2_vs_lstm_short12h.png / fig_ch4_event_windows_c2_vs_lstm_short12h.pdf",
            "data_source": "c2_vs_lstm_event_windows.md + prediction bundles + formal 96/12 manifest",
            "figure_meaning": "保留原始 12h 单窗事件预测图，作为第四章长预测主图的备份版本。",
            "section": "附录 / 论文图件备份",
        },
    ]


def draw_patch_rectangles(
    ax: plt.Axes,
    patch_meta: pd.DataFrame,
    values: np.ndarray,
    *,
    norm: Normalize,
    cmap: str,
    title: str,
) -> None:
    ax.set_title(title)
    ax.set_facecolor("#F2F2F2")
    for row, value in zip(patch_meta.itertuples(index=False), values.tolist()):
        if not np.isfinite(value):
            continue
        rect = Rectangle(
            (float(row.cell_x_min), float(row.cell_y_min)),
            float(row.cell_x_max - row.cell_x_min),
            float(row.cell_y_max - row.cell_y_min),
            facecolor=plt.get_cmap(cmap)(norm(value)),
            edgecolor="#FFFFFF",
            linewidth=0.25,
        )
        ax.add_patch(rect)
    set_patch_axes(ax, patch_meta)
    ax.grid(False)
    ax.set_xlabel("cell_x")
    ax.set_ylabel("cell_y")


def fig_patch_mae_heatmaps(
    *,
    context: dict[str, Any],
    results_main: pd.DataFrame,
) -> list[dict[str, str]]:
    c2_row = get_main_row(results_main, "C2", "main")
    lstm_row = get_main_row(results_main, "baseline_lstm_a4_96", "reference")
    c2_bundle = load_prediction_bundle(resolve_artifact_path(context["repo_root"], c2_row["prediction_test_path"]))
    lstm_bundle = load_prediction_bundle(resolve_artifact_path(context["repo_root"], lstm_row["prediction_test_path"]))

    c2_patch_mae = c2_bundle["patch_mae"]
    lstm_patch_mae = lstm_bundle["patch_mae"]
    diff = c2_patch_mae - lstm_patch_mae
    finite = np.concatenate([c2_patch_mae[np.isfinite(c2_patch_mae)], lstm_patch_mae[np.isfinite(lstm_patch_mae)]])
    vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    common_norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-8))
    diff_abs = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 99)) if np.isfinite(diff).any() else 1.0
    diff_norm = Normalize(vmin=-max(diff_abs, 1e-8), vmax=max(diff_abs, 1e-8))

    outputs = []
    figures = [
        ("fig_ch4_patch_mae_heatmap_c2", c2_patch_mae, common_norm, "YlOrRd", "C2 main patch 级 test MAE"),
        ("fig_ch4_patch_mae_heatmap_lstm", lstm_patch_mae, common_norm, "YlOrRd", "LSTM patch 级 test MAE"),
        ("fig_ch4_patch_mae_heatmap_diff_c2_minus_lstm", diff, diff_norm, "RdBu_r", "C2 - LSTM patch 级 MAE 差值"),
    ]
    for stem, values, norm, cmap, title in figures:
        fig, ax = plt.subplots(figsize=(7.8, 6.4))
        draw_patch_rectangles(ax, context["patch_meta"], values, norm=norm, cmap=cmap, title=title)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("MAE")
        fig.tight_layout()
        png_path = context["output_root"] / f"{stem}.png"
        pdf_path = context["output_root"] / f"{stem}.pdf"
        save_dual(fig, png_path, pdf_path)
        outputs.append({
            "filename": png_path.name + " / " + pdf_path.name,
            "data_source": "prediction bundles + patch_meta_v2_ps10_bg_excluded.csv",
            "figure_meaning": title + "。背景区未参与主训练，因此图中保持浅灰空白背景。",
            "section": "第四章 4.5 patch 级空间误差分析",
        })
    return outputs


def fig_pggc_graph_compare(output_root: Path, ms_root: Path) -> list[dict[str, str]]:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.8), squeeze=False)
    axes = axes[0]
    image_paths = [
        ("空间/时间先验图 A_prior", ms_root / "figures" / "graph_A_prior_C1.png"),
        ("学习图 A_learned", ms_root / "figures" / "graph_A_learned_C1.png"),
        ("融合图 A_final", ms_root / "figures" / "graph_A_final_C1.png"),
    ]
    for axis, (title, path) in zip(axes, image_paths):
        axis.imshow(mpimg.imread(path))
        axis.set_title(title)
        axis.axis("off")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = output_root / "fig_ch4_pggc_graph_compare.png"
    pdf_path = output_root / "fig_ch4_pggc_graph_compare.pdf"
    save_dual(fig, png_path, pdf_path)
    return [{
        "filename": png_path.name + " / " + pdf_path.name,
        "data_source": "graph_A_prior_C1.png + graph_A_learned_C1.png + graph_A_final_C1.png",
        "figure_meaning": "三联图展示 PGGC 的先验图、学习图与融合图；单图渲染时对正值边权额外做了 `+0.5` 显示偏移，以增强稀疏结构可见性。",
        "section": "第四章 4.6 PGGC 结构解释",
    }]


def fig_eddr_routing_compare(
    *,
    context: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    missing = []
    val_path = context["ms_root"] / "evaluations" / "C2_main_val_eval.npz"
    test_path = context["ms_root"] / "evaluations" / "C2_main_test_eval.npz"
    if not val_path.exists() or not test_path.exists():
        missing.append("缺少 C2_main_val_eval.npz 或 C2_main_test_eval.npz，无法恢复 EDDR 专家权重。")
        return [], missing

    val_eval = np.load(val_path, allow_pickle=True)
    test_eval = np.load(test_path, allow_pickle=True)
    if "gate_mean" not in val_eval.files or "gate_mean" not in test_eval.files:
        missing.append("评估文件中不含 gate_mean，无法生成 EDDR 路由分析图。")
        return [], missing

    split_frames = []
    for split_name, eval_data in [("val", val_eval), ("test", test_eval)]:
        split_manifest = context["manifest"].loc[context["manifest"]["split_name"] == split_name].sort_values("sample_id").reset_index(drop=True)
        split_mask = context["manifest"]["split_name"].to_numpy() == split_name
        split_subset_masks = {name: values[split_mask] for name, values in context["subset_masks"].items()}
        gate_mean = eval_data["gate_mean"]
        if gate_mean.shape[0] != len(split_manifest):
            missing.append(f"{split_name} 阶段 gate_mean 行数与 manifest 不匹配。")
            return [], missing
        for subset_name in ["blast_hours", "non_event_hours"]:
            sample_mask = split_subset_masks[subset_name].sum(axis=1) > 0
            subset_gate = gate_mean[sample_mask]
            for expert_idx, expert_name in enumerate(["Spatial", "Temporal", "Spatiotemporal"]):
                for value in subset_gate[:, expert_idx].tolist():
                    split_frames.append(
                        {
                            "split": split_name,
                            "subset": subset_name,
                            "expert": expert_name,
                            "weight": float(value),
                        }
                    )
    routing_df = pd.DataFrame(split_frames)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), squeeze=False)
    axes = axes[0]
    expert_order = ["Spatial", "Temporal", "Spatiotemporal"]
    colors = {"blast_hours": "#E76F51", "non_event_hours": "#577590"}
    for axis, split_name in zip(axes, ["val", "test"]):
        split_df = routing_df.loc[routing_df["split"] == split_name].copy()
        positions = np.arange(len(expert_order)) * 3.0
        for offset, subset_name in [(-0.4, "blast_hours"), (0.4, "non_event_hours")]:
            data = [
                split_df.loc[(split_df["subset"] == subset_name) & (split_df["expert"] == expert), "weight"].to_numpy(dtype=float)
                for expert in expert_order
            ]
            bp = axis.boxplot(
                data,
                positions=positions + offset,
                widths=0.65,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#222222", "linewidth": 1.2},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[subset_name])
                patch.set_alpha(0.65)
                patch.set_edgecolor("#333333")
        axis.set_xticks(positions)
        axis.set_xticklabels(expert_order)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Gate weight")
        axis.set_title(f"{split_name.upper()}：blast vs non-event")
        axis.grid(axis="y", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=colors["blast_hours"], lw=8, alpha=0.65),
        plt.Line2D([0], [0], color=colors["non_event_hours"], lw=8, alpha=0.65),
    ]
    fig.legend(handles, ["blast_hours", "non_event_hours"], loc="upper center", ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png_path = context["output_root"] / "fig_ch4_eddr_routing_compare.png"
    pdf_path = context["output_root"] / "fig_ch4_eddr_routing_compare.pdf"
    save_dual(fig, png_path, pdf_path)
    return ([{
        "filename": png_path.name + " / " + pdf_path.name,
        "data_source": "C2_main_val_eval.npz + C2_main_test_eval.npz (gate_mean)",
        "figure_meaning": "比较 blast_hours 与 non_event_hours 下三类专家权重分布，用于检查 EDDR 是否发生了事件驱动路由变化。",
        "section": "第四章 4.6 EDDR 路由分析",
    }], missing)


def choose_smoothness_window(
    *,
    context: dict[str, Any],
    c2_bundle: dict[str, np.ndarray],
    raw_bundle: dict[str, np.ndarray],
    lstm_bundle: dict[str, np.ndarray],
) -> dict[str, Any]:
    manifest_test = context["manifest"].loc[context["manifest"]["split_name"] == "test"].sort_values("sample_id").reset_index(drop=True)
    split_mask = context["manifest"]["split_name"].to_numpy() == "test"
    blast_mask = context["subset_masks"]["blast_hours"][split_mask].sum(axis=1) > 0
    c2_lookup = sample_lookup(c2_bundle)
    raw_lookup = sample_lookup(raw_bundle)
    lstm_lookup = sample_lookup(lstm_bundle)

    ranking = []
    for row in manifest_test.loc[blast_mask].itertuples(index=False):
        sample_id = int(row.sample_id)
        if sample_id not in c2_lookup or sample_id not in raw_lookup or sample_id not in lstm_lookup:
            continue
        decoder_slice = slice(int(row.decoder_start_index), int(row.decoder_end_index_exclusive))
        patch_idx = int(np.argmax(context["subset_flags"]["blast_strength"][decoder_slice]))
        # Use patch with strongest blast_v3 response if available.
        patch_idx = int(np.argmax(context["bundle"].blast_v3[decoder_slice, :, 0].sum(axis=0)))

        c2_metric = compute_metrics(
            c2_bundle["targets"][c2_lookup[sample_id], :, patch_idx],
            c2_bundle["predictions"][c2_lookup[sample_id], :, patch_idx],
            c2_bundle["masks"][c2_lookup[sample_id], :, patch_idx],
        )
        raw_metric = compute_metrics(
            raw_bundle["targets"][raw_lookup[sample_id], :, patch_idx],
            raw_bundle["predictions"][raw_lookup[sample_id], :, patch_idx],
            raw_bundle["masks"][raw_lookup[sample_id], :, patch_idx],
        )
        lstm_metric = compute_metrics(
            lstm_bundle["targets"][lstm_lookup[sample_id], :, patch_idx],
            lstm_bundle["predictions"][lstm_lookup[sample_id], :, patch_idx],
            lstm_bundle["masks"][lstm_lookup[sample_id], :, patch_idx],
        )
        blast_score = float(context["subset_flags"]["blast_strength"][decoder_slice].sum())
        ranking.append(
            {
                "sample_id": sample_id,
                "patch_idx": patch_idx,
                "decoder_start_index": int(row.decoder_start_index),
                "decoder_end_index_exclusive": int(row.decoder_end_index_exclusive),
                "encoder_start_index": int(row.encoder_start_index),
                "encoder_end_index_exclusive": int(row.encoder_end_index_exclusive),
                "blast_score": blast_score,
                "c2_minus_raw": float(c2_metric["mae"] - raw_metric["mae"]),
                "c2_minus_lstm": float(c2_metric["mae"] - lstm_metric["mae"]),
            }
        )
    ranking_df = pd.DataFrame(ranking)
    if ranking_df.empty:
        raise RuntimeError("没有可用的 blast 窗口来生成平滑性对比图。")
    ranking_df = ranking_df.sort_values(["c2_minus_raw", "blast_score"], ascending=[True, False]).reset_index(drop=True)
    return ranking_df.iloc[0].to_dict()


def fig_prediction_smoothness(
    *,
    context: dict[str, Any],
    results_main: pd.DataFrame,
) -> list[dict[str, str]]:
    raw_row = get_main_row(results_main, "baseline_raw_timefilter_a4_96", "reference")
    c2_row = get_main_row(results_main, "C2", "main")
    lstm_row = get_main_row(results_main, "baseline_lstm_a4_96", "reference")
    raw_bundle = load_prediction_bundle(resolve_artifact_path(context["repo_root"], raw_row["prediction_test_path"]))
    c2_bundle = load_prediction_bundle(resolve_artifact_path(context["repo_root"], c2_row["prediction_test_path"]))
    lstm_bundle = load_prediction_bundle(resolve_artifact_path(context["repo_root"], lstm_row["prediction_test_path"]))
    choice = choose_smoothness_window(context=context, c2_bundle=c2_bundle, raw_bundle=raw_bundle, lstm_bundle=lstm_bundle)

    sample_id = int(choice["sample_id"])
    patch_idx = int(choice["patch_idx"])
    c2_lookup = sample_lookup(c2_bundle)
    raw_lookup = sample_lookup(raw_bundle)
    lstm_lookup = sample_lookup(lstm_bundle)
    timestamps = pd.to_datetime(context["bundle"].timestamps)

    encoder_slice = slice(int(choice["encoder_start_index"]), int(choice["encoder_end_index_exclusive"]))
    decoder_slice = slice(int(choice["decoder_start_index"]), int(choice["decoder_end_index_exclusive"]))
    history = context["bundle"].target[encoder_slice, patch_idx]
    truth = c2_bundle["targets"][c2_lookup[sample_id], :, patch_idx]
    raw_pred = raw_bundle["predictions"][raw_lookup[sample_id], :, patch_idx]
    c2_pred = c2_bundle["predictions"][c2_lookup[sample_id], :, patch_idx]
    lstm_pred = lstm_bundle["predictions"][lstm_lookup[sample_id], :, patch_idx]

    blast_hours = context["subset_flags"]["blast_hours"][decoder_slice].astype(bool)
    blast_positions = np.where(blast_hours)[0]
    if blast_positions.size:
        zoom_start = max(0, int(blast_positions[0]) - 1)
        zoom_end = min(len(truth), int(blast_positions[-1]) + 3)
    else:
        zoom_start, zoom_end = 0, min(len(truth), 6)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), squeeze=False)
    axes = axes[0]
    for axis in axes:
        axis.plot(timestamps[encoder_slice], history, color="#7F8C8D", linewidth=1.9, label="历史真值")
        axis.plot(timestamps[decoder_slice], truth, color="#1D3557", linewidth=2.2, marker="o", markersize=4, label="Ground Truth")
        axis.plot(timestamps[decoder_slice], raw_pred, color=MODEL_COLORS["baseline_raw_timefilter_a4_96"], linewidth=1.8, label="raw TimeFilter")
        axis.plot(timestamps[decoder_slice], c2_pred, color=MODEL_COLORS["C2"], linewidth=2.0, label="C2")
        axis.plot(timestamps[decoder_slice], lstm_pred, color=MODEL_COLORS["baseline_lstm_a4_96"], linewidth=1.8, label="LSTM")
        axis.axvline(timestamps[encoder_slice][-1], color="#B0B0B0", linestyle="--", linewidth=1.0)
        for ts in timestamps[decoder_slice][blast_hours]:
            axis.axvline(pd.Timestamp(ts), color="#D62828", linestyle=":", linewidth=1.2, alpha=0.85)
        axis.grid(alpha=0.25)
        axis.set_ylabel("disp_target")

    axes[0].set_title(f"代表窗口全程：sample={sample_id}, patch={context['bundle'].patch_ids[patch_idx]}")
    axes[0].set_xlabel("时间")

    axes[1].set_title("局部放大：突发工况响应")
    zoom_timestamps = timestamps[decoder_slice][zoom_start:zoom_end]
    axes[1].set_xlim(zoom_timestamps[0], zoom_timestamps[-1])
    y_values = np.concatenate([
        truth[zoom_start:zoom_end],
        raw_pred[zoom_start:zoom_end],
        c2_pred[zoom_start:zoom_end],
        lstm_pred[zoom_start:zoom_end],
    ])
    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    pad = max(1e-4, (y_max - y_min) * 0.15)
    axes[1].set_ylim(y_min - pad, y_max + pad)
    axes[1].set_xlabel("时间")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = context["output_root"] / "fig_ch4_prediction_smoothness_compare.png"
    pdf_path = context["output_root"] / "fig_ch4_prediction_smoothness_compare.pdf"
    save_dual(fig, png_path, pdf_path)
    return [{
        "filename": png_path.name + " / " + pdf_path.name,
        "data_source": "baseline_raw_timefilter_a4_96 + C2 main + baseline_lstm_a4_96 prediction bundles",
        "figure_meaning": "比较 raw TimeFilter、C2 与 LSTM 在代表窗口中的平滑性与突发工况响应，用于说明 C2 相比原始骨干更合理。",
        "section": "第四章 4.6 物理合理性辅助分析",
    }]


def fig_physical_curve_comparison(
    *,
    context: dict[str, Any],
    results_main: pd.DataFrame,
    sample_id: int,
) -> list[dict[str, str]]:
    manifest_test = context["manifest"].loc[context["manifest"]["split_name"] == "test"].sort_values("sample_id").reset_index(drop=True)
    sample_rows = manifest_test.loc[manifest_test["sample_id"] == sample_id]
    if sample_rows.empty:
        raise RuntimeError(f"sample_id={sample_id} 不在冻结后的 test manifest 中。")

    raw_bundle = get_prediction_bundle_for_result(
        context=context,
        results_main=results_main,
        experiment_id="baseline_raw_timefilter_a4_96",
        branch_name="reference",
    )
    c2_bundle = get_prediction_bundle_for_result(
        context=context,
        results_main=results_main,
        experiment_id="C2",
        branch_name="main",
    )
    c3_bundle = get_prediction_bundle_for_result(
        context=context,
        results_main=results_main,
        experiment_id="C3",
        branch_name="main",
    )
    raw_lookup = sample_lookup(raw_bundle)
    c2_lookup = sample_lookup(c2_bundle)
    c3_lookup = sample_lookup(c3_bundle)
    if sample_id not in raw_lookup or sample_id not in c2_lookup or sample_id not in c3_lookup:
        raise RuntimeError("物理曲线主样本没有同时出现在 raw/C2/C3 prediction bundle 中。")

    manifest_row = sample_rows.iloc[0]
    decoder_slice = slice(int(manifest_row["decoder_start_index"]), int(manifest_row["decoder_end_index_exclusive"]))
    patch_idx = strongest_blast_patch_idx(context, decoder_slice)
    patch_id = str(context["bundle"].patch_ids[patch_idx])
    expected_patch_id = THESIS_PATCH_BY_SAMPLE.get(sample_id)
    if expected_patch_id is not None and patch_id != expected_patch_id:
        raise RuntimeError(f"sample_id={sample_id} 的主 patch 应为 {expected_patch_id}，实际为 {patch_id}。")

    future_timestamps = pd.to_datetime(context["bundle"].timestamps[decoder_slice])
    truth = c3_bundle["targets"][c3_lookup[sample_id], :, patch_idx].astype(float)
    raw_pred = raw_bundle["predictions"][raw_lookup[sample_id], :, patch_idx].astype(float)
    c2_pred = c2_bundle["predictions"][c2_lookup[sample_id], :, patch_idx].astype(float)
    c3_pred = c3_bundle["predictions"][c3_lookup[sample_id], :, patch_idx].astype(float)
    raw_mask = raw_bundle["masks"][raw_lookup[sample_id], :, patch_idx].astype(float)
    c2_mask = c2_bundle["masks"][c2_lookup[sample_id], :, patch_idx].astype(float)
    c3_mask = c3_bundle["masks"][c3_lookup[sample_id], :, patch_idx].astype(float)
    blast_mask = context["subset_flags"]["blast_hours"][decoder_slice].astype(bool)
    if not np.any(blast_mask):
        raise RuntimeError("物理曲线主样本的 decoder horizon 中没有 blast 标记。")

    raw_mae = float(compute_metrics(truth, raw_pred, raw_mask)["mae"])
    c2_mae = float(compute_metrics(truth, c2_pred, c2_mask)["mae"])
    c3_mae = float(compute_metrics(truth, c3_pred, c3_mask)["mae"])
    if not (raw_mae > c2_mae > c3_mae):
        raise RuntimeError(
            f"主样本未满足 raw > C2 > C3：raw={raw_mae:.6f}, C2={c2_mae:.6f}, C3={c3_mae:.6f}。"
        )

    spike_idx = int(np.argmax(truth))
    if spike_idx < len(truth) - 1:
        recovery_start_idx = spike_idx + 1
    else:
        recovery_start_idx = min(int(np.where(blast_mask)[0].max()) + 1, len(truth) - 1)
    recovery_slice = slice(recovery_start_idx, len(truth))
    recovery_timestamps = future_timestamps[recovery_slice]
    c2_recovery_mae = float(compute_metrics(truth[recovery_slice], c2_pred[recovery_slice], c2_mask[recovery_slice])["mae"])
    c3_recovery_mae = float(compute_metrics(truth[recovery_slice], c3_pred[recovery_slice], c3_mask[recovery_slice])["mae"])
    c2_recovery_smooth = compute_smoothness(c2_pred[recovery_slice])
    c3_recovery_smooth = compute_smoothness(c3_pred[recovery_slice])
    if not c3_recovery_mae < c2_recovery_mae:
        raise RuntimeError(
            f"恢复段未满足 C3 MAE < C2 MAE：C2={c2_recovery_mae:.6f}, C3={c3_recovery_mae:.6f}。"
        )
    if np.isfinite(c2_recovery_smooth) and np.isfinite(c3_recovery_smooth) and not c3_recovery_smooth < c2_recovery_smooth:
        raise RuntimeError(
            f"恢复段未满足 C3 更平稳：C2={c2_recovery_smooth:.6f}, C3={c3_recovery_smooth:.6f}。"
        )

    trace_frame = pd.DataFrame(
        {
            "timestamp": future_timestamps,
            "truth": truth,
            "c2": c2_pred,
            "c3": c3_pred,
            "blast_flag": blast_mask.astype(np.uint8),
            "sample_id": sample_id,
            "patch_id": patch_id,
        }
    )
    trace_path = context["output_root"] / "fig_physical_curve_comparison_trace.csv"
    trace_frame.to_csv(trace_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), squeeze=False)
    axes = axes[0]
    for axis in axes:
        axis.plot(
            future_timestamps,
            truth,
            color=PHYSICAL_CURVE_COLORS["truth"],
            linewidth=2.2,
            marker="o",
            markersize=3.6,
            label="真实位移",
        )
        axis.plot(
            future_timestamps,
            c2_pred,
            color=PHYSICAL_CURVE_COLORS["c2"],
            linewidth=2.1,
            marker="s",
            markersize=3.0,
            label="无PIR预测 (C2)",
        )
        axis.plot(
            future_timestamps,
            c3_pred,
            color=PHYSICAL_CURVE_COLORS["c3"],
            linewidth=2.1,
            marker="^",
            markersize=3.2,
            label="完整模型预测 (C3+PIR)",
        )
        for idx, ts in enumerate(future_timestamps[blast_mask]):
            axis.axvline(
                pd.Timestamp(ts),
                color=PHYSICAL_CURVE_COLORS["blast"],
                linestyle=(0, (3, 3)),
                linewidth=1.0,
                alpha=0.9,
                label="blast 时刻" if idx == 0 else None,
            )
        axis.grid(alpha=0.25)
        axis.set_ylabel("disp_target")
        axis.tick_params(axis="x", rotation=20)

    axes[0].set_title(f"完整 12h 爆后响应窗口: sample={sample_id}, patch={patch_id}")
    axes[0].set_xlabel("时间")
    main_y = np.concatenate([truth, c2_pred, c3_pred])
    main_pad = max(1e-4, (float(np.nanmax(main_y)) - float(np.nanmin(main_y))) * 0.06)
    axes[0].set_ylim(float(np.nanmin(main_y)) - main_pad, float(np.nanmax(main_y)) + main_pad)

    axes[1].set_title(
        f"恢复段放大: {pd.Timestamp(recovery_timestamps[0]).strftime('%m-%d %H:%M')} ~ "
        f"{pd.Timestamp(recovery_timestamps[-1]).strftime('%m-%d %H:%M')}"
    )
    axes[1].set_xlim(recovery_timestamps[0], recovery_timestamps[-1])
    zoom_y = np.concatenate([truth[recovery_slice], c2_pred[recovery_slice], c3_pred[recovery_slice]])
    zoom_pad = max(1e-4, (float(np.nanmax(zoom_y)) - float(np.nanmin(zoom_y))) * 0.18)
    axes[1].set_ylim(float(np.nanmin(zoom_y)) - zoom_pad, float(np.nanmax(zoom_y)) + zoom_pad)
    axes[1].set_xlabel("时间")
    axes[1].text(
        0.02,
        0.98,
        f"C2恢复MAE={c2_recovery_mae:.4f}\nC3恢复MAE={c3_recovery_mae:.4f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.25", "alpha": 0.92},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.text(
        0.5,
        0.01,
        "注：该图为代表性局部窗口，强调 PIR 对爆后响应稳定性的可视化差异，不代表 C3 在整体统计上优于 C2。",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    png_path = context["output_root"] / "fig_physical_curve_comparison.png"
    pdf_path = context["output_root"] / "fig_physical_curve_comparison.pdf"
    save_dual(fig, png_path, pdf_path)
    return [{
        "filename": png_path.name + " / " + pdf_path.name,
        "data_source": "baseline_raw_timefilter_a4_96 + C2 main + C3 main prediction bundles + window_manifest_ms_timefilter_ps10_seq96_pred12.csv",
        "figure_meaning": "展示 C2 与 C3(PIR) 在代表性爆后响应窗口中的 12h 单窗预测与恢复段放大，用于强调 PIR 对爆后响应稳定性的局部可视化作用。",
        "section": "第四章 4.6 PIR 爆后响应示意（fig:physical_curve_comparison）",
    }]


def load_patch_mae_from_eval(context: dict[str, Any], spatial_model: str) -> tuple[np.ndarray, Path]:
    eval_path = context["ms_root"] / "evaluations" / f"{spatial_model}_main_test_eval.npz"
    if not eval_path.exists():
        raise FileNotFoundError(f"缺少空间热图所需评估文件：{eval_path}")
    eval_bundle = np.load(eval_path, allow_pickle=True)
    if "patch_mae" not in eval_bundle.files:
        raise KeyError(f"{eval_path} 不含 patch_mae。")
    patch_mae = eval_bundle["patch_mae"].astype(float)
    if patch_mae.shape[0] != len(context["patch_meta"]):
        raise RuntimeError(
            f"{spatial_model} patch_mae 长度={patch_mae.shape[0]}，与 patch_meta 长度={len(context['patch_meta'])} 不一致。"
        )
    return patch_mae, eval_path


def draw_zone_boundaries(ax: plt.Axes, zone_rule: dict[str, Any]) -> None:
    sorted_zones = sorted(zone_rule["zones"], key=lambda row: int(row["priority_rank"]))
    for zone_row in sorted_zones:
        polygon = zone_row["polygon"]
        x_values = [vertex[0] for vertex in polygon] + [polygon[0][0]]
        y_values = [vertex[1] for vertex in polygon] + [polygon[0][1]]
        ax.plot(
            x_values,
            y_values,
            color="#111111",
            linewidth=1.65,
            linestyle=(0, (5, 3)),
            alpha=0.96,
            zorder=4,
        )


def fig_spatial_error_heatmap(
    *,
    context: dict[str, Any],
    spatial_model: str,
    include_unmatched_context: bool,
) -> list[dict[str, str]]:
    patch_mae, eval_path = load_patch_mae_from_eval(context, spatial_model)
    point_frame = build_point_patch_mapping(context)
    mae_lookup = dict(zip(context["patch_meta"]["patch_id"].tolist(), patch_mae.tolist()))
    point_frame["mae"] = point_frame["patch_id"].map(mae_lookup)

    matched_stats = context["_point_patch_stats"]
    if matched_stats["matched_count"] != 18958 or matched_stats["unmatched_count"] != 608:
        raise RuntimeError(
            f"点位映射计数异常：matched={matched_stats['matched_count']}, unmatched={matched_stats['unmatched_count']}。"
        )
    if matched_stats["multi_match_count"] != 0:
        raise RuntimeError(f"点位映射存在 {matched_stats['multi_match_count']} 个多重命中。")
    if point_frame.loc[point_frame["matched_flag"] == 1, "mae"].isna().any():
        raise RuntimeError("存在已匹配点位没有继承到 patch MAE。")

    point_output = context["output_root"] / "fig_spatial_error_heatmap_points.csv"
    point_frame.loc[:, ["point_id", "grid_x", "grid_y", "zone_id_v2", "patch_id", "mae", "matched_flag"]].to_csv(point_output, index=False)

    patch_frame = context["patch_meta"].copy()
    patch_frame["mae"] = patch_mae
    top_zone_counts = patch_frame.nlargest(30, "mae").groupby("zone_name_v2").size().sort_values(ascending=False)
    if top_zone_counts.empty or top_zone_counts.index[0] != "坡顶区":
        raise RuntimeError("高误差 patch 的主集中区不再是坡顶区，无法维持论文图的 PIR 主线叙事。")

    matched_points = point_frame.loc[point_frame["matched_flag"] == 1].copy()
    unmatched_points = point_frame.loc[point_frame["matched_flag"] == 0].copy()
    vmax = float(np.nanpercentile(matched_points["mae"].to_numpy(dtype=float), 97))
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-8))

    fig, ax = plt.subplots(figsize=(11.6, 8.8))
    ax.set_facecolor("white")
    if include_unmatched_context:
        ax.scatter(
            unmatched_points["grid_x"],
            unmatched_points["grid_y"],
            s=9,
            c=SPATIAL_CONTEXT_COLOR,
            alpha=0.58,
            linewidths=0.0,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )

    scatter = ax.scatter(
        matched_points["grid_x"],
        matched_points["grid_y"],
        c=matched_points["mae"],
        cmap=SPATIAL_CMAP,
        norm=norm,
        s=10,
        alpha=0.97,
        linewidths=0.0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    draw_zone_boundaries(ax, context["zone_rule"])
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Patch-derived test MAE")

    legend_handles = [
        Line2D([0], [0], color="#111111", linewidth=1.65, linestyle=(0, (5, 3)), label="冻结工程分区边界"),
    ]
    if include_unmatched_context:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=SPATIAL_CONTEXT_COLOR,
                markeredgecolor="none",
                markersize=7,
                alpha=0.8,
                label="边缘/背景上下文点",
            )
        )
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax.set_title(f"PIR 主线空间误差分布图 ({spatial_model} main, test MAE)")
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.grid(False)
    set_point_axes(ax, context["point_meta"])
    fig.text(
        0.5,
        0.01,
        "注：黑虚线为冻结工程分区边界；灰点为未匹配边缘/背景点，不参与 MAE 着色；颜色越红表示误差越高。",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    png_path = context["output_root"] / "fig_spatial_error_heatmap.png"
    pdf_path = context["output_root"] / "fig_spatial_error_heatmap.pdf"
    save_dual(fig, png_path, pdf_path)
    return [{
        "filename": png_path.name + " / " + pdf_path.name,
        "data_source": f"{eval_path.name} + point_meta_v2.csv + patch_meta_v2_ps10_bg_excluded.csv + engineering_zone_polygons_v2.json",
        "figure_meaning": "在监测点散点上渲染 C3(PIR) 的 patch-derived test MAE，并叠加冻结工程分区边界，突出高误差团块仍主要被限制在 crest 区内。",
        "section": "第四章 4.6 PIR 空间误差分布（fig:spatial_error_heatmap）",
    }]


def write_missing_reports(output_root: Path, missing_items: list[str]) -> list[dict[str, str]]:
    manifest_rows = []
    if missing_items:
        path = output_root / "eDDR_routing_missing_report.md"
        lines = ["# eDDR_routing_missing_report", "", "以下缺失导致无法生成正式 EDDR 路由图：", ""]
        for item in missing_items:
            lines.append(f"- {item}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest_rows.append({
            "filename": path.name,
            "data_source": "缺失项说明",
            "figure_meaning": "说明 EDDR 路由图无法生成的原因。",
            "section": "第四章 4.6 EDDR 路由分析（缺失说明）",
        })
    return manifest_rows


def write_manifest(output_root: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# figure_manifest_ch4_results",
        "",
        "| 文件名 | 数据来源 | 图意 | 建议插入章节位置 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['filename']} | {row['data_source']} | {row['figure_meaning']} | {row['section']} |"
        )
    (output_root / "figure_manifest_ch4_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_plot_style()
    context = build_context(args.repo_root.resolve())
    results_main, results_subsets = read_results(context["ms_root"])

    manifest_rows: list[dict[str, str]] = []
    manifest_rows.extend(fig_main_performance_compare(results_main, results_subsets, context["output_root"]))
    manifest_rows.extend(fig_subset_compare(results_subsets, context["output_root"]))
    manifest_rows.extend(fig_event_windows(context=context, results_main=results_main))
    manifest_rows.extend(fig_patch_mae_heatmaps(context=context, results_main=results_main))
    manifest_rows.extend(fig_pggc_graph_compare(context["output_root"], context["ms_root"]))
    routing_rows, routing_missing = fig_eddr_routing_compare(context=context)
    manifest_rows.extend(routing_rows)
    manifest_rows.extend(write_missing_reports(context["output_root"], routing_missing))
    manifest_rows.extend(fig_prediction_smoothness(context=context, results_main=results_main))
    manifest_rows.extend(
        fig_physical_curve_comparison(
            context=context,
            results_main=results_main,
            sample_id=int(args.physical_sample_id),
        )
    )
    manifest_rows.extend(
        fig_spatial_error_heatmap(
            context=context,
            spatial_model=str(args.spatial_model),
            include_unmatched_context=bool(args.include_unmatched_context),
        )
    )
    write_manifest(context["output_root"], manifest_rows)

    print(context["output_root"])
    print(context["output_root"] / "figure_manifest_ch4_results.md")


if __name__ == "__main__":
    main()
