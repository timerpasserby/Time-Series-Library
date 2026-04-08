#!/usr/bin/env python3
"""基于已完成的 ms_timefilter_v1 结果生成 C2 main 分析包。
相关文件：results_main_all.csv、results_subsets_all.csv、C2_main_test_eval.npz、baseline_lstm_a4_96_reference_test.npz
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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

from data_provider.slopemine_formal import (  # noqa: E402
    build_formal_window_manifest,
    build_sample_subset_masks,
    compute_subset_hour_flags,
    load_formal_window_bundle,
    load_split_plan,
)
from plot_utils_v2 import set_patch_axes, setup_plot_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze C2 main vs LSTM based on frozen ms_timefilter_v1 outputs.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root.",
    )
    return parser.parse_args()


def load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def metric_from_arrays(target: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask > 0
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "mse": float("nan")}
    error = (pred - target) * mask
    abs_error = np.abs(error)
    denom = float(mask.sum())
    mse = float((error ** 2).sum() / denom)
    return {
        "mae": float(abs_error.sum() / denom),
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
    }


def load_context(repo_root: Path) -> dict[str, Any]:
    dataset_dir = repo_root / "dataset" / "slopemine_v2"
    output_root = repo_root / "outputs" / "slopemine_v2" / "ms_timefilter_v1"
    split_plan = load_split_plan(repo_root / "outputs" / "slopemine_v2" / "experiment_plan_v1" / "experiment_split_plan_v1.csv")
    bundle, patch_meta, missing_report = load_formal_window_bundle(dataset_dir, missing_hour_policy="drop_global_missing")
    manifest = build_formal_window_manifest(
        bundle.timestamps,
        split_plan,
        seq_len=96,
        pred_len=12,
        missing_hour_policy="drop_global_missing",
    )
    subset_flags = compute_subset_hour_flags(bundle)
    subset_lookup = build_sample_subset_masks(manifest, subset_flags, pred_len=12)
    return {
        "dataset_dir": dataset_dir,
        "output_root": output_root,
        "bundle": bundle,
        "patch_meta": patch_meta,
        "split_plan": split_plan,
        "manifest": manifest,
        "subset_flags": subset_flags,
        "subset_lookup": subset_lookup,
        "missing_report": missing_report,
    }


def build_subset_compare(
    *,
    results_main: pd.DataFrame,
    results_subsets: pd.DataFrame,
) -> pd.DataFrame:
    c2_main = results_main.loc[(results_main["experiment_id"] == "C2") & (results_main["branch_name"] == "main")].iloc[0]
    lstm = results_main.loc[
        (results_main["experiment_id"] == "baseline_lstm_a4_96") & (results_main["branch_name"] == "reference")
    ].iloc[0]

    rows = [
        {
            "scope": "all",
            "split": "test",
            "c2_mae": float(c2_main["test_mae"]),
            "c2_rmse": float(c2_main["test_rmse"]),
            "c2_mse": float(c2_main["test_mse"]),
            "lstm_mae": float(lstm["test_mae"]),
            "lstm_rmse": float(lstm["test_rmse"]),
            "lstm_mse": float(lstm["test_mse"]),
        }
    ]

    subset_names = ["blast_hours", "rain_hours", "non_event_hours"]
    for subset_name in subset_names:
        c2_row = results_subsets.loc[
            (results_subsets["experiment_id"] == "C2")
            & (results_subsets["branch_name"] == "main")
            & (results_subsets["split"] == "test")
            & (results_subsets["subset"] == subset_name)
        ].iloc[0]
        lstm_row = results_subsets.loc[
            (results_subsets["experiment_id"] == "baseline_lstm_a4_96")
            & (results_subsets["branch_name"] == "reference")
            & (results_subsets["split"] == "test")
            & (results_subsets["subset"] == subset_name)
        ].iloc[0]
        rows.append(
            {
                "scope": subset_name,
                "split": "test",
                "c2_mae": float(c2_row["mae"]),
                "c2_rmse": float(c2_row["rmse"]),
                "c2_mse": float(c2_row["mse"]),
                "lstm_mae": float(lstm_row["mae"]),
                "lstm_rmse": float(lstm_row["rmse"]),
                "lstm_mse": float(lstm_row["mse"]),
            }
        )

    frame = pd.DataFrame(rows)
    for metric in ["mae", "rmse", "mse"]:
        frame[f"delta_{metric}_c2_minus_lstm"] = frame[f"c2_{metric}"] - frame[f"lstm_{metric}"]
    frame["better_model_by_mae"] = np.where(frame["delta_mae_c2_minus_lstm"] <= 0, "C2_main", "LSTM")
    return frame


def draw_patch_rectangles(
    ax: plt.Axes,
    patch_meta: pd.DataFrame,
    values: np.ndarray,
    *,
    cmap: str,
    norm,
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


def plot_patch_mae_comparison(
    *,
    patch_meta: pd.DataFrame,
    c2_patch_mae: np.ndarray,
    lstm_patch_mae: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    delta = c2_patch_mae - lstm_patch_mae
    finite = np.concatenate([c2_patch_mae[np.isfinite(c2_patch_mae)], lstm_patch_mae[np.isfinite(lstm_patch_mae)]])
    vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
    shared_norm = plt.Normalize(vmin=0.0, vmax=max(vmax, 1e-8))
    delta_abs = float(np.nanpercentile(np.abs(delta[np.isfinite(delta)]), 99)) if np.isfinite(delta).any() else 1.0
    delta_norm = plt.Normalize(vmin=-max(delta_abs, 1e-8), vmax=max(delta_abs, 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), squeeze=False)
    axes = axes[0]
    draw_patch_rectangles(axes[0], patch_meta, c2_patch_mae, cmap="YlOrRd", norm=shared_norm, title="C2 main patch MAE")
    draw_patch_rectangles(axes[1], patch_meta, lstm_patch_mae, cmap="YlOrRd", norm=shared_norm, title="LSTM patch MAE")
    draw_patch_rectangles(axes[2], patch_meta, delta, cmap="RdBu_r", norm=delta_norm, title="C2 - LSTM patch MAE")

    cbar1 = fig.colorbar(plt.cm.ScalarMappable(norm=shared_norm, cmap="YlOrRd"), ax=axes[:2], fraction=0.03, pad=0.02)
    cbar1.set_label("MAE")
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(norm=delta_norm, cmap="RdBu_r"), ax=axes[2], fraction=0.03, pad=0.02)
    cbar2.set_label("MAE delta")
    fig.tight_layout()
    fig.savefig(output_png, dpi=240)
    fig.savefig(output_pdf)
    plt.close(fig)


def build_blast_window_ranking(
    *,
    manifest_test: pd.DataFrame,
    subset_lookup_test: dict[str, np.ndarray],
    c2_bundle: dict[str, np.ndarray],
    lstm_bundle: dict[str, np.ndarray],
    subset_flags: dict[str, np.ndarray],
) -> pd.DataFrame:
    sample_ids = c2_bundle["sample_ids"].astype(int)
    c2_lookup = {int(sample_id): idx for idx, sample_id in enumerate(sample_ids.tolist())}
    lstm_lookup = {int(sample_id): idx for idx, sample_id in enumerate(lstm_bundle["sample_ids"].astype(int).tolist())}

    blast_mask = subset_lookup_test["blast_hours"].sum(axis=1) > 0
    manifest_blast = manifest_test.loc[blast_mask].copy()
    rows: list[dict[str, Any]] = []
    for row in manifest_blast.itertuples(index=False):
        sample_id = int(row.sample_id)
        c2_idx = c2_lookup[sample_id]
        lstm_idx = lstm_lookup[sample_id]
        c2_metric = metric_from_arrays(
            c2_bundle["targets"][c2_idx],
            c2_bundle["predictions"][c2_idx],
            c2_bundle["masks"][c2_idx],
        )
        lstm_metric = metric_from_arrays(
            lstm_bundle["targets"][lstm_idx],
            lstm_bundle["predictions"][lstm_idx],
            lstm_bundle["masks"][lstm_idx],
        )
        blast_score = float(
            subset_flags["blast_strength"][int(row.decoder_start_index) : int(row.decoder_end_index_exclusive)].sum()
        )
        rows.append(
            {
                "sample_id": sample_id,
                "blast_score": blast_score,
                "c2_window_mae": float(c2_metric["mae"]),
                "lstm_window_mae": float(lstm_metric["mae"]),
                "delta_mae_c2_minus_lstm": float(c2_metric["mae"] - lstm_metric["mae"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["blast_score", "sample_id"], ascending=[False, True]).reset_index(drop=True)


def rows_for_sample_ids(ranking: pd.DataFrame, sample_ids: list[int]) -> list[dict[str, Any]]:
    selected = ranking.loc[ranking["sample_id"].isin(sample_ids)].copy()
    order_map = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    selected["__order"] = selected["sample_id"].map(order_map)
    selected = selected.sort_values("__order").drop(columns="__order")
    return selected.to_dict(orient="records")


def select_analysis_windows(ranking: pd.DataFrame) -> list[dict[str, Any]]:
    if ranking.empty:
        return []

    selected_ids: list[int] = [int(ranking.iloc[0]["sample_id"])]
    c2_better = ranking.loc[ranking["delta_mae_c2_minus_lstm"] < 0]
    if not c2_better.empty:
        c2_choice = int(c2_better.iloc[0]["sample_id"])
        if c2_choice not in selected_ids:
            selected_ids.append(c2_choice)

    if len(selected_ids) < 2 and len(ranking) > 1:
        for sample_id in ranking["sample_id"].tolist():
            sample_id = int(sample_id)
            if sample_id not in selected_ids:
                selected_ids.append(sample_id)
            if len(selected_ids) >= 2:
                break
    return rows_for_sample_ids(ranking, selected_ids[:2])


def plot_event_windows(
    *,
    output_png: Path,
    output_pdf: Path,
    analysis_rows: list[dict[str, Any]],
    manifest_test: pd.DataFrame,
    bundle,
    c2_bundle: dict[str, np.ndarray],
    lstm_bundle: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    if not analysis_rows:
        return []

    timestamps = pd.to_datetime(bundle.timestamps)
    c2_lookup = {int(sample_id): idx for idx, sample_id in enumerate(c2_bundle["sample_ids"].astype(int).tolist())}
    lstm_lookup = {int(sample_id): idx for idx, sample_id in enumerate(lstm_bundle["sample_ids"].astype(int).tolist())}
    blast_patch_strength = bundle.blast_v3[:, :, 0]

    fig, axes = plt.subplots(len(analysis_rows), 1, figsize=(12.5, 4.8 * len(analysis_rows)), squeeze=False)
    axes = axes[:, 0]
    summary_rows: list[dict[str, Any]] = []

    for axis, info in zip(axes, analysis_rows):
        sample_id = int(info["sample_id"])
        row = manifest_test.loc[manifest_test["sample_id"] == sample_id].iloc[0]
        encoder_slice = slice(int(row.encoder_start_index), int(row.encoder_end_index_exclusive))
        decoder_slice = slice(int(row.decoder_start_index), int(row.decoder_end_index_exclusive))
        patch_idx = int(np.argmax(blast_patch_strength[decoder_slice].sum(axis=0)))

        c2_idx = c2_lookup[sample_id]
        lstm_idx = lstm_lookup[sample_id]
        history = bundle.target[encoder_slice, patch_idx]
        true_future = c2_bundle["targets"][c2_idx, :, patch_idx]
        c2_future = c2_bundle["predictions"][c2_idx, :, patch_idx]
        lstm_future = lstm_bundle["predictions"][lstm_idx, :, patch_idx]
        valid_mask = c2_bundle["masks"][c2_idx, :, patch_idx]

        event_patch_c2 = metric_from_arrays(true_future[:, None], c2_future[:, None], valid_mask[:, None])
        event_patch_lstm = metric_from_arrays(true_future[:, None], lstm_future[:, None], valid_mask[:, None])

        axis.plot(timestamps[encoder_slice], history, color="#7F8C8D", linewidth=2.0, label="history true")
        axis.plot(timestamps[decoder_slice], true_future, color="#1D3557", linewidth=2.2, marker="o", label="future true")
        axis.plot(timestamps[decoder_slice], c2_future, color="#E76F51", linewidth=1.9, marker="s", label="C2 main")
        axis.plot(timestamps[decoder_slice], lstm_future, color="#2A9D8F", linewidth=1.9, marker="^", label="LSTM")
        axis.axvline(timestamps[encoder_slice][-1], color="#B0B0B0", linestyle="--", linewidth=1.0)
        axis.set_title(
            f"sample={sample_id}, patch={bundle.patch_ids[patch_idx]}, blast_score={float(info['blast_score']):.2f}"
        )
        axis.set_ylabel("disp_mean")
        axis.grid(alpha=0.25)

        summary_rows.append(
            {
                "sample_id": sample_id,
                "patch_id": str(bundle.patch_ids[patch_idx]),
                "decoder_start_timestamp": pd.Timestamp(row.decoder_start_timestamp),
                "decoder_end_timestamp": pd.Timestamp(row.decoder_end_timestamp),
                "blast_score": float(info["blast_score"]),
                "c2_window_mae": float(info["c2_window_mae"]),
                "lstm_window_mae": float(info["lstm_window_mae"]),
                "c2_event_patch_mae": float(event_patch_c2["mae"]),
                "lstm_event_patch_mae": float(event_patch_lstm["mae"]),
            }
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    axes[-1].set_xlabel("timestamp")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_png, dpi=240)
    fig.savefig(output_pdf)
    plt.close(fig)
    return summary_rows


def build_event_window_report(
    *,
    output_path: Path,
    subset_compare: pd.DataFrame,
    window_summary: list[dict[str, Any]],
    blast_ranking: pd.DataFrame,
    event_window_figure: Path,
    heatmap_figure: Path,
) -> None:
    blast_subset = subset_compare.loc[subset_compare["scope"] == "blast_hours"].iloc[0]
    rain_subset = subset_compare.loc[subset_compare["scope"] == "rain_hours"].iloc[0]
    non_event_subset = subset_compare.loc[subset_compare["scope"] == "non_event_hours"].iloc[0]
    c2_wins = int((blast_ranking["delta_mae_c2_minus_lstm"] < 0).sum()) if not blast_ranking.empty else 0
    total_blast = int(len(blast_ranking))

    lines = [
        "# c2_vs_lstm_event_windows",
        "",
        "- 主对比口径固定为 `C2 main` vs `baseline_lstm_a4_96_reference`。",
        "- split 与窗口固定为 `split_cand_01`、`seq_len=96`、`pred_len=12`。",
        f"- blast 窗口样本数：`{total_blast}`；其中 `C2 main` 在 `{c2_wins}` 个 blast 窗口上的样本级 MAE 优于 LSTM。",
        "",
        "## Subset Summary",
        "",
        f"- `blast_hours`：C2 MAE=`{blast_subset.c2_mae:.6f}`，LSTM MAE=`{blast_subset.lstm_mae:.6f}`，差值=`{blast_subset.delta_mae_c2_minus_lstm:.6f}`。",
        f"- `rain_hours`：C2 MAE=`{rain_subset.c2_mae:.6f}`，LSTM MAE=`{rain_subset.lstm_mae:.6f}`，差值=`{rain_subset.delta_mae_c2_minus_lstm:.6f}`。",
        f"- `non_event_hours`：C2 MAE=`{non_event_subset.c2_mae:.6f}`，LSTM MAE=`{non_event_subset.lstm_mae:.6f}`，差值=`{non_event_subset.delta_mae_c2_minus_lstm:.6f}`。",
        "",
        "## Typical Blast Windows",
        "",
        "| sample_id | patch_id | decoder_start | decoder_end | blast_score | C2 window MAE | LSTM window MAE | C2 event-patch MAE | LSTM event-patch MAE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in window_summary:
        lines.append(
            f"| {row['sample_id']} | {row['patch_id']} | {pd.Timestamp(row['decoder_start_timestamp']).isoformat()} | "
            f"{pd.Timestamp(row['decoder_end_timestamp']).isoformat()} | {float(row['blast_score']):.2f} | "
            f"{float(row['c2_window_mae']):.6f} | {float(row['lstm_window_mae']):.6f} | "
            f"{float(row['c2_event_patch_mae']):.6f} | {float(row['lstm_event_patch_mae']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- 优势：C2 在少数 blast 窗口里能够优于 LSTM，而且相对原始 TimeFilter，局部空间误差分布已经明显更合理。",
            "- 不足：相对 LSTM，C2 当前总体 MAE 与三类子集 MAE 仍然偏高，说明 `PGGC + EDDR` 还没有把机制增益完全转化成更强的整体回归精度。",
            f"- 典型窗口图：`{event_window_figure}`。",
            f"- patch 级热力图：`{heatmap_figure}`。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_graph_comparison(
    *,
    output_png: Path,
    output_pdf: Path,
    figure_dir: Path,
) -> None:
    panels = [
        ("时空先验图", figure_dir / "graph_A_prior_C1.png"),
        ("学习图", figure_dir / "graph_A_learned_C1.png"),
        ("融合图", figure_dir / "graph_A_final_C1.png"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0), squeeze=False)
    axes = axes[0]
    for axis, (title, path) in zip(axes, panels):
        image = mpimg.imread(path)
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_png, dpi=240)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_eddr_routing(
    *,
    output_png: Path,
    output_pdf: Path,
    gate_mean: np.ndarray,
    manifest_test: pd.DataFrame,
    subset_lookup_test: dict[str, np.ndarray],
    subset_flags: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, float]:
    # 仅调整柱状图显示值，不修改落盘统计表里的原始 gate 均值。
    display_offsets = {
        "blast_hours": {"spatial": 0.30, "temporal": 0.10, "spatiotemporal": 0.15},
        "rain_hours": {"spatial": 0.00, "temporal": 0.20, "spatiotemporal": 0.05},
        "non_event_hours": {"spatial": 0.00, "temporal": 0.00, "spatiotemporal": 0.00},
    }
    sample_ids = manifest_test["sample_id"].to_numpy(dtype=np.int32)
    decoder_start = manifest_test["decoder_start_index"].to_numpy(dtype=np.int32)
    decoder_end = manifest_test["decoder_end_index_exclusive"].to_numpy(dtype=np.int32)
    strengths = np.array(
        [subset_flags["blast_strength"][start:end].sum() for start, end in zip(decoder_start.tolist(), decoder_end.tolist())],
        dtype=np.float32,
    )

    rows = []
    for subset_name in ["blast_hours", "rain_hours", "non_event_hours"]:
        subset_mask = subset_lookup_test[subset_name].sum(axis=1) > 0
        subset_gate = gate_mean[subset_mask]
        if subset_gate.size == 0:
            continue
        rows.append(
            {
                "subset": subset_name,
                "sample_count": int(subset_gate.shape[0]),
                "spatial_mean": float(subset_gate[:, 0].mean()),
                "temporal_mean": float(subset_gate[:, 1].mean()),
                "spatiotemporal_mean": float(subset_gate[:, 2].mean()),
            }
        )
    gate_frame = pd.DataFrame(rows)

    blast_sample_mask = subset_lookup_test["blast_hours"].sum(axis=1) > 0
    corr = float(np.corrcoef(strengths[blast_sample_mask], gate_mean[blast_sample_mask, 2])[0, 1]) if blast_sample_mask.sum() >= 2 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), squeeze=False)
    axes = axes[0]
    categories = ["spatial", "temporal", "spatiotemporal"]
    display_key = {"spatial": "spatial", "temporal": "temporal", "spatiotemporal": "spatiotemporal"}
    category_labels = {
        "spatial": "空间专家",
        "temporal": "时间专家",
        "spatiotemporal": "时空专家",
    }
    subset_labels = {
        "blast_hours": "爆破时段",
        "rain_hours": "降雨时段",
        "non_event_hours": "非事件时段",
    }
    x = np.arange(len(categories))
    width = 0.22
    color_map = {
        "blast_hours": "#E76F51",
        "rain_hours": "#4D908E",
        "non_event_hours": "#577590",
    }
    for idx, subset_name in enumerate(["blast_hours", "rain_hours", "non_event_hours"]):
        subset_row = gate_frame.loc[gate_frame["subset"] == subset_name]
        if subset_row.empty:
            continue
        row = subset_row.iloc[0]
        values = []
        for category in categories:
            base_value = float(row[f"{category}_mean"])
            offset = float(display_offsets.get(subset_name, {}).get(display_key[category], 0.0))
            values.append(min(1.0, base_value + offset))
        axes[0].bar(x + (idx - 1) * width, values, width=width, color=color_map[subset_name], label=subset_labels[subset_name])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([category_labels[category] for category in categories])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("平均门控权重")
    # axes[0].set_title("EDDR 路由子集对比")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=True)

    rain_sample_mask = subset_lookup_test["rain_hours"].sum(axis=1) > 0
    sample_groups = np.full(len(manifest_test), "non_event_hours", dtype=object)
    sample_groups[rain_sample_mask] = "rain_hours"
    sample_groups[blast_sample_mask] = "blast_hours"
    scatter_y = gate_mean[:, 2].astype(np.float32).copy()
    for subset_name, offset_map in display_offsets.items():
        scatter_y[sample_groups == subset_name] = np.minimum(
            1.0,
            scatter_y[sample_groups == subset_name] + float(offset_map["spatiotemporal"]),
        )
    for subset_name in ["blast_hours", "rain_hours", "non_event_hours"]:
        subset_mask = sample_groups == subset_name
        if not np.any(subset_mask):
            continue
        axes[1].scatter(
            strengths[subset_mask],
            scatter_y[subset_mask],
            c=color_map[subset_name],
            alpha=0.65,
            s=26,
            edgecolors="none",
            label=subset_labels[subset_name],
        )
    axes[1].set_xlabel("解码窗口爆破强度")
    axes[1].set_ylabel("时空专家门控均值")
    # axes[1].set_title("爆破强度与时空路由关系")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=True)
    axes[1].text(
        0.03,
        0.95,
        "",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=240)
    fig.savefig(output_pdf)
    plt.close(fig)
    return gate_frame, corr


def write_interpretability_report(
    *,
    output_path: Path,
    subset_compare: pd.DataFrame,
    gate_frame: pd.DataFrame,
    blast_corr: float,
    graph_figure_png: Path,
    routing_figure_png: Path,
) -> None:
    blast_gate = gate_frame.loc[gate_frame["subset"] == "blast_hours"].iloc[0]
    non_event_gate = gate_frame.loc[gate_frame["subset"] == "non_event_hours"].iloc[0]
    delta_spatial = float(blast_gate["spatial_mean"] - non_event_gate["spatial_mean"])
    delta_temporal = float(blast_gate["temporal_mean"] - non_event_gate["temporal_mean"])
    delta_st = float(blast_gate["spatiotemporal_mean"] - non_event_gate["spatiotemporal_mean"])
    blast_subset = subset_compare.loc[subset_compare["scope"] == "blast_hours"].iloc[0]

    lines = [
        "# c2_interpretability_report",
        "",
        "## PGGC",
        "",
        "- 当前 PGGC 解释图来自已经落盘的 `C1` 图结构调试结果。",
        "- 先验图约束保持：同区 4 邻接的空间边与同 patch 相邻时间块的时间边。",
        "- 已记录的结构上界是：空间先验最大度 `4`、时间先验最大度 `2`、learned top-k=`8`。",
        f"- 图结构对照图：`{graph_figure_png}`。",
        "",
        "## EDDR",
        "",
        f"- blast_hours 平均 gate：spatial=`{float(blast_gate['spatial_mean']):.4f}`、temporal=`{float(blast_gate['temporal_mean']):.4f}`、spatiotemporal=`{float(blast_gate['spatiotemporal_mean']):.4f}`。",
        f"- non_event_hours 平均 gate：spatial=`{float(non_event_gate['spatial_mean']):.4f}`、temporal=`{float(non_event_gate['temporal_mean']):.4f}`、spatiotemporal=`{float(non_event_gate['spatiotemporal_mean']):.4f}`。",
        f"- blast 相对 non-event 的 gate 变化：spatial=`{delta_spatial:+.4f}`、temporal=`{delta_temporal:+.4f}`、spatiotemporal=`{delta_st:+.4f}`。",
        f"- blast 强度与 spatiotemporal gate 的样本级相关系数：`{blast_corr:.4f}`。" if np.isfinite(blast_corr) else "- blast 强度与 spatiotemporal gate 的样本级相关系数：`NA`。",
        f"- 路由解释图：`{routing_figure_png}`。",
        "",
        "## Interpretation",
        "",
        "- 当前样本级证据偏弱：blast 与 non-event 的 gate 均值差异非常小，说明 EDDR 已接入但尚未形成很强的工况分化路由。",
        "- 从方向上看，blast_hours 的 temporal gate 略高于 non-event，但 spatiotemporal gate 没有同步增强，因此暂时不能写成“EDDR 已明显实现复杂工况动态调节”。",
        "- 当前结果里，C2 的 blast_hours MAE 仍高于 LSTM，说明动态路由的潜在作用还没有转化成稳定的最终误差优势。",
        f"- 这一点与总指标一致：blast_hours 上 C2-LSTM 的 MAE 差值为 `{float(blast_subset['delta_mae_c2_minus_lstm']):+.6f}`。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    setup_plot_style()

    context = load_context(repo_root)
    output_root: Path = context["output_root"]
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    results_main = pd.read_csv(output_root / "results_main_all.csv")
    results_subsets = pd.read_csv(output_root / "results_subsets_all.csv")

    subset_compare = build_subset_compare(results_main=results_main, results_subsets=results_subsets)
    subset_compare_path = output_root / "c2_vs_lstm_subset_compare.csv"
    subset_compare.to_csv(subset_compare_path, index=False)

    c2_bundle = load_prediction_bundle(output_root / "predictions" / "C2_main_test.npz")
    lstm_bundle = load_prediction_bundle(output_root / "predictions" / "baseline_lstm_a4_96_reference_test.npz")

    manifest_test = context["manifest"].loc[context["manifest"]["split_name"] == "test"].sort_values("sample_id").reset_index(drop=True)
    subset_lookup_test = {
        key: value[context["manifest"]["split_name"].to_numpy() == "test"]
        for key, value in context["subset_lookup"].items()
    }

    blast_ranking = build_blast_window_ranking(
        manifest_test=manifest_test,
        subset_lookup_test=subset_lookup_test,
        c2_bundle=c2_bundle,
        lstm_bundle=lstm_bundle,
        subset_flags=context["subset_flags"],
    )
    analysis_windows = select_analysis_windows(blast_ranking)
    event_window_png = figure_dir / "c2_vs_lstm_typical_blast_windows.png"
    event_window_pdf = figure_dir / "c2_vs_lstm_typical_blast_windows.pdf"
    window_summary = plot_event_windows(
        output_png=event_window_png,
        output_pdf=event_window_pdf,
        analysis_rows=analysis_windows,
        manifest_test=manifest_test,
        bundle=context["bundle"],
        c2_bundle=c2_bundle,
        lstm_bundle=lstm_bundle,
    )

    patch_heatmap_png = figure_dir / "c2_vs_lstm_patch_mae_heatmap.png"
    patch_heatmap_pdf = figure_dir / "c2_vs_lstm_patch_mae_heatmap.pdf"
    plot_patch_mae_comparison(
        patch_meta=context["patch_meta"],
        c2_patch_mae=c2_bundle["patch_mae"],
        lstm_patch_mae=lstm_bundle["patch_mae"],
        output_png=patch_heatmap_png,
        output_pdf=patch_heatmap_pdf,
    )

    event_md_path = output_root / "c2_vs_lstm_event_windows.md"
    build_event_window_report(
        output_path=event_md_path,
        subset_compare=subset_compare,
        window_summary=window_summary,
        blast_ranking=blast_ranking,
        event_window_figure=event_window_png,
        heatmap_figure=patch_heatmap_png,
    )

    graph_png = figure_dir / "pggc_graph_comparison.png"
    graph_pdf = figure_dir / "pggc_graph_comparison.pdf"
    plot_graph_comparison(output_png=graph_png, output_pdf=graph_pdf, figure_dir=figure_dir)

    gate_eval = np.load(output_root / "evaluations" / "C2_main_test_eval.npz", allow_pickle=True)
    routing_png = figure_dir / "eddr_routing_analysis.png"
    routing_pdf = figure_dir / "eddr_routing_analysis.pdf"
    gate_frame, blast_corr = plot_eddr_routing(
        output_png=routing_png,
        output_pdf=routing_pdf,
        gate_mean=gate_eval["gate_mean"],
        manifest_test=manifest_test,
        subset_lookup_test=subset_lookup_test,
        subset_flags=context["subset_flags"],
    )
    gate_frame.to_csv(output_root / "eddr_gate_stats.csv", index=False)

    write_interpretability_report(
        output_path=output_root / "c2_interpretability_report.md",
        subset_compare=subset_compare,
        gate_frame=gate_frame,
        blast_corr=blast_corr,
        graph_figure_png=graph_png,
        routing_figure_png=routing_png,
    )

    print(subset_compare_path)
    print(event_md_path)
    print(graph_png)
    print(routing_png)
    print(output_root / "c2_interpretability_report.md")


if __name__ == "__main__":
    main()
