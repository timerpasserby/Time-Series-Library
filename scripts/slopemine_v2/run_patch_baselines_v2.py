#!/usr/bin/env python3
"""这个脚本负责运行 slopemine v2 的 A1-A4 基线实验并导出评估结果。
相关文件：config_v2.toml、build_patch_dataset_v2.py
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 兼容 Python 3.10
    import tomli as tomllib

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.lib.stride_tricks import sliding_window_view


EXPERIMENTS = {
    "A1": "patch-only internal features",
    "A2": "A1 + weather",
    "A3": "A2 + blast V2",
    "A4": "A2 + blast V3",
}

CJK_FONT_CANDIDATES = [
    "Songti SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "STHeiti",
    "Arial Unicode MS",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run slopemine v2 patch baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/slopemine_v2/config_v2.toml"),
        help="Path to the TOML config file.",
    )
    return parser.parse_args()


def setup_plot_style() -> None:
    """设置统一的中文友好绘图风格。"""

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    cjk_font = next((name for name in CJK_FONT_CANDIDATES if name in available_fonts), "DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": [cjk_font, "Times New Roman", "DejaVu Serif"],
            "font.serif": [cjk_font, "Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#D0D0D0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def set_patch_axes_limits(ax: plt.Axes, patch_meta: pd.DataFrame, padding: float = 6.0) -> None:
    """把坐标轴锁定到 patch 的真实空间范围。"""

    x_min = float(patch_meta["cell_x_min"].min()) - padding
    x_max = float(patch_meta["cell_x_max"].max()) + padding
    y_min = float(patch_meta["cell_y_min"].min()) - padding
    y_max = float(patch_meta["cell_y_max"].max()) + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def load_config(config_path: Path) -> dict[str, Any]:
    """读取并解析配置文件。"""

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    repo_root = config_path.resolve().parents[2]
    for key in ["dataset_dir", "output_dir"]:
        path = Path(config["paths"][key])
        config["paths"][key] = (repo_root / path).resolve() if not path.is_absolute() else path.resolve()
    return config


def load_bundle(dataset_dir: Path) -> dict[str, Any]:
    """读取基础张量和窗口清单。"""

    bundle = np.load(dataset_dir / "patch_tensor_base_v2_ps10.npz", allow_pickle=True)
    manifest = pd.read_csv(dataset_dir / "window_manifest_v2_ps10.csv")
    patch_meta = pd.read_csv(dataset_dir / "patch_meta_v2_ps10.csv")
    bundle_patch_ids = bundle["patch_ids"].tolist()
    patch_meta = (
        patch_meta.set_index("patch_id")
        .reindex(bundle_patch_ids)
        .reset_index()
    )
    return {
        "bundle": bundle,
        "manifest": manifest,
        "patch_meta": patch_meta,
    }


def broadcast_global(matrix: np.ndarray, patch_count: int) -> np.ndarray:
    """把全局特征广播到 patch 维。"""

    return np.broadcast_to(matrix[:, None, :], (matrix.shape[0], patch_count, matrix.shape[1])).astype(np.float32)


def build_feature_tensor(bundle: dict[str, Any], experiment_key: str) -> tuple[np.ndarray, list[str]]:
    """按实验配置拼接输入特征。"""

    internal = bundle["bundle"]["internal"].astype(np.float32)
    weather = broadcast_global(bundle["bundle"]["weather"].astype(np.float32), internal.shape[1])
    blast_v2 = broadcast_global(bundle["bundle"]["blast_v2"].astype(np.float32), internal.shape[1])
    blast_v3 = bundle["bundle"]["blast_v3"].astype(np.float32)

    internal_names = bundle["bundle"]["internal_feature_names"].tolist()
    weather_names = bundle["bundle"]["weather_feature_names"].tolist()
    blast_v2_names = bundle["bundle"]["blast_v2_feature_names"].tolist()
    blast_v3_names = bundle["bundle"]["blast_v3_feature_names"].tolist()

    if experiment_key == "A1":
        return internal, list(internal_names)
    if experiment_key == "A2":
        return np.concatenate([internal, weather], axis=-1), list(internal_names) + list(weather_names)
    if experiment_key == "A3":
        return np.concatenate([internal, weather, blast_v2], axis=-1), list(internal_names) + list(weather_names) + list(blast_v2_names)
    if experiment_key == "A4":
        return np.concatenate([internal, weather, blast_v3], axis=-1), list(internal_names) + list(weather_names) + list(blast_v3_names)
    raise ValueError(f"未知实验：{experiment_key}")


def summarize_windows(feature_tensor: np.ndarray, seq_len: int, recent_window: int) -> np.ndarray:
    """把输入窗口压缩成可训练的统计摘要特征。"""

    windows = sliding_window_view(feature_tensor, window_shape=seq_len, axis=0)
    recent_steps = min(recent_window, seq_len)

    with np.errstate(invalid="ignore"):
        last = windows[..., -1]
        mean = np.nanmean(windows, axis=-1)
        std = np.nanstd(windows, axis=-1)
        trend = windows[..., -1] - windows[..., 0]
        recent_mean = np.nanmean(windows[..., -recent_steps:], axis=-1)

    summary = np.concatenate([last, mean, std, trend, recent_mean], axis=-1)
    return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def prepare_targets(target: np.ndarray, input_mask: np.ndarray, pred_len: int, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造监督学习需要的预测窗口和有效掩码。"""

    target_windows = sliding_window_view(target, window_shape=pred_len, axis=0)
    target_mask = (~np.isnan(target)).astype(np.uint8)
    target_mask_windows = sliding_window_view(target_mask, window_shape=pred_len, axis=0)
    input_windows = sliding_window_view(input_mask.astype(np.float32), window_shape=seq_len, axis=0)

    usable_count = target.shape[0] - seq_len - pred_len + 1
    y = np.moveaxis(target_windows[seq_len : seq_len + usable_count], -1, 1)
    y_mask = np.moveaxis(target_mask_windows[seq_len : seq_len + usable_count], -1, 1)
    x_valid = input_windows[:usable_count].mean(axis=-1) > 0
    return y.astype(np.float32), y_mask.astype(np.uint8), x_valid.astype(np.uint8)


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练共享权重的 ridge 回归基线。"""

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)

    x_norm = (x_train - x_mean) / x_std
    y_center = y_train - y_mean

    feature_dim = x_norm.shape[1]
    reg = math.sqrt(alpha) * np.eye(feature_dim, dtype=np.float32)
    x_aug = np.vstack([x_norm, reg])
    y_aug = np.vstack([y_center, np.zeros((feature_dim, y_center.shape[1]), dtype=np.float32)])
    weights = np.linalg.lstsq(x_aug, y_aug, rcond=None)[0]
    return weights.astype(np.float32), x_mean.astype(np.float32), x_std.astype(np.float32), y_mean.astype(np.float32)


def predict_ridge(
    x_value: np.ndarray,
    weights: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
) -> np.ndarray:
    """使用 ridge 模型做多步预测。"""

    x_norm = (x_value - x_mean) / x_std
    return (x_norm @ weights + y_mean).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """计算 MAE、RMSE 和 MSE。"""

    if mask is None:
        error = (y_pred - y_true).reshape(-1)
    else:
        error = (y_pred - y_true)[mask]

    if error.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "mse": np.nan}

    mse = float(np.mean(np.square(error)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
    }


def build_hour_subsets(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    """构造 blast、rain 和 non-event 三类小时标签。"""

    weather = bundle["bundle"]["weather"].astype(np.float32)
    blast_v2 = bundle["bundle"]["blast_v2"].astype(np.float32)
    rainfall = weather[:, 3] > 0
    blast_hour = blast_v2[:, 0] > 0
    non_event = ~(rainfall | blast_hour)
    return {
        "blast_hours": blast_hour.astype(np.uint8),
        "rain_hours": rainfall.astype(np.uint8),
        "non_event_hours": non_event.astype(np.uint8),
        "blast_strength": blast_v2[:, 1].astype(np.float32),
    }


def run_single_experiment(
    experiment_key: str,
    seq_len: int,
    pred_len: int,
    bundle: dict[str, Any],
    config: dict[str, Any],
    subsets: dict[str, np.ndarray],
) -> dict[str, Any]:
    """运行单个实验配置并返回评估结果。"""

    feature_tensor, feature_names = build_feature_tensor(bundle, experiment_key)
    summary = summarize_windows(
        feature_tensor=feature_tensor,
        seq_len=seq_len,
        recent_window=int(config["baseline"]["recent_window"]),
    )

    target = bundle["bundle"]["target"].astype(np.float32)
    input_mask = bundle["bundle"]["input_mask"].astype(np.uint8)
    y, y_mask, x_valid = prepare_targets(target, input_mask, pred_len, seq_len)
    num_samples = y.shape[0]
    summary = summary[:num_samples]

    manifest_row = bundle["manifest"].loc[
        (bundle["manifest"]["seq_len"] == seq_len) & (bundle["manifest"]["pred_len"] == pred_len)
    ].iloc[0]

    train_end = int(manifest_row["train_end"])
    val_end = int(manifest_row["val_end"])
    patch_count = summary.shape[1]
    summary_dim = summary.shape[2]

    x_flat = summary.reshape(num_samples * patch_count, summary_dim)
    y_flat = np.moveaxis(y, 2, 1).reshape(num_samples * patch_count, pred_len)
    mask_flat = np.moveaxis(y_mask, 2, 1).reshape(num_samples * patch_count, pred_len)
    row_valid = (mask_flat.min(axis=1) == 1) & x_valid.reshape(-1)

    sample_index = np.repeat(np.arange(num_samples), patch_count)
    patch_index = np.tile(np.arange(patch_count), num_samples)

    train_rows = row_valid & (sample_index < train_end)
    test_rows = row_valid & (sample_index >= val_end)

    weights, x_mean, x_std, y_mean = fit_ridge(
        x_train=x_flat[train_rows],
        y_train=y_flat[train_rows],
        alpha=float(config["baseline"]["ridge_alpha"]),
    )
    pred_test = predict_ridge(x_flat[test_rows], weights, x_mean, x_std, y_mean)
    true_test = y_flat[test_rows]
    test_sample_index = sample_index[test_rows]
    test_patch_index = patch_index[test_rows]

    total_metrics = compute_metrics(true_test, pred_test)

    subset_rows = []
    sample_hour_masks: dict[str, np.ndarray] = {}
    for subset_name in ["blast_hours", "rain_hours", "non_event_hours"]:
        hour_windows = sliding_window_view(subsets[subset_name], window_shape=pred_len, axis=0)
        hour_windows = hour_windows[seq_len : seq_len + num_samples]
        subset_flat = np.repeat(hour_windows[:, None, :], patch_count, axis=1).reshape(num_samples * patch_count, pred_len)
        sample_hour_masks[subset_name] = subset_flat[test_rows].astype(bool)
        subset_metric = compute_metrics(true_test, pred_test, mask=sample_hour_masks[subset_name])
        subset_rows.append(
            {
                "experiment": experiment_key,
                "description": EXPERIMENTS[experiment_key],
                "seq_len": seq_len,
                "pred_len": pred_len,
                "subset": subset_name,
                **subset_metric,
            }
        )

    abs_error = np.abs(pred_test - true_test)
    patch_mae = np.zeros(patch_count, dtype=np.float32)
    for patch_idx in range(patch_count):
        patch_mask = test_patch_index == patch_idx
        patch_mae[patch_idx] = float(abs_error[patch_mask].mean()) if patch_mask.any() else np.nan

    return {
        "experiment": experiment_key,
        "description": EXPERIMENTS[experiment_key],
        "seq_len": seq_len,
        "pred_len": pred_len,
        "feature_count": len(feature_names),
        "total_metrics": total_metrics,
        "subset_rows": subset_rows,
        "patch_mae": patch_mae,
        "test_sample_index": test_sample_index,
        "test_patch_index": test_patch_index,
        "pred_test": pred_test,
        "true_test": true_test,
        "sample_hour_masks": sample_hour_masks,
    }


def plot_error_heatmaps(
    patch_meta: pd.DataFrame,
    heatmaps: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """绘制四组实验的 patch 级误差热力图。"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    vmax = max(float(np.nanmax(values)) for values in heatmaps.values())
    vmax = max(vmax, 1e-6)
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("magma")

    for axis, experiment_key in zip(axes, ["A1", "A2", "A3", "A4"]):
        frame = patch_meta.copy()
        frame["patch_mae"] = heatmaps[experiment_key]
        axis.set_facecolor("#F4F4F4")
        for row in frame.itertuples(index=False):
            rect = Rectangle(
                (row.cell_x_min, row.cell_y_min),
                getattr(row, "cell_width", row.patch_size),
                getattr(row, "cell_height", row.patch_size),
                facecolor=cmap(norm(row.patch_mae)),
                edgecolor="#555555",
                linewidth=0.45,
            )
            axis.add_patch(rect)
        axis.set_title(f"{experiment_key}: {EXPERIMENTS[experiment_key]}")
        axis.set_xlabel("grid_x")
        axis.set_ylabel("grid_y")
        set_patch_axes_limits(axis, patch_meta)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes.tolist(), label="Patch MAE", fraction=0.03, pad=0.02)
    fig.suptitle("patch-level error heatmap", fontsize=15)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def choose_representative_event_sample(
    valid_samples: np.ndarray,
    blast_strength: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> int:
    """选择测试集里最有代表性的爆破预测窗口。"""

    num_samples = len(blast_strength) - seq_len - pred_len + 1
    strength_windows = sliding_window_view(blast_strength, window_shape=pred_len, axis=0)
    strength_windows = strength_windows[seq_len : seq_len + num_samples]
    if len(valid_samples) == 0:
        return 0
    valid_strength = strength_windows[valid_samples]
    local_index = int(np.argmax(valid_strength.sum(axis=1)))
    return int(valid_samples[local_index])


def plot_representative_event(
    bundle: dict[str, Any],
    rerun_results: dict[str, dict[str, Any]],
    seq_len: int,
    pred_len: int,
    patch_meta: pd.DataFrame,
    valid_samples: np.ndarray,
    blast_strength: np.ndarray,
    out_path: Path,
) -> None:
    """绘制代表性爆破窗口的预测对比图。"""

    target = bundle["bundle"]["target"].astype(np.float32)
    blast_v3 = bundle["bundle"]["blast_v3"].astype(np.float32)[:, :, 0]
    timestamps = pd.to_datetime(bundle["bundle"]["timestamps"].astype(str))
    event_sample = choose_representative_event_sample(
        valid_samples=valid_samples,
        blast_strength=blast_strength,
        seq_len=seq_len,
        pred_len=pred_len,
    )

    patch_idx = int(np.argmax(blast_v3[event_sample + seq_len : event_sample + seq_len + pred_len].max(axis=0)))
    history = target[event_sample : event_sample + seq_len, patch_idx]
    future = target[event_sample + seq_len : event_sample + seq_len + pred_len, patch_idx]
    history_time = timestamps[event_sample : event_sample + seq_len]
    future_time = timestamps[event_sample + seq_len : event_sample + seq_len + pred_len]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(history_time, history, color="#6C757D", linewidth=2, label="历史真实值")
    ax.plot(future_time, future, color="#1D3557", linewidth=2.2, label="未来真实值")

    color_map = {"A1": "#8D99AE", "A2": "#2A9D8F", "A3": "#E76F51", "A4": "#264653"}
    for experiment_key, result in rerun_results.items():
        sample_mask = result["test_sample_index"] == event_sample
        patch_mask = result["test_patch_index"] == patch_idx
        row_index = np.where(sample_mask & patch_mask)[0]
        if len(row_index) == 0:
            continue
        ax.plot(
            future_time,
            result["pred_test"][row_index[0]],
            linewidth=1.8,
            marker="o",
            markersize=4,
            color=color_map[experiment_key],
            label=f"{experiment_key} 预测",
        )

    patch_id = patch_meta.iloc[patch_idx]["patch_id"]
    ax.axvline(history_time[-1], color="#ADB5BD", linestyle="--", linewidth=1)
    ax.set_title(f"代表性事件窗口预测图（patch={patch_id}）")
    ax.set_xlabel("时间")
    ax.set_ylabel("disp_mean")
    ax.legend(ncol=3, frameon=True)
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    """执行 A1-A4 基线实验。"""

    args = parse_args()
    config = load_config(args.config)
    setup_plot_style()
    data = load_bundle(Path(config["paths"]["dataset_dir"]))
    output_dir = Path(config["paths"]["output_dir"]) / "baseline_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = build_hour_subsets(data)
    total_rows = []
    subset_rows = []

    print("1/3 运行 A1-A4 基线实验...")
    for seq_len in config["windows"]["seq_lens"]:
        for pred_len in config["windows"]["pred_lens"]:
            for experiment_key in ["A1", "A2", "A3", "A4"]:
                result = run_single_experiment(
                    experiment_key=experiment_key,
                    seq_len=int(seq_len),
                    pred_len=int(pred_len),
                    bundle=data,
                    config=config,
                    subsets=subsets,
                )
                total_rows.append(
                    {
                        "experiment": experiment_key,
                        "description": EXPERIMENTS[experiment_key],
                        "seq_len": int(seq_len),
                        "pred_len": int(pred_len),
                        "feature_count": int(result["feature_count"]),
                        **result["total_metrics"],
                    }
                )
                subset_rows.extend(result["subset_rows"])

    total_metrics = pd.DataFrame(total_rows).sort_values(["experiment", "seq_len", "pred_len"]).reset_index(drop=True)
    subset_metrics = pd.DataFrame(subset_rows).sort_values(["experiment", "seq_len", "pred_len", "subset"]).reset_index(drop=True)
    total_metrics.to_csv(output_dir / "baseline_total_metrics_v2.csv", index=False)
    subset_metrics.to_csv(output_dir / "baseline_subset_metrics_v2.csv", index=False)

    summary_metrics = (
        total_metrics.groupby(["experiment", "description"], as_index=False)[["mae", "rmse", "mse"]]
        .mean()
        .sort_values("experiment")
        .reset_index(drop=True)
    )
    summary_metrics.to_csv(output_dir / "baseline_total_summary_v2.csv", index=False)

    print("2/3 生成热力图和代表性事件图...")
    best_metric = str(config["baseline"]["best_metric"])
    best_a4 = total_metrics.loc[total_metrics["experiment"] == "A4"].sort_values(best_metric).iloc[0]
    best_seq_len = int(best_a4["seq_len"])
    best_pred_len = int(best_a4["pred_len"])

    rerun_results = {}
    heatmaps = {}
    for experiment_key in ["A1", "A2", "A3", "A4"]:
        rerun = run_single_experiment(
            experiment_key=experiment_key,
            seq_len=best_seq_len,
            pred_len=best_pred_len,
            bundle=data,
            config=config,
            subsets=subsets,
        )
        rerun_results[experiment_key] = rerun
        heatmaps[experiment_key] = rerun["patch_mae"]

    plot_error_heatmaps(
        patch_meta=data["patch_meta"],
        heatmaps=heatmaps,
        out_path=output_dir / "patch_level_error_heatmap_v2.png",
    )

    plot_representative_event(
        bundle=data,
        rerun_results=rerun_results,
        seq_len=best_seq_len,
        pred_len=best_pred_len,
        patch_meta=data["patch_meta"],
        valid_samples=np.sort(np.unique(rerun_results["A4"]["test_sample_index"])),
        blast_strength=subsets["blast_strength"],
        out_path=output_dir / "representative_event_prediction_v2.png",
    )

    print("3/3 评估 A4 是否稳定优于 A3...")
    compare = (
        total_metrics.loc[total_metrics["experiment"].isin(["A3", "A4"])]
        .pivot(index=["seq_len", "pred_len"], columns="experiment", values=["mae", "rmse", "mse"])
        .sort_index()
    )
    mae_improvement = compare[("mae", "A3")] - compare[("mae", "A4")]
    rmse_improvement = compare[("rmse", "A3")] - compare[("rmse", "A4")]
    stable_superior = bool((mae_improvement > 0).mean() >= 0.75 and (rmse_improvement > 0).mean() >= 0.75)

    gate_summary = {
        "best_a4_seq_len": best_seq_len,
        "best_a4_pred_len": best_pred_len,
        "a4_better_than_a3_mae_ratio": float((mae_improvement > 0).mean()),
        "a4_better_than_a3_rmse_ratio": float((rmse_improvement > 0).mean()),
        "a4_stably_better_than_a3": stable_superior,
        "note": "仅在 A4 稳定优于 A3 时，才建议进入 PGGC/EDDR/PIR 机制阶段。",
    }
    (output_dir / "baseline_gate_summary_v2.json").write_text(
        json.dumps(gate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"baseline_total_metrics_v2.csv: {output_dir / 'baseline_total_metrics_v2.csv'}")
    print(f"baseline_subset_metrics_v2.csv: {output_dir / 'baseline_subset_metrics_v2.csv'}")
    print(f"最佳 A4 配置: seq_len={best_seq_len}, pred_len={best_pred_len}")
    print(f"A4 是否稳定优于 A3: {stable_superior}")


if __name__ == "__main__":
    main()
