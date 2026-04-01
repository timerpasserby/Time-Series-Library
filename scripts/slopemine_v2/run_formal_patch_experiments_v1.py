#!/usr/bin/env python3
"""这个脚本负责在冻结后的正式 split 上运行 patch 级第一轮正式实验。
相关文件：config_v2.toml、dataset_manifest_v1.md、experiment_split_plan_v1.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 兼容 Python 3.10
    import tomli as tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font")

from data_provider.slopemine_formal import (  # noqa: E402
    SlopeMineFormalDataset,
    build_feature_tensor_by_experiment,
    build_formal_window_manifest,
    build_sample_subset_masks,
    compute_feature_scaler,
    compute_subset_hour_flags,
    compute_target_scaler,
    load_formal_window_bundle,
    load_split_plan,
    resolve_formal_window_config,
)
from models.slopemine_formal_wrappers import (  # noqa: E402
    FormalPatchModel,
    count_parameters,
    masked_mse_loss,
)
from plot_utils_v2 import set_patch_axes, setup_plot_style  # noqa: E402


EXPERIMENT_DESCRIPTIONS = {
    "A1": "patch internal only",
    "A2": "A1 + weather",
    "A3": "A2 + blast V2",
    "A4": "A2 + blast V3 real-coordinate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first-round formal slopemine patch experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/slopemine_v2/config_v2.toml"),
        help="Path to the TOML config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    for key in ["dataset_dir", "output_dir"]:
        value = Path(config["paths"][key])
        config["paths"][key] = value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求使用 CUDA，但当前环境不可用。")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "histories": output_root / "histories",
        "predictions": output_root / "predictions",
        "figures": output_root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask.astype(bool)
    if not valid.any():
        return {"mae": np.nan, "rmse": np.nan, "mse": np.nan}
    error = y_pred[valid] - y_true[valid]
    mse = float(np.mean(np.square(error)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(mse)),
        "mse": mse,
    }


def write_window_build_report(
    *,
    report_path: Path,
    split_plan: pd.DataFrame,
    manifest: pd.DataFrame,
    missing_report: dict[str, Any],
    seq_len: int,
    pred_len: int,
    window_source: str,
) -> None:
    split_counts = manifest.groupby("split_name").size().to_dict()
    cross_counts = manifest.groupby("split_name")["encoder_crosses_split_start"].sum().to_dict()
    lines = [
        "# window_build_report_v1",
        "",
        "- 正式窗口严格依据 `experiment_split_plan_v1.csv` 构造，样本归属以 decoder horizon 完全落入对应 split 为准。",
        f"- 当前正式窗口配置：`seq_len={seq_len}`、`pred_len={pred_len}`，来源：`{window_source}`。",
        f"- 全局缺失小时处理策略：`{missing_report['missing_hour_policy']}`。",
        f"- 原始整轴共 `{missing_report['natural_total_steps']}` 小时，其中显式全局缺失 `{missing_report['dropped_global_missing_hours']}` 小时，正式窗口仅基于 `{missing_report['effective_total_steps']}` 个有效时间步构造。",
        f"- 全局缺失小时时间戳：`{', '.join(missing_report['dropped_global_missing_timestamps'])}`。",
        "",
        "## Split Summary",
        "",
        "| split | effective_steps | sample_count | encoder_crosses_split_start | blast_hours | rain_hours |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in split_plan.itertuples(index=False):
        lines.append(
            f"| {row.split_name} | {int(row.effective_steps)} | {int(split_counts.get(row.split_name, 0))} | {int(cross_counts.get(row.split_name, 0))} | {int(row.blast_hours)} | {int(row.rain_hours)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 本轮没有复用旧的 `window_manifest_v2_ps10.csv` 默认切分。",
            "- 由于 split 基于有效时间步，val/test 前几个窗口会使用前一段历史作为 encoder 上下文；这属于只用过去信息的滚动预测，不会读取未来标签。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataloaders(
    *,
    manifest: pd.DataFrame,
    feature_array: np.ndarray,
    bundle,
    train_end_index: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], dict[str, SlopeMineFormalDataset], dict[str, np.ndarray], dict[str, np.ndarray]]:
    feature_mean, feature_std = compute_feature_scaler(feature_array, bundle.input_mask, train_end_index)
    target_mean, target_std = compute_target_scaler(bundle.target, bundle.target_mask, train_end_index)

    datasets: dict[str, SlopeMineFormalDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for split_name, shuffle in [("train", True), ("val", False), ("test", False)]:
        split_manifest = manifest.loc[manifest["split_name"] == split_name].sort_values("sample_id").reset_index(drop=True)
        dataset = SlopeMineFormalDataset(
            manifest=split_manifest,
            feature_array=feature_array,
            target_array=bundle.target,
            input_mask=bundle.input_mask,
            target_mask=bundle.target_mask,
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        datasets[split_name] = dataset
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders, datasets, {"feature_mean": feature_mean, "feature_std": feature_std}, {"target_mean": target_mean, "target_std": target_std}


def expand_subset_lookup(manifest: pd.DataFrame, subset_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """把仅含已接纳窗口的子集掩码扩成按全局 sample_id 索引的查找表。"""

    max_sample_id = int(manifest["sample_id"].max()) + 1
    sample_ids = manifest["sample_id"].to_numpy(dtype=np.int32)
    lookup: dict[str, np.ndarray] = {}
    for subset_name, values in subset_masks.items():
        expanded = np.zeros((max_sample_id, values.shape[1]), dtype=np.uint8)
        expanded[sample_ids] = values
        lookup[subset_name] = expanded
    return lookup


def train_one_epoch(
    model: FormalPatchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        enc_mask = batch["enc_mask"].to(device)
        y = batch["y"].to(device)
        y_mask = batch["y_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x, enc_mask)
        loss = masked_mse_loss(pred, y, y_mask)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_model(
    model: FormalPatchModel,
    loader: DataLoader,
    device: torch.device,
    target_scaler: dict[str, np.ndarray],
    subset_lookup: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    model.eval()
    target_mean = torch.from_numpy(target_scaler["target_mean"]).to(device)
    target_std = torch.from_numpy(target_scaler["target_std"]).to(device)

    sample_ids: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        enc_mask = batch["enc_mask"].to(device)
        y_raw = batch["y_raw"].to(device)
        y_mask = batch["y_mask"].to(device)

        pred_norm = model(x, enc_mask)
        pred_raw = pred_norm * target_std[None, None, :] + target_mean[None, None, :]

        sample_ids.append(batch["sample_id"].cpu().numpy())
        predictions.append(pred_raw.cpu().numpy())
        targets.append(y_raw.cpu().numpy())
        masks.append(y_mask.cpu().numpy())

    sample_ids_array = np.concatenate(sample_ids, axis=0)
    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    mask_array = np.concatenate(masks, axis=0)

    total_metrics = compute_metrics(target_array, pred_array, mask_array)
    subset_metrics: dict[str, dict[str, float]] = {}
    if subset_lookup is not None:
        for subset_name, lookup in subset_lookup.items():
            hour_mask = lookup[sample_ids_array]
            subset_mask = mask_array * hour_mask[:, :, None]
            subset_metrics[subset_name] = compute_metrics(target_array, pred_array, subset_mask)

    abs_error = np.abs(pred_array - target_array) * mask_array
    denom = mask_array.sum(axis=(0, 1))
    patch_mae = np.divide(
        abs_error.sum(axis=(0, 1)),
        denom,
        out=np.full(mask_array.shape[-1], np.nan, dtype=np.float32),
        where=denom > 0,
    ).astype(np.float32)

    return {
        "sample_ids": sample_ids_array.astype(np.int32),
        "predictions": pred_array.astype(np.float32),
        "targets": target_array.astype(np.float32),
        "masks": mask_array.astype(np.float32),
        "total_metrics": total_metrics,
        "subset_metrics": subset_metrics,
        "patch_mae": patch_mae,
    }


def save_prediction_bundle(path: Path, evaluation: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        sample_ids=evaluation["sample_ids"],
        predictions=evaluation["predictions"],
        targets=evaluation["targets"],
        masks=evaluation["masks"],
        patch_mae=evaluation["patch_mae"],
    )


def build_compare_tables(results_main: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    best_rows = (
        results_main.sort_values(["experiment", "val_mae", "test_mae", "model_name"])
        .groupby("experiment", as_index=False)
        .first()
        .sort_values("experiment")
        .reset_index(drop=True)
    )
    lines = [
        "# results_compare_A1_A4",
        "",
        "| experiment | description | best_model | val_mae | val_rmse | test_mae | test_rmse | seq_len | pred_len | feature_count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_rows.itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.description} | {row.model_name} | {row.val_mae:.6e} | {row.val_rmse:.6e} | {row.test_mae:.6e} | {row.test_rmse:.6e} | {int(row.seq_len)} | {int(row.pred_len)} | {int(row.feature_count)} |"
        )
    return best_rows, "\n".join(lines) + "\n"


def plot_blast_hours_compare(subset_results: pd.DataFrame, output_path: Path) -> None:
    plot_df = subset_results.loc[
        (subset_results["split"] == "test") & (subset_results["subset"] == "blast_hours") & (subset_results["experiment"].isin(["A3", "A4"]))
    ].copy()
    if plot_df.empty:
        return
    models = sorted(plot_df["model_name"].unique().tolist())
    x = np.arange(len(models))
    width = 0.35

    a3 = plot_df.loc[plot_df["experiment"] == "A3"].set_index("model_name").reindex(models)
    a4 = plot_df.loc[plot_df["experiment"] == "A4"].set_index("model_name").reindex(models)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, a3["mae"].to_numpy(dtype=float), width=width, color="#E76F51", label="A3")
    ax.bar(x + width / 2, a4["mae"].to_numpy(dtype=float), width=width, color="#264653", label="A4")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("MAE")
    ax.set_title("A3 vs A4 on blast hours")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def select_event_windows(
    test_manifest: pd.DataFrame,
    blast_strength: np.ndarray,
    top_k: int,
) -> list[int]:
    rows: list[tuple[int, float]] = []
    for row in test_manifest.itertuples(index=False):
        score = float(blast_strength[int(row.decoder_start_index) : int(row.decoder_end_index_exclusive)].sum())
        rows.append((int(row.sample_id), score))
    rows.sort(key=lambda item: (item[1], item[0]), reverse=True)
    chosen = [sample_id for sample_id, score in rows if score > 0][:top_k]
    if len(chosen) < top_k:
        fallback = [sample_id for sample_id, _ in rows if sample_id not in chosen]
        chosen.extend(fallback[: max(0, top_k - len(chosen))])
    return chosen[:top_k]


def load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def plot_typical_event_windows(
    *,
    output_path: Path,
    bundle,
    manifest_test: pd.DataFrame,
    sample_ids: list[int],
    a3_pred_path: Path,
    a4_pred_path: Path,
) -> None:
    a3 = load_prediction_bundle(a3_pred_path)
    a4 = load_prediction_bundle(a4_pred_path)
    a3_lookup = {int(sample_id): idx for idx, sample_id in enumerate(a3["sample_ids"].tolist())}
    a4_lookup = {int(sample_id): idx for idx, sample_id in enumerate(a4["sample_ids"].tolist())}

    timestamps = pd.to_datetime(bundle.timestamps)
    blast_strength = bundle.blast_v3[:, :, 0]

    fig, axes = plt.subplots(len(sample_ids), 1, figsize=(12, 4.6 * len(sample_ids)), squeeze=False)
    axes = axes[:, 0]
    for axis, sample_id in zip(axes, sample_ids):
        row = manifest_test.loc[manifest_test["sample_id"] == sample_id].iloc[0]
        decoder_slice = slice(int(row.decoder_start_index), int(row.decoder_end_index_exclusive))
        encoder_slice = slice(int(row.encoder_start_index), int(row.encoder_end_index_exclusive))

        a3_idx = a3_lookup[int(sample_id)]
        a4_idx = a4_lookup[int(sample_id)]
        patch_idx = int(np.argmax(blast_strength[decoder_slice].sum(axis=0)))

        history = bundle.target[encoder_slice, patch_idx]
        future = a4["targets"][a4_idx, :, patch_idx]
        pred_a3 = a3["predictions"][a3_idx, :, patch_idx]
        pred_a4 = a4["predictions"][a4_idx, :, patch_idx]

        history_time = timestamps[encoder_slice]
        future_time = timestamps[decoder_slice]

        axis.plot(history_time, history, color="#6C757D", linewidth=2, label="history true")
        axis.plot(future_time, future, color="#1D3557", linewidth=2.1, label="future true")
        axis.plot(future_time, pred_a3, color="#E76F51", linewidth=1.8, marker="o", markersize=3, label="A3 pred")
        axis.plot(future_time, pred_a4, color="#264653", linewidth=1.8, marker="s", markersize=3, label="A4 pred")
        axis.axvline(history_time[-1], color="#ADB5BD", linestyle="--", linewidth=1)
        axis.set_title(f"sample={sample_id}, patch={bundle.patch_ids[patch_idx]}")
        axis.set_ylabel("disp_mean")
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("timestamp")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_patch_mae_heatmap(
    *,
    patch_meta: pd.DataFrame,
    patch_mae: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    frame = patch_meta.copy()
    frame["patch_mae"] = patch_mae
    vmax = max(float(np.nanmax(frame["patch_mae"])), 1e-10)
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("magma")

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.set_facecolor("#F3F3F3")
    for row in frame.itertuples(index=False):
        color = "#D9D9D9" if np.isnan(row.patch_mae) else cmap(norm(row.patch_mae))
        rect = Rectangle(
            (row.cell_x_min, row.cell_y_min),
            getattr(row, "cell_width", 10.0),
            getattr(row, "cell_height", 10.0),
            facecolor=color,
            edgecolor="#666666",
            linewidth=0.35,
        )
        ax.add_patch(rect)
    set_patch_axes(ax, frame)
    ax.set_title(title)
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Patch MAE", fraction=0.04, pad=0.03)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_experiment_log(
    *,
    config: dict[str, Any],
    split_plan: pd.DataFrame,
    window_info: dict[str, Any],
    missing_report: dict[str, Any],
    results_main: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> str:
    best_a4 = results_main.loc[results_main["experiment"] == "A4"].sort_values(["val_mae", "test_mae"]).iloc[0]
    lines = [
        "# experiment_log_A1_A4",
        "",
        "## Run Context",
        "",
        f"- 正式 split 文件：`outputs/slopemine_v2/experiment_plan_v1/experiment_split_plan_v1.csv`。",
        f"- 选定方案：`{split_plan['selected_candidate_id'].iloc[0]}`。",
        f"- 正式窗口配置：`seq_len={window_info['seq_len']}`、`pred_len={window_info['pred_len']}`，来源：`{window_info['source']}`。",
        f"- 缺失小时策略：`{missing_report['missing_hour_policy']}`，显式剔除 `{missing_report['dropped_global_missing_hours']}` 个全局缺失小时。",
        f"- 训练设备：`{config['formal_round1']['resolved_device']}`。",
        f"- 统一训练配置：`epochs={config['formal_round1']['train_epochs']}`、`batch_size={config['formal_round1']['batch_size']}`、`lr={config['formal_round1']['learning_rate']}`、`weight_decay={config['formal_round1']['weight_decay']}`。",
        "",
        "## Best A4 Run",
        "",
        f"- 最优 A4 组合：`{best_a4.model_name}`。",
        f"- `val_mae={best_a4.val_mae:.6e}`，`test_mae={best_a4.test_mae:.6e}`。",
        f"- checkpoint：`{best_a4.checkpoint_path}`。",
        "",
        "## Outputs",
        "",
        f"- 结果总表：`{output_dirs['root'] / 'results_main_A1_A4.csv'}`。",
        f"- 子集结果：`{output_dirs['root'] / 'results_subsets_A1_A4.csv'}`。",
        f"- 图表目录：`{output_dirs['figures']}`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_plot_style()
    set_seed(int(config["formal_round1"]["seed"]))

    device = resolve_device(str(config["formal_round1"]["device"]))
    config["formal_round1"]["resolved_device"] = str(device)

    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_root = Path(config["paths"]["output_dir"]) / str(config["formal_round1"]["output_subdir"])
    output_dirs = ensure_output_dirs(output_root)

    split_plan = load_split_plan(Path(config["paths"]["output_dir"]) / "experiment_plan_v1" / "experiment_split_plan_v1.csv")
    bundle, patch_meta, missing_report = load_formal_window_bundle(
        dataset_dir,
        missing_hour_policy=str(config["formal_round1"]["missing_hour_policy"]),
    )
    window_info = resolve_formal_window_config(config, REPO_ROOT)
    seq_len = int(window_info["seq_len"])
    pred_len = int(window_info["pred_len"])

    manifest = build_formal_window_manifest(
        bundle.timestamps,
        split_plan,
        seq_len=seq_len,
        pred_len=pred_len,
        missing_hour_policy=str(config["formal_round1"]["missing_hour_policy"]),
    )
    manifest_path = dataset_dir / "window_manifest_v1_ps10.csv"
    manifest.to_csv(manifest_path, index=False)
    write_window_build_report(
        report_path=output_root / "window_build_report_v1.md",
        split_plan=split_plan,
        manifest=manifest,
        missing_report=missing_report,
        seq_len=seq_len,
        pred_len=pred_len,
        window_source=window_info["source"],
    )

    subset_flags = compute_subset_hour_flags(bundle)
    subset_masks_all = expand_subset_lookup(
        manifest,
        build_sample_subset_masks(manifest, subset_flags, pred_len),
    )

    train_end_index = int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0])

    main_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []

    model_names = [str(item) for item in config["formal_round1"]["models"]]
    experiments = [str(item) for item in config["formal_round1"]["experiments"]]

    for experiment_key in experiments:
        feature_array, feature_names = build_feature_tensor_by_experiment(bundle, experiment_key)
        loaders, datasets, _feature_scaler, target_scaler = build_dataloaders(
            manifest=manifest,
            feature_array=feature_array,
            bundle=bundle,
            train_end_index=train_end_index,
            batch_size=int(config["formal_round1"]["batch_size"]),
            num_workers=int(config["formal_round1"]["num_workers"]),
        )
        split_sizes = {split_name: len(dataset) for split_name, dataset in datasets.items()}

        for model_name in model_names:
            model = FormalPatchModel(
                model_name=model_name,
                feature_dim=feature_array.shape[-1],
                num_nodes=feature_array.shape[1],
                seq_len=seq_len,
                pred_len=pred_len,
                model_cfg=dict(config["formal_round1"]["model"]),
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(config["formal_round1"]["learning_rate"]),
                weight_decay=float(config["formal_round1"]["weight_decay"]),
            )

            history_rows: list[dict[str, Any]] = []
            best_val_mae = float("inf")
            best_epoch = 0
            checkpoint_path = output_dirs["checkpoints"] / f"{experiment_key}_{model_name}_best.pt"
            history_path = output_dirs["histories"] / f"{experiment_key}_{model_name}_history.csv"
            prediction_val_path = output_dirs["predictions"] / f"{experiment_key}_{model_name}_val.npz"
            prediction_test_path = output_dirs["predictions"] / f"{experiment_key}_{model_name}_test.npz"

            for epoch in range(1, int(config["formal_round1"]["train_epochs"]) + 1):
                train_loss = train_one_epoch(
                    model=model,
                    loader=loaders["train"],
                    optimizer=optimizer,
                    device=device,
                    grad_clip=float(config["formal_round1"]["grad_clip"]),
                )
                val_eval = evaluate_model(
                    model=model,
                    loader=loaders["val"],
                    device=device,
                    target_scaler=target_scaler,
                    subset_lookup=subset_masks_all,
                )
                history_row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mae": val_eval["total_metrics"]["mae"],
                    "val_rmse": val_eval["total_metrics"]["rmse"],
                    "val_mse": val_eval["total_metrics"]["mse"],
                }
                history_rows.append(history_row)

                if history_row["val_mae"] < best_val_mae:
                    best_val_mae = float(history_row["val_mae"])
                    best_epoch = int(epoch)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "experiment": experiment_key,
                            "model_name": model_name,
                            "seq_len": seq_len,
                            "pred_len": pred_len,
                            "feature_names": feature_names,
                            "target_mean": target_scaler["target_mean"],
                            "target_std": target_scaler["target_std"],
                        },
                        checkpoint_path,
                    )
                    save_prediction_bundle(prediction_val_path, val_eval)

            pd.DataFrame(history_rows).to_csv(history_path, index=False)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            val_eval = evaluate_model(
                model=model,
                loader=loaders["val"],
                device=device,
                target_scaler=target_scaler,
                subset_lookup=subset_masks_all,
            )
            test_eval = evaluate_model(
                model=model,
                loader=loaders["test"],
                device=device,
                target_scaler=target_scaler,
                subset_lookup=subset_masks_all,
            )
            save_prediction_bundle(prediction_val_path, val_eval)
            save_prediction_bundle(prediction_test_path, test_eval)

            main_rows.append(
                {
                    "experiment": experiment_key,
                    "description": EXPERIMENT_DESCRIPTIONS[experiment_key],
                    "model_name": model_name,
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "feature_count": len(feature_names),
                    "num_parameters": count_parameters(model),
                    "best_epoch": best_epoch,
                    "train_samples": split_sizes["train"],
                    "val_samples": split_sizes["val"],
                    "test_samples": split_sizes["test"],
                    "val_mae": val_eval["total_metrics"]["mae"],
                    "val_rmse": val_eval["total_metrics"]["rmse"],
                    "val_mse": val_eval["total_metrics"]["mse"],
                    "test_mae": test_eval["total_metrics"]["mae"],
                    "test_rmse": test_eval["total_metrics"]["rmse"],
                    "test_mse": test_eval["total_metrics"]["mse"],
                    "checkpoint_path": str(checkpoint_path),
                    "history_path": str(history_path),
                    "prediction_val_path": str(prediction_val_path),
                    "prediction_test_path": str(prediction_test_path),
                }
            )

            for split_name, evaluation in [("val", val_eval), ("test", test_eval)]:
                for subset_name, metrics in evaluation["subset_metrics"].items():
                    subset_rows.append(
                        {
                            "experiment": experiment_key,
                            "description": EXPERIMENT_DESCRIPTIONS[experiment_key],
                            "model_name": model_name,
                            "split": split_name,
                            "subset": subset_name,
                            "seq_len": seq_len,
                            "pred_len": pred_len,
                            **metrics,
                        }
                    )

    results_main = pd.DataFrame(main_rows).sort_values(["experiment", "model_name"]).reset_index(drop=True)
    results_subsets = pd.DataFrame(subset_rows).sort_values(["experiment", "model_name", "split", "subset"]).reset_index(drop=True)
    results_main.to_csv(output_root / "results_main_A1_A4.csv", index=False)
    results_subsets.to_csv(output_root / "results_subsets_A1_A4.csv", index=False)

    compare_df, compare_md = build_compare_tables(results_main)
    compare_df.to_csv(output_root / "results_compare_A1_A4.csv", index=False)
    (output_root / "results_compare_A1_A4.md").write_text(compare_md, encoding="utf-8")

    plot_blast_hours_compare(
        subset_results=results_subsets,
        output_path=output_dirs["figures"] / "a3_vs_a4_blast_hours_mae.png",
    )

    best_a4 = results_main.loc[results_main["experiment"] == "A4"].sort_values(["val_mae", "test_mae"]).iloc[0]
    best_model_name = str(best_a4["model_name"])
    a3_best_row = results_main.loc[
        (results_main["experiment"] == "A3") & (results_main["model_name"] == best_model_name)
    ].sort_values(["val_mae", "test_mae"]).iloc[0]
    manifest_test = manifest.loc[manifest["split_name"] == "test"].sort_values("sample_id").reset_index(drop=True)
    selected_samples = select_event_windows(
        test_manifest=manifest_test,
        blast_strength=subset_flags["blast_strength"],
        top_k=int(config["formal_round1"]["plot_event_windows"]),
    )
    plot_typical_event_windows(
        output_path=output_dirs["figures"] / "typical_event_windows_A3_A4.png",
        bundle=bundle,
        manifest_test=manifest_test,
        sample_ids=selected_samples,
        a3_pred_path=Path(a3_best_row["prediction_test_path"]),
        a4_pred_path=Path(best_a4["prediction_test_path"]),
    )

    a3_test = load_prediction_bundle(Path(a3_best_row["prediction_test_path"]))
    a4_test = load_prediction_bundle(Path(best_a4["prediction_test_path"]))
    plot_patch_mae_heatmap(
        patch_meta=patch_meta,
        patch_mae=a3_test["patch_mae"],
        title=f"A3 patch MAE heatmap ({best_model_name})",
        output_path=output_dirs["figures"] / "patch_mae_heatmap_A3.png",
    )
    plot_patch_mae_heatmap(
        patch_meta=patch_meta,
        patch_mae=a4_test["patch_mae"],
        title=f"A4 patch MAE heatmap ({best_model_name})",
        output_path=output_dirs["figures"] / "patch_mae_heatmap_A4.png",
    )

    log_text = build_experiment_log(
        config=config,
        split_plan=split_plan,
        window_info=window_info,
        missing_report=missing_report,
        results_main=results_main,
        output_dirs=output_dirs,
    )
    (output_root / "experiment_log_A1_A4.md").write_text(log_text, encoding="utf-8")

    print(f"window_manifest_v1_ps10.csv: {manifest_path}")
    print(f"results_main_A1_A4.csv: {output_root / 'results_main_A1_A4.csv'}")
    print(f"results_subsets_A1_A4.csv: {output_root / 'results_subsets_A1_A4.csv'}")
    print(f"experiment_log_A1_A4.md: {output_root / 'experiment_log_A1_A4.md'}")


if __name__ == "__main__":
    main()
