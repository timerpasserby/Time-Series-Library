#!/usr/bin/env python3
"""运行冻结数据版本上的 MS-TimeFilter 正式实验链（A2~C3）。
相关文件：config_v2.toml、data_provider/slopemine_formal.py、models/ms_timefilter.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_provider.slopemine_formal import (  # noqa: E402
    build_feature_tensor_by_experiment,
    build_formal_window_manifest,
    build_multibranch_dataloaders,
    build_sample_subset_masks,
    compute_subset_hour_flags,
    load_formal_window_bundle,
    load_split_plan,
)
from models.ms_timefilter import MSTimeFilter  # noqa: E402
from models.slopemine_formal_wrappers import FormalPatchModel, count_parameters, masked_mse_loss  # noqa: E402
from plot_utils_v2 import setup_plot_style  # noqa: E402
from run_formal_patch_experiments_v1 import (  # noqa: E402
    build_dataloaders,
    compute_metrics,
    expand_subset_lookup,
    load_config,
    plot_patch_mae_heatmap,
    resolve_device,
    save_prediction_bundle,
    select_event_windows,
    set_seed,
)


DESCRIPTION_MAP = {
    "baseline_lstm_a4_96": "A4-LSTM reference (seq96)",
    "baseline_raw_timefilter_a4_96": "A4 raw TimeFilter reference (seq96)",
    "baseline_naive_weather_concat": "naive weather concat control",
    "A2": "MS-TimeFilter A2: internal + weather branch",
    "A3": "MS-TimeFilter A3: A2 + blast V2 branch",
    "A4": "MS-TimeFilter A4: A2 + blast V3 real-coordinate branch",
    "C1": "MS-TimeFilter C1: A4 + PGGC",
    "C2": "MS-TimeFilter C2: C1 + EDDR",
    "C3": "MS-TimeFilter C3: C2 + PIR",
}


def save_evaluation_bundle(path: Path, evaluation: dict[str, Any]) -> None:
    save_data = {
        "total_metrics_mae": float(evaluation["total_metrics"]["mae"]),
        "total_metrics_rmse": float(evaluation["total_metrics"]["rmse"]),
        "total_metrics_mse": float(evaluation["total_metrics"]["mse"]),
        "patch_mae": evaluation["patch_mae"],
    }
    if "gate_mean" in evaluation:
        save_data["gate_mean"] = evaluation["gate_mean"]
    if "subset_metrics" in evaluation:
        for subset_name, metrics in evaluation["subset_metrics"].items():
            for metric_name, value in metrics.items():
                save_data[f"subset_{subset_name}_{metric_name}"] = float(value)
    np.savez_compressed(path, **save_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MS-TimeFilter formal experiments on the frozen slopemine split.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/slopemine_v2/config_v2.toml"),
        help="Path to the TOML config file.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "histories": output_root / "histories",
        "predictions": output_root / "predictions",
        "figures": output_root / "figures",
        "debug": output_root / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_window_report(
    *,
    report_path: Path,
    split_plan: pd.DataFrame,
    manifest: pd.DataFrame,
    missing_report: dict[str, Any],
    seq_len: int,
    pred_len: int,
) -> None:
    split_counts = manifest.groupby("split_name").size().to_dict()
    lines = [
        "# window_build_report_ms_timefilter",
        "",
        "- 本轮窗口严格依据 `experiment_split_plan_v1.csv` 构造，decoder horizon 必须完整落入 split。",
        f"- 固定窗口配置：`seq_len={seq_len}`、`pred_len={pred_len}`。",
        f"- 缺失小时处理策略：`{missing_report['missing_hour_policy']}`。",
        f"- 整轴 720 小时中显式剔除 `{missing_report['dropped_global_missing_hours']}` 个全局缺失小时，正式有效时间步为 `{missing_report['effective_total_steps']}`。",
        "",
        "| split | effective_steps | sample_count | blast_hours | rain_hours |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in split_plan.itertuples(index=False):
        lines.append(
            f"| {row.split_name} | {int(row.effective_steps)} | {int(split_counts.get(row.split_name, 0))} | {int(row.blast_hours)} | {int(row.rain_hours)} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_physical_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    alpha_v: float,
    alpha_a: float,
) -> torch.Tensor:
    velocity_pred = pred_raw[:, 1:, :] - pred_raw[:, :-1, :]
    velocity_true = target_raw[:, 1:, :] - target_raw[:, :-1, :]
    velocity_mask = target_mask[:, 1:, :] * target_mask[:, :-1, :]
    loss_v = masked_mse_loss(velocity_pred, velocity_true, velocity_mask)

    accel_pred = velocity_pred[:, 1:, :] - velocity_pred[:, :-1, :]
    accel_true = velocity_true[:, 1:, :] - velocity_true[:, :-1, :]
    accel_mask = velocity_mask[:, 1:, :] * velocity_mask[:, :-1, :]
    loss_a = masked_mse_loss(accel_pred, accel_true, accel_mask)
    return float(alpha_v) * loss_v + float(alpha_a) * loss_a


def train_baseline_one_epoch(
    model: FormalPatchModel,
    loader,
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
def evaluate_baseline_model(
    model: FormalPatchModel,
    loader,
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

    return finalize_evaluation(sample_ids, predictions, targets, masks, subset_lookup)


def train_ms_one_epoch(
    model: MSTimeFilter,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_scaler: dict[str, np.ndarray],
    grad_clip: float,
    pir_cfg: dict[str, Any],
) -> dict[str, float]:
    model.train()
    target_mean = torch.from_numpy(target_scaler["target_mean"]).to(device)
    target_std = torch.from_numpy(target_scaler["target_std"]).to(device)

    loss_total_list: list[float] = []
    loss_pred_list: list[float] = []
    loss_phy_list: list[float] = []
    loss_moe_list: list[float] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch, return_aux=False)
        pred_norm = outputs["pred_norm"]
        loss_pred = masked_mse_loss(pred_norm, batch["y"], batch["y_mask"])

        pred_raw = pred_norm * target_std[None, None, :] + target_mean[None, None, :]
        if model.use_pir:
            loss_phy = compute_physical_loss(
                pred_raw,
                batch["y_raw"],
                batch["y_mask"],
                alpha_v=float(pir_cfg["alpha_v"]),
                alpha_a=float(pir_cfg["alpha_a"]),
            )
            loss_total = loss_pred + float(pir_cfg["beta"]) * loss_phy
        else:
            loss_phy = pred_norm.new_tensor(0.0)
            loss_total = loss_pred

        loss_total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        moe_loss = outputs["aux"]["loss_moe"]
        moe_value = float(moe_loss.item()) if torch.is_tensor(moe_loss) else float(moe_loss)
        loss_total_list.append(float(loss_total.item()))
        loss_pred_list.append(float(loss_pred.item()))
        loss_phy_list.append(float(loss_phy.item()))
        loss_moe_list.append(moe_value)

    return {
        "loss_total": float(np.mean(loss_total_list)) if loss_total_list else float("nan"),
        "loss_pred": float(np.mean(loss_pred_list)) if loss_pred_list else float("nan"),
        "loss_phy": float(np.mean(loss_phy_list)) if loss_phy_list else float("nan"),
        "loss_moe": float(np.mean(loss_moe_list)) if loss_moe_list else float("nan"),
    }


def finalize_evaluation(
    sample_ids: list[np.ndarray],
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    masks: list[np.ndarray],
    subset_lookup: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
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


@torch.no_grad()
def evaluate_ms_model(
    model: MSTimeFilter,
    loader,
    device: torch.device,
    target_scaler: dict[str, np.ndarray],
    subset_lookup: dict[str, np.ndarray] | None = None,
    capture_gate_mean: bool = False,
) -> dict[str, Any]:
    model.eval()
    target_mean = torch.from_numpy(target_scaler["target_mean"]).to(device)
    target_std = torch.from_numpy(target_scaler["target_std"]).to(device)

    sample_ids: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    gate_means: list[np.ndarray] = []

    for batch in loader:
        batch_device = move_batch_to_device(batch, device)
        outputs = model(batch_device, return_aux=capture_gate_mean)
        pred_norm = outputs["pred_norm"]
        pred_raw = pred_norm * target_std[None, None, :] + target_mean[None, None, :]

        sample_ids.append(batch["sample_id"].cpu().numpy())
        predictions.append(pred_raw.cpu().numpy())
        targets.append(batch["y_raw"].cpu().numpy())
        masks.append(batch["y_mask"].cpu().numpy())

        gate_weights = outputs["aux"].get("gate_weights")
        if capture_gate_mean and gate_weights is not None:
            gate_means.append(gate_weights.mean(dim=1).cpu().numpy())

    result = finalize_evaluation(sample_ids, predictions, targets, masks, subset_lookup)
    if gate_means:
        result["gate_mean"] = np.concatenate(gate_means, axis=0).astype(np.float32)
    return result


def get_sample_batch(dataset, sample_id: int) -> dict[str, torch.Tensor]:
    match = dataset.manifest.index[dataset.manifest["sample_id"] == int(sample_id)]
    if len(match) == 0:
        raise KeyError(f"sample_id={sample_id} 不在当前 dataset 中。")
    sample = dataset[int(match[0])]
    batch: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
    return batch


def load_model_for_inference(
    spec: dict[str, Any],
    checkpoint_path: Path,
    config: dict[str, Any],
    bundle,
    patch_meta: pd.DataFrame,
    device: torch.device,
) -> MSTimeFilter:
    model = MSTimeFilter(
        experiment_id=str(spec["experiment_id"]),
        patch_meta=patch_meta,
        seq_len=int(config["seq_len"]),
        pred_len=int(config["pred_len"]),
        patch_count=bundle.target.shape[1],
        internal_dim=bundle.internal.shape[-1],
        weather_dim=bundle.weather.shape[-1],
        blast_v2_dim=bundle.blast_v2.shape[-1],
        blast_v3_dim=bundle.blast_v3.shape[-1],
        model_cfg=dict(config["model"]),
        graph_cfg=dict(config["graph"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_gate_subset_frame_from_saved(
    gate_mean: np.ndarray,
    subset_lookup: dict[str, np.ndarray],
    eval_data,
) -> pd.DataFrame:
    mean_values = gate_mean.mean(axis=0)
    rows = [
        {
            "subset": "all",
            "sample_count": int(gate_mean.shape[0]),
            "spatial_mean": float(mean_values[0]),
            "temporal_mean": float(mean_values[1]),
            "spatiotemporal_mean": float(mean_values[2]),
        }
    ]
    return pd.DataFrame(rows)


def train_baseline_spec(
    *,
    spec: dict[str, Any],
    manifest: pd.DataFrame,
    bundle,
    split_plan: pd.DataFrame,
    output_dirs: dict[str, Path],
    config: dict[str, Any],
    device: torch.device,
    subset_lookup: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_array, feature_names = build_feature_tensor_by_experiment(bundle, str(spec["feature_key"]))
    train_end_index = int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0])
    loaders, datasets, _feature_scaler, target_scaler = build_dataloaders(
        manifest=manifest,
        feature_array=feature_array,
        bundle=bundle,
        train_end_index=train_end_index,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )
    model_cfg = dict(config["baseline_model"])
    model_cfg["dropout"] = float(spec["dropout"])
    model = FormalPatchModel(
        model_name=str(spec["model_name"]),
        feature_dim=feature_array.shape[-1],
        num_nodes=feature_array.shape[1],
        seq_len=int(config["seq_len"]),
        pred_len=int(config["pred_len"]),
        model_cfg=model_cfg,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    stem = f"{spec['experiment_id']}_{spec['branch_name']}"
    checkpoint_path = output_dirs["checkpoints"] / f"{stem}.pt"
    history_path = output_dirs["histories"] / f"{stem}.csv"
    prediction_val_path = output_dirs["predictions"] / f"{stem}_val.npz"
    prediction_test_path = output_dirs["predictions"] / f"{stem}_test.npz"

    history_rows: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, int(config["train_epochs"]) + 1):
        train_loss = train_baseline_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            grad_clip=float(config["grad_clip"]),
        )
        val_eval = evaluate_baseline_model(
            model=model,
            loader=loaders["val"],
            device=device,
            target_scaler=target_scaler,
            subset_lookup=subset_lookup,
        )
        history_row = {
            "epoch": epoch,
            "loss_total": train_loss,
            "loss_pred": train_loss,
            "loss_phy": np.nan,
            "loss_dyn": np.nan,
            "loss_imp": np.nan,
            "loss_moe": np.nan,
            "val_mae": float(val_eval["total_metrics"]["mae"]),
        }
        history_rows.append(history_row)
        if history_row["val_mae"] < best_val_mae:
            best_val_mae = float(history_row["val_mae"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "spec": spec,
                    "feature_names": feature_names,
                },
                checkpoint_path,
            )

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_eval = evaluate_baseline_model(
        model=model,
        loader=loaders["val"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
    )
    test_eval = evaluate_baseline_model(
        model=model,
        loader=loaders["test"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
    )
    save_prediction_bundle(prediction_val_path, val_eval)
    save_prediction_bundle(prediction_test_path, test_eval)

    main_row = {
        "experiment_id": str(spec["experiment_id"]),
        "branch_name": str(spec["branch_name"]),
        "description": DESCRIPTION_MAP[str(spec["experiment_id"])],
        "model_name": str(spec["model_name"]),
        "seq_len": int(config["seq_len"]),
        "pred_len": int(config["pred_len"]),
        "learning_rate": float(spec["learning_rate"]),
        "dropout": float(spec["dropout"]),
        "best_epoch": int(best_epoch),
        "train_samples": int(len(datasets["train"])),
        "val_samples": int(len(datasets["val"])),
        "test_samples": int(len(datasets["test"])),
        "num_parameters": count_parameters(model),
        "feature_count": int(feature_array.shape[-1]),
        "val_mae": float(val_eval["total_metrics"]["mae"]),
        "val_rmse": float(val_eval["total_metrics"]["rmse"]),
        "val_mse": float(val_eval["total_metrics"]["mse"]),
        "test_mae": float(test_eval["total_metrics"]["mae"]),
        "test_rmse": float(test_eval["total_metrics"]["rmse"]),
        "test_mse": float(test_eval["total_metrics"]["mse"]),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "prediction_val_path": str(prediction_val_path),
        "prediction_test_path": str(prediction_test_path),
    }

    subset_rows: list[dict[str, Any]] = []
    for split_name, evaluation in [("val", val_eval), ("test", test_eval)]:
        for subset_name, metrics in evaluation["subset_metrics"].items():
            subset_rows.append(
                {
                    "experiment_id": str(spec["experiment_id"]),
                    "branch_name": str(spec["branch_name"]),
                    "description": DESCRIPTION_MAP[str(spec["experiment_id"])],
                    "model_name": str(spec["model_name"]),
                    "split": split_name,
                    "subset": subset_name,
                    "seq_len": int(config["seq_len"]),
                    "pred_len": int(config["pred_len"]),
                    **metrics,
                }
            )
    return main_row, subset_rows


def train_ms_spec(
    *,
    spec: dict[str, Any],
    manifest: pd.DataFrame,
    bundle,
    patch_meta: pd.DataFrame,
    split_plan: pd.DataFrame,
    output_dirs: dict[str, Path],
    config: dict[str, Any],
    device: torch.device,
    subset_lookup: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_end_index = int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0])
    loaders, datasets, _feature_scalers, target_scaler = build_multibranch_dataloaders(
        manifest=manifest,
        bundle=bundle,
        train_end_index=train_end_index,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )
    model = MSTimeFilter(
        experiment_id=str(spec["experiment_id"]),
        patch_meta=patch_meta,
        seq_len=int(config["seq_len"]),
        pred_len=int(config["pred_len"]),
        patch_count=bundle.target.shape[1],
        internal_dim=bundle.internal.shape[-1],
        weather_dim=bundle.weather.shape[-1],
        blast_v2_dim=bundle.blast_v2.shape[-1],
        blast_v3_dim=bundle.blast_v3.shape[-1],
        model_cfg=dict(config["model"]),
        graph_cfg=dict(config["graph"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    stem = f"{spec['experiment_id']}_{spec['branch_name']}"
    checkpoint_path = output_dirs["checkpoints"] / f"{stem}.pt"
    history_path = output_dirs["histories"] / f"{stem}.csv"
    prediction_val_path = output_dirs["predictions"] / f"{stem}_val.npz"
    prediction_test_path = output_dirs["predictions"] / f"{stem}_test.npz"

    history_rows: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, int(config["train_epochs"]) + 1):
        train_losses = train_ms_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            target_scaler=target_scaler,
            grad_clip=float(config["grad_clip"]),
            pir_cfg=dict(config["pir"]),
        )
        val_eval = evaluate_ms_model(
            model=model,
            loader=loaders["val"],
            device=device,
            target_scaler=target_scaler,
            subset_lookup=subset_lookup,
            capture_gate_mean=False,
        )
        history_row = {
            "epoch": epoch,
            "loss_total": train_losses["loss_total"],
            "loss_pred": train_losses["loss_pred"],
            "loss_phy": train_losses["loss_phy"],
            "loss_dyn": np.nan,
            "loss_imp": np.nan,
            "loss_moe": train_losses["loss_moe"],
            "val_mae": float(val_eval["total_metrics"]["mae"]),
        }
        history_rows.append(history_row)
        if history_row["val_mae"] < best_val_mae:
            best_val_mae = float(history_row["val_mae"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "spec": spec,
                },
                checkpoint_path,
            )

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_eval = evaluate_ms_model(
        model=model,
        loader=loaders["val"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
        capture_gate_mean=bool(model.use_eddr),
    )
    test_eval = evaluate_ms_model(
        model=model,
        loader=loaders["test"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
        capture_gate_mean=bool(model.use_eddr),
    )
    save_prediction_bundle(prediction_val_path, val_eval)
    save_prediction_bundle(prediction_test_path, test_eval)

    eval_output = output_dirs["root"] / "evaluations"
    eval_output.mkdir(parents=True, exist_ok=True)
    save_evaluation_bundle(eval_output / f"{stem}_val_eval.npz", val_eval)
    save_evaluation_bundle(eval_output / f"{stem}_test_eval.npz", test_eval)

    main_row = {
        "experiment_id": str(spec["experiment_id"]),
        "branch_name": str(spec["branch_name"]),
        "description": DESCRIPTION_MAP[str(spec["experiment_id"])],
        "model_name": "MS-TimeFilter",
        "seq_len": int(config["seq_len"]),
        "pred_len": int(config["pred_len"]),
        "learning_rate": float(spec["learning_rate"]),
        "dropout": float(spec["dropout"]),
        "best_epoch": int(best_epoch),
        "train_samples": int(len(datasets["train"])),
        "val_samples": int(len(datasets["val"])),
        "test_samples": int(len(datasets["test"])),
        "num_parameters": count_parameters(model),
        "feature_count": int(bundle.internal.shape[-1] + bundle.weather.shape[-1] + bundle.blast_v2.shape[-1] + bundle.blast_v3.shape[-1]),
        "val_mae": float(val_eval["total_metrics"]["mae"]),
        "val_rmse": float(val_eval["total_metrics"]["rmse"]),
        "val_mse": float(val_eval["total_metrics"]["mse"]),
        "test_mae": float(test_eval["total_metrics"]["mae"]),
        "test_rmse": float(test_eval["total_metrics"]["rmse"]),
        "test_mse": float(test_eval["total_metrics"]["mse"]),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "prediction_val_path": str(prediction_val_path),
        "prediction_test_path": str(prediction_test_path),
    }

    subset_rows: list[dict[str, Any]] = []
    for split_name, evaluation in [("val", val_eval), ("test", test_eval)]:
        for subset_name, metrics in evaluation["subset_metrics"].items():
            subset_rows.append(
                {
                    "experiment_id": str(spec["experiment_id"]),
                    "branch_name": str(spec["branch_name"]),
                    "description": DESCRIPTION_MAP[str(spec["experiment_id"])],
                    "model_name": "MS-TimeFilter",
                    "split": split_name,
                    "subset": subset_name,
                    "seq_len": int(config["seq_len"]),
                    "pred_len": int(config["pred_len"]),
                    **metrics,
                }
            )
    del model, loaders, datasets, optimizer
    torch.cuda.empty_cache()
    return main_row, subset_rows


def plot_multi_prediction_windows(
    *,
    output_path: Path,
    bundle,
    manifest_test: pd.DataFrame,
    sample_ids: list[int],
    result_rows: list[dict[str, Any]],
    include_ids: list[str],
) -> None:
    selected_rows = [row for row in result_rows if row["experiment_id"] in include_ids and row["branch_name"] == "main"]
    if not selected_rows:
        return

    prediction_lookup = {
        (row["experiment_id"], row["branch_name"]): np.load(row["prediction_test_path"])
        for row in selected_rows
    }
    timestamps = pd.to_datetime(bundle.timestamps)
    blast_strength = bundle.blast_v3[:, :, 0]

    fig, axes = plt.subplots(len(sample_ids), 1, figsize=(12, 4.4 * len(sample_ids)), squeeze=False)
    axes = axes[:, 0]
    for axis, sample_id in zip(axes, sample_ids):
        row = manifest_test.loc[manifest_test["sample_id"] == int(sample_id)].iloc[0]
        encoder_slice = slice(int(row.encoder_start_index), int(row.encoder_end_index_exclusive))
        decoder_slice = slice(int(row.decoder_start_index), int(row.decoder_end_index_exclusive))
        patch_idx = int(np.argmax(blast_strength[decoder_slice].sum(axis=0)))
        history = bundle.target[encoder_slice, patch_idx]
        future = bundle.target[decoder_slice, patch_idx]

        axis.plot(timestamps[encoder_slice], history, color="#6C757D", linewidth=2.0, label="history true")
        axis.plot(timestamps[decoder_slice], future, color="#1D3557", linewidth=2.1, label="future true")
        for color, result_row in zip(["#F4A261", "#E76F51", "#2A9D8F", "#264653"], selected_rows):
            pred_bundle = prediction_lookup[(result_row["experiment_id"], result_row["branch_name"])]
            sample_lookup = {int(value): idx for idx, value in enumerate(pred_bundle["sample_ids"].tolist())}
            pred = pred_bundle["predictions"][sample_lookup[int(sample_id)], :, patch_idx]
            axis.plot(
                timestamps[decoder_slice],
                pred,
                linewidth=1.8,
                marker="o",
                markersize=3,
                color=color,
                label=result_row["experiment_id"],
            )
        axis.axvline(timestamps[encoder_slice][-1], color="#ADB5BD", linestyle="--", linewidth=1.0)
        axis.set_title(f"sample={sample_id}, patch={bundle.patch_ids[patch_idx]}")
        axis.set_ylabel("disp_mean")
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("timestamp")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_graph_heatmap(matrix: torch.Tensor, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    image = ax.imshow(matrix.numpy(), cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("token_j")
    ax.set_ylabel("token_i")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gate_usage(gate_mean: np.ndarray, output_path: Path, title: str) -> None:
    mean_values = gate_mean.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(["spatial", "temporal", "spatiotemporal"], mean_values, color=["#E9C46A", "#2A9D8F", "#264653"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average gate weight")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gate_subset_compare(subset_gate_frame: pd.DataFrame, output_path: Path) -> None:
    plot_df = subset_gate_frame.loc[subset_gate_frame["subset"].isin(["blast_hours", "non_event_hours"])].copy()
    if plot_df.empty:
        return
    categories = ["spatial", "temporal", "spatiotemporal"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(categories))
    width = 0.35
    for offset, subset_name, color in [(-width / 2, "blast_hours", "#E76F51"), (width / 2, "non_event_hours", "#264653")]:
        row = plot_df.loc[plot_df["subset"] == subset_name].iloc[0]
        values = [float(row[f"{name}_mean"]) for name in categories]
        ax.bar(x + offset, values, width=width, label=subset_name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average gate weight")
    ax.set_title("Gate distribution by subset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gate_over_time(
    gate_weights: np.ndarray,
    *,
    patch_count: int,
    time_block_count: int,
    output_path: Path,
) -> None:
    block_gate = gate_weights.reshape(patch_count, time_block_count, 3).mean(axis=0)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for idx, (label, color) in enumerate([("spatial", "#E9C46A"), ("temporal", "#2A9D8F"), ("spatiotemporal", "#264653")]):
        ax.plot(np.arange(time_block_count), block_gate[:, idx], marker="o", linewidth=1.8, label=label, color=color)
    ax.set_xlabel("time_block_id")
    ax.set_ylabel("Average gate weight")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Gate weights over time blocks")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pir_dynamics(
    *,
    true_curve: np.ndarray,
    pred_curve: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    velocity_true = np.diff(true_curve)
    velocity_pred = np.diff(pred_curve)
    accel_true = np.diff(velocity_true)
    accel_pred = np.diff(velocity_pred)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), squeeze=False)
    axes = axes[:, 0]
    for axis, true_values, pred_values, label in [
        (axes[0], true_curve, pred_curve, "displacement"),
        (axes[1], velocity_true, velocity_pred, "velocity"),
        (axes[2], accel_true, accel_pred, "acceleration"),
    ]:
        axis.plot(true_values, color="#1D3557", linewidth=2.0, marker="o", label="true")
        axis.plot(pred_values, color="#E76F51", linewidth=1.8, marker="s", label="pred")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("decoder_step")
    axes[0].legend(frameon=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_weather_debug(
    *,
    output_root: Path,
    sample_timestamp: list[pd.Timestamp],
    attention_weights: np.ndarray,
    debug_shapes: dict[str, Any],
) -> None:
    shape_lines = [f"{name}: {value}" for name, value in debug_shapes.items()]
    (output_root / "weather_conv_feature_shapes.txt").write_text("\n".join(shape_lines) + "\n", encoding="utf-8")

    attention_frame = pd.DataFrame(
        {
            "timestamp": [value.isoformat() for value in sample_timestamp],
            "attention_weight": attention_weights.astype(float),
        }
    )
    attention_frame.to_csv(output_root / "weather_attention_weights_sample.csv", index=False)

    top_rows = attention_frame.sort_values("attention_weight", ascending=False).head(8)
    lines = [
        "# weather_branch_debug",
        "",
        "- 正式天气分支已使用 `Linear Projection + Multi-scale Conv1D + Temporal Attention + Time-block Pooling`。",
        "- 这里的注意力权重来自一个代表性样本的 encoder 96 小时窗口。",
        "",
        "## Tensor Shapes",
        "",
    ]
    lines.extend([f"- `{name}` = `{value}`" for name, value in debug_shapes.items()])
    lines.extend(
        [
            "",
            "## Top Attention Hours",
            "",
            "| timestamp | attention_weight |",
            "| --- | ---: |",
        ]
    )
    for row in top_rows.itertuples(index=False):
        lines.append(f"| {row.timestamp} | {float(row.attention_weight):.6f} |")
    (output_root / "weather_branch_debug.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_graph_debug_report(
    *,
    output_path: Path,
    experiment_id: str,
    branch_name: str,
    graph_state: dict[str, Any],
    heatmap_paths: dict[str, Path],
) -> None:
    lines = [
        "# graph_debug_report_C1",
        "",
        f"- 调试模型：`{experiment_id}_{branch_name}`。",
        f"- 空间先验邻接最大度：`{int(graph_state['spatial_mask'].sum(dim=-1).max().item())}`。",
        f"- 时间先验邻接最大度：`{int(graph_state['temporal_mask'].sum(dim=-1).max().item())}`。",
        f"- learned top-k：`{graph_state['learned_indices'].shape[-1]}`。",
        "",
        "## Heatmaps",
        "",
    ]
    for name, path in heatmap_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate_subset_frame(
    *,
    evaluation: dict[str, Any],
    subset_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    gate_mean = evaluation.get("gate_mean")
    if gate_mean is None:
        return pd.DataFrame()

    sample_ids = evaluation["sample_ids"]
    rows = []
    for subset_name in ["blast_hours", "rain_hours", "non_event_hours"]:
        subset_mask = subset_lookup[subset_name][sample_ids].sum(axis=1) > 0
        if not subset_mask.any():
            continue
        subset_gate = gate_mean[subset_mask]
        rows.append(
            {
                "subset": subset_name,
                "sample_count": int(subset_gate.shape[0]),
                "spatial_mean": float(subset_gate[:, 0].mean()),
                "temporal_mean": float(subset_gate[:, 1].mean()),
                "spatiotemporal_mean": float(subset_gate[:, 2].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_simple_comparison_md(output_path: Path, title: str, frame: pd.DataFrame) -> None:
    lines = [f"# {title}", "", "| experiment_id | branch_name | model_name | val_mae | test_mae | blast_test_mae |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.experiment_id} | {row.branch_name} | {row.model_name} | {float(row.val_mae):.6f} | {float(row.test_mae):.6f} | {float(row.blast_test_mae):.6f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_blast_subset_metrics(results_main: pd.DataFrame, results_subsets: pd.DataFrame) -> pd.DataFrame:
    blast = results_subsets.loc[(results_subsets["split"] == "test") & (results_subsets["subset"] == "blast_hours")].copy()
    blast = blast[["experiment_id", "branch_name", "mae"]].rename(columns={"mae": "blast_test_mae"})
    return results_main.merge(blast, on=["experiment_id", "branch_name"], how="left")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ms_cfg = dict(config["ms_timefilter_v1"])
    ms_cfg["model"] = dict(config["ms_timefilter_v1"]["model"])
    ms_cfg["graph"] = dict(config["ms_timefilter_v1"]["graph"])
    ms_cfg["pir"] = dict(config["ms_timefilter_v1"]["pir"])
    ms_cfg["baseline_model"] = dict(config["formal_round1"]["model"])

    setup_plot_style()
    set_seed(int(ms_cfg["seed"]))
    device = resolve_device(str(ms_cfg["device"]))

    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_root = Path(config["paths"]["output_dir"]) / str(ms_cfg["output_subdir"])
    output_dirs = ensure_output_dirs(output_root)

    split_plan = load_split_plan(Path(config["paths"]["output_dir"]) / "experiment_plan_v1" / "experiment_split_plan_v1.csv")
    bundle, patch_meta, missing_report = load_formal_window_bundle(
        dataset_dir,
        missing_hour_policy=str(ms_cfg["missing_hour_policy"]),
    )
    manifest = build_formal_window_manifest(
        bundle.timestamps,
        split_plan,
        seq_len=int(ms_cfg["seq_len"]),
        pred_len=int(ms_cfg["pred_len"]),
        missing_hour_policy=str(ms_cfg["missing_hour_policy"]),
    )
    manifest_path = dataset_dir / "window_manifest_ms_timefilter_ps10_seq96_pred12.csv"
    manifest.to_csv(manifest_path, index=False)
    build_window_report(
        report_path=output_root / "window_build_report_ms_timefilter.md",
        split_plan=split_plan,
        manifest=manifest,
        missing_report=missing_report,
        seq_len=int(ms_cfg["seq_len"]),
        pred_len=int(ms_cfg["pred_len"]),
    )

    subset_flags = compute_subset_hour_flags(bundle)
    subset_lookup = expand_subset_lookup(
        manifest,
        build_sample_subset_masks(manifest, subset_flags, int(ms_cfg["pred_len"])),
    )

    baseline_specs = [
        {
            "experiment_id": "baseline_lstm_a4_96",
            "branch_name": "reference",
            "model_name": "LSTM",
            "feature_key": "A4",
            "learning_rate": float(ms_cfg["main_learning_rate"]),
            "dropout": float(ms_cfg["model"]["dropout"]),
        },
        {
            "experiment_id": "baseline_raw_timefilter_a4_96",
            "branch_name": "reference",
            "model_name": "TimeFilter",
            "feature_key": "A4",
            "learning_rate": float(ms_cfg["main_learning_rate"]),
            "dropout": float(ms_cfg["model"]["dropout"]),
        },
        {
            "experiment_id": "baseline_naive_weather_concat",
            "branch_name": "control",
            "model_name": "TimeFilter",
            "feature_key": "A2",
            "learning_rate": float(ms_cfg["main_learning_rate"]),
            "dropout": float(ms_cfg["model"]["dropout"]),
        },
    ]
    ms_specs = [
        {"experiment_id": "A2", "branch_name": "main", "learning_rate": float(ms_cfg["main_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "A3", "branch_name": "main", "learning_rate": float(ms_cfg["main_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "A4", "branch_name": "main", "learning_rate": float(ms_cfg["main_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "A4", "branch_name": "blast_opt", "learning_rate": float(ms_cfg["blast_opt_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "C1", "branch_name": "main", "learning_rate": float(ms_cfg["main_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "C1", "branch_name": "blast_opt", "learning_rate": float(ms_cfg["blast_opt_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "C2", "branch_name": "main", "learning_rate": float(ms_cfg["main_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "C2", "branch_name": "blast_opt", "learning_rate": float(ms_cfg["blast_opt_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "C3", "branch_name": "main", "learning_rate": float(ms_cfg["main_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
        {"experiment_id": "C3", "branch_name": "blast_opt", "learning_rate": float(ms_cfg["blast_opt_learning_rate"]), "dropout": float(ms_cfg["model"]["dropout"])},
    ]

    main_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []

    for spec in baseline_specs:
        main_row, subset_part = train_baseline_spec(
            spec=spec,
            manifest=manifest,
            bundle=bundle,
            split_plan=split_plan,
            output_dirs=output_dirs,
            config=ms_cfg,
            device=device,
            subset_lookup=subset_lookup,
        )
        main_rows.append(main_row)
        subset_rows.extend(subset_part)

    for spec in ms_specs:
        main_row, subset_part = train_ms_spec(
            spec=spec,
            manifest=manifest,
            bundle=bundle,
            patch_meta=patch_meta,
            split_plan=split_plan,
            output_dirs=output_dirs,
            config=ms_cfg,
            device=device,
            subset_lookup=subset_lookup,
        )
        main_rows.append(main_row)
        subset_rows.extend(subset_part)

    results_main = pd.DataFrame(main_rows).sort_values(["experiment_id", "branch_name"]).reset_index(drop=True)
    results_subsets = pd.DataFrame(subset_rows).sort_values(["experiment_id", "branch_name", "split", "subset"]).reset_index(drop=True)
    results_main.to_csv(output_root / "results_main_all.csv", index=False)
    results_subsets.to_csv(output_root / "results_subsets_all.csv", index=False)

    for experiment_id in ["A2", "A3", "A4", "C1", "C2", "C3"]:
        results_main.loc[results_main["experiment_id"] == experiment_id].to_csv(output_root / f"results_{experiment_id}.csv", index=False)
        results_subsets.loc[results_subsets["experiment_id"] == experiment_id].to_csv(output_root / f"results_subsets_{experiment_id}.csv", index=False)

    results_c1_c3 = results_main.loc[results_main["experiment_id"].isin(["C1", "C2", "C3"])].reset_index(drop=True)
    results_subsets_c1_c3 = results_subsets.loc[results_subsets["experiment_id"].isin(["C1", "C2", "C3"])].reset_index(drop=True)
    results_c1_c3.to_csv(output_root / "results_C1_C3.csv", index=False)
    results_subsets_c1_c3.to_csv(output_root / "results_subsets_C1_C3.csv", index=False)

    a3_a4_compare = add_blast_subset_metrics(
        results_main.loc[results_main["experiment_id"].isin(["A3", "A4"])].reset_index(drop=True),
        results_subsets,
    )
    write_simple_comparison_md(output_root / "comparison_A3_A4.md", "comparison_A3_A4", a3_a4_compare)

    compare_frame = add_blast_subset_metrics(
        results_main.loc[
            results_main["experiment_id"].isin(
                [
                    "baseline_lstm_a4_96",
                    "baseline_raw_timefilter_a4_96",
                    "C1",
                    "C2",
                    "C3",
                ]
            )
        ].reset_index(drop=True),
        results_subsets,
    )
    write_simple_comparison_md(output_root / "compare_to_lstm_and_timefilter.md", "compare_to_lstm_and_timefilter", compare_frame)

    manifest_test = manifest.loc[manifest["split_name"] == "test"].sort_values("sample_id").reset_index(drop=True)
    selected_samples = select_event_windows(
        test_manifest=manifest_test,
        blast_strength=subset_flags["blast_strength"],
        top_k=int(ms_cfg["plot_event_windows"]),
    )
    debug_split_name = str(ms_cfg["debug_graph_sample_split"])
    debug_manifest = manifest.loc[manifest["split_name"] == debug_split_name].sort_values("sample_id").reset_index(drop=True)
    debug_sample_id = int(
        select_event_windows(
            test_manifest=debug_manifest,
            blast_strength=subset_flags["blast_strength"],
            top_k=max(1, int(ms_cfg["debug_graph_sample_rank"])),
        )[int(ms_cfg["debug_graph_sample_rank"]) - 1]
    )
    plot_multi_prediction_windows(
        output_path=output_dirs["figures"] / "typical_event_windows_A2_A3_A4.png",
        bundle=bundle,
        manifest_test=manifest_test,
        sample_ids=selected_samples,
        result_rows=main_rows,
        include_ids=["A2", "A3", "A4"],
    )
    plot_multi_prediction_windows(
        output_path=output_dirs["figures"] / "typical_event_windows_C2_C3.png",
        bundle=bundle,
        manifest_test=manifest_test,
        sample_ids=selected_samples,
        result_rows=main_rows,
        include_ids=["C2", "C3"],
    )

    for experiment_id in ["A2", "A3", "A4", "C1", "C2", "C3"]:
        eval_path = output_dirs["evaluations"] / f"{experiment_id}_main_test_eval.npz"
        if not eval_path.exists():
            continue
        eval_data = np.load(eval_path, allow_pickle=True)
        plot_patch_mae_heatmap(
            patch_meta=patch_meta,
            patch_mae=eval_data["patch_mae"],
            title=f"{experiment_id} patch MAE heatmap",
            output_path=output_dirs["figures"] / f"patch_mae_heatmap_{experiment_id}.png",
        )

    a4_spec = {"experiment_id": "A4", "branch_name": "main"}
    a4_row = results_main.loc[(results_main["experiment_id"] == "A4") & (results_main["branch_name"] == "main")].iloc[0]
    a4_model = load_model_for_inference(a4_spec, Path(a4_row["checkpoint_path"]), ms_cfg, bundle, patch_meta, device)
    weather_debug_sample = int(selected_samples[0])
    _, datasets_a4, _, _ = build_multibranch_dataloaders(
        manifest=manifest,
        bundle=bundle,
        train_end_index=int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0]),
        batch_size=int(ms_cfg["batch_size"]),
        num_workers=int(ms_cfg["num_workers"]),
    )
    weather_batch = move_batch_to_device(get_sample_batch(datasets_a4["test"], weather_debug_sample), device)
    weather_debug = a4_model(weather_batch, return_aux=True)
    sample_row = manifest_test.loc[manifest_test["sample_id"] == weather_debug_sample].iloc[0]
    encoder_range = slice(int(sample_row.encoder_start_index), int(sample_row.encoder_end_index_exclusive))
    write_weather_debug(
        output_root=output_root,
        sample_timestamp=pd.to_datetime(bundle.timestamps[encoder_range]).tolist(),
        attention_weights=weather_debug["aux"]["weather_attention_weights"][0].detach().cpu().numpy(),
        debug_shapes=weather_debug["aux"]["weather_debug_shapes"],
    )
    del a4_model, datasets_a4
    torch.cuda.empty_cache()

    c1_spec = {"experiment_id": "C1", "branch_name": "main"}
    c1_row = results_main.loc[(results_main["experiment_id"] == "C1") & (results_main["branch_name"] == "main")].iloc[0]
    c1_model = load_model_for_inference(c1_spec, Path(c1_row["checkpoint_path"]), ms_cfg, bundle, patch_meta, device)
    _, datasets_c1, _, _ = build_multibranch_dataloaders(
        manifest=manifest,
        bundle=bundle,
        train_end_index=int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0]),
        batch_size=int(ms_cfg["batch_size"]),
        num_workers=int(ms_cfg["num_workers"]),
    )
    c1_batch = move_batch_to_device(get_sample_batch(datasets_c1[debug_split_name], debug_sample_id), device)
    c1_debug = c1_model(c1_batch, return_aux=True)
    graph_state = c1_debug["aux"]["graph_state"]
    graph_heatmap_paths = {
        "A_prior": output_dirs["figures"] / "graph_A_prior_C1.png",
        "A_learned": output_dirs["figures"] / "graph_A_learned_C1.png",
        "A_final": output_dirs["figures"] / "graph_A_final_C1.png",
    }
    plot_graph_heatmap(graph_state["a_prior_dense"], "C1 A_prior", graph_heatmap_paths["A_prior"])
    plot_graph_heatmap(graph_state["a_learned_dense"], "C1 A_learned", graph_heatmap_paths["A_learned"])
    plot_graph_heatmap(graph_state["a_final_dense"], "C1 A_final", graph_heatmap_paths["A_final"])
    write_graph_debug_report(
        output_path=output_root / "graph_debug_report_C1.md",
        experiment_id="C1",
        branch_name="main",
        graph_state=graph_state,
        heatmap_paths=graph_heatmap_paths,
    )
    write_simple_comparison_md(
        output_root / "comparison_A4_C1.md",
        "comparison_A4_C1",
        add_blast_subset_metrics(results_main.loc[results_main["experiment_id"].isin(["A4", "C1"])].reset_index(drop=True), results_subsets),
    )
    del c1_model, datasets_c1
    torch.cuda.empty_cache()

    c2_spec = {"experiment_id": "C2", "branch_name": "main"}
    c2_eval_path = output_dirs["evaluations"] / "C2_main_test_eval.npz"
    c2_eval_data = np.load(c2_eval_path, allow_pickle=True)
    c2_row = results_main.loc[(results_main["experiment_id"] == "C2") & (results_main["branch_name"] == "main")].iloc[0]

    gate_mean = c2_eval_data.get("gate_mean", None)
    if gate_mean is not None:
        gate_frame = build_gate_subset_frame_from_saved(gate_mean, subset_lookup, c2_eval_data)
        gate_frame.to_csv(output_root / "eddr_gate_stats.csv", index=False)
        plot_gate_usage(gate_mean, output_dirs["figures"] / "eddr_gate_usage_bar.png", "EDDR expert average usage")
        plot_gate_subset_compare(gate_frame, output_dirs["figures"] / "eddr_gate_subset_compare.png")

        c2_model = load_model_for_inference(c2_spec, Path(c2_row["checkpoint_path"]), ms_cfg, bundle, patch_meta, device)
        _, datasets_c2, _, _ = build_multibranch_dataloaders(
            manifest=manifest,
            bundle=bundle,
            train_end_index=int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0]),
            batch_size=int(ms_cfg["batch_size"]),
            num_workers=int(ms_cfg["num_workers"]),
        )
        c2_batch = move_batch_to_device(get_sample_batch(datasets_c2["test"], weather_debug_sample), device)
        c2_debug = c2_model(c2_batch, return_aux=True)
        plot_gate_over_time(
            c2_debug["aux"]["gate_weights"][0].detach().cpu().numpy(),
            patch_count=bundle.target.shape[1],
            time_block_count=int(ms_cfg["seq_len"]) // int(ms_cfg["model"]["patch_len"]),
            output_path=output_dirs["figures"] / "eddr_gate_over_time.png",
        )
        del c2_model, datasets_c2
        torch.cuda.empty_cache()
    eddr_lines = [
        "# eddr_gate_report",
        "",
        "- `C2` 使用 3 专家 softmax gate：spatial / temporal / spatiotemporal。",
        f"- gate 统计表：`{output_root / 'eddr_gate_stats.csv'}`。",
        f"- 专家平均使用率图：`{output_dirs['figures'] / 'eddr_gate_usage_bar.png'}`。",
        f"- blast vs non-event gate 对比图：`{output_dirs['figures'] / 'eddr_gate_subset_compare.png'}`。",
        f"- 典型事件 gate 随时间变化图：`{output_dirs['figures'] / 'eddr_gate_over_time.png'}`。",
    ]
    (output_root / "eddr_gate_report.md").write_text("\n".join(eddr_lines) + "\n", encoding="utf-8")
    write_simple_comparison_md(
        output_root / "comparison_C1_C2.md",
        "comparison_C1_C2",
        add_blast_subset_metrics(results_main.loc[results_main["experiment_id"].isin(["C1", "C2"])].reset_index(drop=True), results_subsets),
    )

    c2_pred_bundle = np.load(results_main.loc[(results_main["experiment_id"] == "C2") & (results_main["branch_name"] == "main"), "prediction_test_path"].iloc[0])
    c3_pred_bundle = np.load(results_main.loc[(results_main["experiment_id"] == "C3") & (results_main["branch_name"] == "main"), "prediction_test_path"].iloc[0])
    c2_lookup = {int(sample_id): idx for idx, sample_id in enumerate(c2_pred_bundle["sample_ids"].tolist())}
    c3_lookup = {int(sample_id): idx for idx, sample_id in enumerate(c3_pred_bundle["sample_ids"].tolist())}
    patch_idx = int(np.argmax(bundle.blast_v3[int(sample_row.decoder_start_index): int(sample_row.decoder_end_index_exclusive), :, 0].sum(axis=0)))
    true_curve = c3_pred_bundle["targets"][c3_lookup[weather_debug_sample], :, patch_idx]
    pred_curve = c3_pred_bundle["predictions"][c3_lookup[weather_debug_sample], :, patch_idx]
    plot_pir_dynamics(
        true_curve=true_curve,
        pred_curve=pred_curve,
        output_path=output_dirs["figures"] / "pir_dynamics_C3.png",
        title="C3 physical consistency window",
    )
    plot_multi_prediction_windows(
        output_path=output_dirs["figures"] / "typical_event_windows_C2_vs_C3.png",
        bundle=bundle,
        manifest_test=manifest_test,
        sample_ids=selected_samples,
        result_rows=main_rows,
        include_ids=["C2", "C3"],
    )
    pir_lines = [
        "# pir_loss_report",
        "",
        "- `C3` 使用 `L_total = L_pred + beta * L_phy`，其中 `alpha_v=1.0`、`alpha_a=0.5`、`beta=0.05`。",
        "- 当前 open-source TimeFilter 只稳定暴露 `moe_loss`，`loss_dyn` / `loss_imp` 接口保留为日志列并填 `NA`。",
        f"- 动态一致性图：`{output_dirs['figures'] / 'pir_dynamics_C3.png'}`。",
        f"- C2 vs C3 典型事件窗口图：`{output_dirs['figures'] / 'typical_event_windows_C2_vs_C3.png'}`。",
    ]
    (output_root / "pir_loss_report.md").write_text("\n".join(pir_lines) + "\n", encoding="utf-8")
    write_simple_comparison_md(
        output_root / "comparison_C2_C3.md",
        "comparison_C2_C3",
        add_blast_subset_metrics(results_main.loc[results_main["experiment_id"].isin(["C2", "C3"])].reset_index(drop=True), results_subsets),
    )

    subset_with_blast = add_blast_subset_metrics(results_main, results_subsets)
    decision_lines = [
        "# ms_timefilter_decision_report",
        "",
        "## Main Table",
        "",
        "| experiment_id | branch_name | model_name | val_mae | test_mae | blast_test_mae |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in subset_with_blast.sort_values(["experiment_id", "branch_name"]).itertuples(index=False):
        decision_lines.append(
            f"| {row.experiment_id} | {row.branch_name} | {row.model_name} | {float(row.val_mae):.6f} | {float(row.test_mae):.6f} | {float(row.blast_test_mae):.6f} |"
        )

    def get_test_mae(experiment_id: str, branch_name: str = "main") -> float:
        row = subset_with_blast.loc[(subset_with_blast["experiment_id"] == experiment_id) & (subset_with_blast["branch_name"] == branch_name)].iloc[0]
        return float(row["test_mae"])

    def get_blast_mae(experiment_id: str, branch_name: str = "main") -> float:
        row = subset_with_blast.loc[(subset_with_blast["experiment_id"] == experiment_id) & (subset_with_blast["branch_name"] == branch_name)].iloc[0]
        return float(row["blast_test_mae"])

    answer_lines = [
        "",
        "## Answers",
        "",
        f"1. A2 相比 naive weather concat 是否更优：{'是' if get_test_mae('A2') <= get_test_mae('baseline_naive_weather_concat', 'control') else '否'}。",
        f"2. A4 是否优于 A3：{'是' if get_test_mae('A4') <= get_test_mae('A3') else '否'}。",
        f"3. C1 是否优于 A4 和原始 TimeFilter：{'是' if (get_test_mae('C1') <= get_test_mae('A4') and get_test_mae('C1') <= get_test_mae('baseline_raw_timefilter_a4_96', 'reference')) else '否'}。",
        f"4. C2 是否在 blast_hours 子集上体现额外增益：{'是' if get_blast_mae('C2') <= get_blast_mae('C1') else '否'}。",
        f"5. C3 是否提升了物理一致性或曲线合理性：请结合 `pir_dynamics_C3.png` 与 `typical_event_windows_C2_vs_C3.png` 人工复核。",
        f"6. 完整 MS-TimeFilter 是否逼近或超过当前最强 LSTM：{'是' if subset_with_blast.loc[subset_with_blast['experiment_id'].isin(['A2', 'A3', 'A4', 'C1', 'C2', 'C3']), 'test_mae'].min() <= get_test_mae('baseline_lstm_a4_96', 'reference') else '否'}。",
        "7. 若未超过 LSTM，优先从 backbone、图构建、事件路由和物理损失四个方向解释；最终以当前总表数值为准。",
    ]
    (output_root / "ms_timefilter_decision_report.md").write_text("\n".join(decision_lines + answer_lines) + "\n", encoding="utf-8")

    print(manifest_path)
    print(output_root / "results_main_all.csv")
    print(output_root / "results_subsets_all.csv")
    print(output_root / "ms_timefilter_decision_report.md")


if __name__ == "__main__":
    main()
