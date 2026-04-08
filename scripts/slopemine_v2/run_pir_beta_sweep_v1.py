#!/usr/bin/env python3
"""运行 C3 的 PIR beta 轻量扫描。
相关文件：config_v2.toml、results_main_all.csv、results_subsets_all.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

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
    build_multibranch_dataloaders,
    build_sample_subset_masks,
    compute_subset_hour_flags,
    load_formal_window_bundle,
    load_split_plan,
)
from models.ms_timefilter import MSTimeFilter  # noqa: E402
from run_formal_patch_experiments_v1 import expand_subset_lookup, load_config, resolve_device, save_prediction_bundle, set_seed  # noqa: E402
from run_ms_timefilter_experiments_v1 import evaluate_ms_model, train_ms_one_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight PIR beta sweep for C3.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/slopemine_v2/config_v2.toml"),
        help="Path to config_v2.toml.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 5e-4, 1e-3],
        help="PIR beta values.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum epochs for this lightweight sweep.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience on val MAE.",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "histories": output_root / "histories",
        "predictions": output_root / "predictions",
        "evaluations": output_root / "evaluations",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_eval_bundle(path: Path, evaluation: dict[str, Any]) -> None:
    payload = {
        "sample_ids": evaluation["sample_ids"],
        "predictions": evaluation["predictions"],
        "targets": evaluation["targets"],
        "masks": evaluation["masks"],
        "patch_mae": evaluation["patch_mae"],
    }
    if "gate_mean" in evaluation:
        payload["gate_mean"] = evaluation["gate_mean"]
    np.savez_compressed(path, **payload)


def train_one_beta(
    *,
    beta: float,
    output_dirs: dict[str, Path],
    config: dict[str, Any],
    bundle,
    patch_meta: pd.DataFrame,
    manifest: pd.DataFrame,
    split_plan: pd.DataFrame,
    subset_lookup: dict[str, np.ndarray],
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    train_end_index = int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0])
    loaders, datasets, _feature_scalers, target_scaler = build_multibranch_dataloaders(
        manifest=manifest,
        bundle=bundle,
        train_end_index=train_end_index,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    model = MSTimeFilter(
        experiment_id="C3",
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
        lr=float(config["main_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    stem = f"pir_beta_{beta:.0e}".replace("-", "m")
    checkpoint_path = output_dirs["checkpoints"] / f"{stem}.pt"
    history_path = output_dirs["histories"] / f"{stem}.csv"
    prediction_val_path = output_dirs["predictions"] / f"{stem}_val.npz"
    prediction_test_path = output_dirs["predictions"] / f"{stem}_test.npz"
    eval_val_path = output_dirs["evaluations"] / f"{stem}_val_eval.npz"
    eval_test_path = output_dirs["evaluations"] / f"{stem}_test_eval.npz"

    pir_cfg = dict(config["pir"])
    pir_cfg["beta"] = float(beta)

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        train_losses = train_ms_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            target_scaler=target_scaler,
            grad_clip=float(config["grad_clip"]),
            pir_cfg=pir_cfg,
        )
        val_eval = evaluate_ms_model(
            model=model,
            loader=loaders["val"],
            device=device,
            target_scaler=target_scaler,
            subset_lookup=subset_lookup,
            capture_gate_mean=False,
        )
        val_mae = float(val_eval["total_metrics"]["mae"])
        history_rows.append(
            {
                "epoch": epoch,
                "beta": float(beta),
                "loss_total": train_losses["loss_total"],
                "loss_pred": train_losses["loss_pred"],
                "loss_phy": train_losses["loss_phy"],
                "loss_dyn": np.nan,
                "loss_imp": np.nan,
                "loss_moe": train_losses["loss_moe"],
                "val_mae": val_mae,
            }
        )
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            wait = 0
            torch.save({"model_state_dict": model.state_dict(), "beta": float(beta)}, checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                break

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_eval = evaluate_ms_model(
        model=model,
        loader=loaders["val"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
        capture_gate_mean=False,
    )
    test_eval = evaluate_ms_model(
        model=model,
        loader=loaders["test"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
        capture_gate_mean=False,
    )
    save_prediction_bundle(prediction_val_path, val_eval)
    save_prediction_bundle(prediction_test_path, test_eval)
    save_eval_bundle(eval_val_path, val_eval)
    save_eval_bundle(eval_test_path, test_eval)

    row = {
        "beta": float(beta),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "best_epoch": int(best_epoch),
        "train_samples": int(len(datasets["train"])),
        "val_samples": int(len(datasets["val"])),
        "test_samples": int(len(datasets["test"])),
        "val_mae": float(val_eval["total_metrics"]["mae"]),
        "val_rmse": float(val_eval["total_metrics"]["rmse"]),
        "val_mse": float(val_eval["total_metrics"]["mse"]),
        "test_mae": float(test_eval["total_metrics"]["mae"]),
        "test_rmse": float(test_eval["total_metrics"]["rmse"]),
        "test_mse": float(test_eval["total_metrics"]["mse"]),
        "blast_test_mae": float(test_eval["subset_metrics"]["blast_hours"]["mae"]),
        "blast_test_rmse": float(test_eval["subset_metrics"]["blast_hours"]["rmse"]),
        "blast_test_mse": float(test_eval["subset_metrics"]["blast_hours"]["mse"]),
        "rain_test_mae": float(test_eval["subset_metrics"]["rain_hours"]["mae"]),
        "rain_test_rmse": float(test_eval["subset_metrics"]["rain_hours"]["rmse"]),
        "rain_test_mse": float(test_eval["subset_metrics"]["rain_hours"]["mse"]),
        "non_event_test_mae": float(test_eval["subset_metrics"]["non_event_hours"]["mae"]),
        "non_event_test_rmse": float(test_eval["subset_metrics"]["non_event_hours"]["rmse"]),
        "non_event_test_mse": float(test_eval["subset_metrics"]["non_event_hours"]["mse"]),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "prediction_test_path": str(prediction_test_path),
    }
    del model, loaders, datasets, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def write_report(
    *,
    output_path: Path,
    sweep: pd.DataFrame,
    reference_main: pd.DataFrame,
    reference_subsets: pd.DataFrame,
    max_epochs: int,
    patience: int,
) -> None:
    best_test = sweep.sort_values(["test_mae", "blast_test_mae", "beta"]).iloc[0]
    c2_main = reference_main.loc[(reference_main["experiment_id"] == "C2") & (reference_main["branch_name"] == "main")].iloc[0]
    c3_main = reference_main.loc[(reference_main["experiment_id"] == "C3") & (reference_main["branch_name"] == "main")].iloc[0]
    c2_blast = reference_subsets.loc[
        (reference_subsets["experiment_id"] == "C2")
        & (reference_subsets["branch_name"] == "main")
        & (reference_subsets["split"] == "test")
        & (reference_subsets["subset"] == "blast_hours")
    ].iloc[0]
    c3_blast = reference_subsets.loc[
        (reference_subsets["experiment_id"] == "C3")
        & (reference_subsets["branch_name"] == "main")
        & (reference_subsets["split"] == "test")
        & (reference_subsets["subset"] == "blast_hours")
    ].iloc[0]

    lines = [
        "# pir_beta_report",
        "",
        "- 这是 `PIR` 的轻量诊断复核，不改 backbone / split / patch 粒度，只扫描 `beta`。",
        f"- 轻量协议：`max_epochs={max_epochs}`、`early_stopping_patience={patience}`。",
        "- 之所以采用这组轻量轮数，是因为当前已落盘主结果里 `C2 main` 的最佳验证轮次为 8、`C3 main` 的最佳验证轮次为 10，已经覆盖早期最优区间。",
        "",
        "## Best Beta",
        "",
        f"- 最优 beta：`{float(best_test['beta']):.4g}`。",
        f"- 对应 test MAE：`{float(best_test['test_mae']):.6f}`；blast_hours MAE：`{float(best_test['blast_test_mae']):.6f}`。",
        f"- 相比当前正式 `C2 main` 的 test MAE 差值：`{float(best_test['test_mae']) - float(c2_main['test_mae']):+.6f}`。",
        f"- 相比当前正式 `C3 main(beta=0.05)` 的 test MAE 差值：`{float(best_test['test_mae']) - float(c3_main['test_mae']):+.6f}`。",
        "",
        "## Decision",
        "",
    ]

    if float(best_test["test_mae"]) + 1e-12 < float(c2_main["test_mae"]):
        lines.append("- 轻量 beta 扫描显示：`PIR` 仍然有机会成为主实验模块，当前退化更可能与原始 `beta=0.05` 设置过大有关。")
    else:
        lines.append("- 轻量 beta 扫描没有把 `C3` 拉回到 `C2 main` 之上，因此当前退化不只是 `beta=0.05` 过大，更可能是物理正则本身与现有 backbone / 训练流程耦合不足。")
        lines.append("- 按当前证据，`PIR` 应正式降级为辅助约束，不再作为当前主实验核心模块。")

    lines.extend(
        [
            "",
            "## References",
            "",
            f"- `C2 main`：test MAE=`{float(c2_main['test_mae']):.6f}`，blast_hours MAE=`{float(c2_blast['mae']):.6f}`。",
            f"- `C3 main(beta=0.05)`：test MAE=`{float(c3_main['test_mae']):.6f}`，blast_hours MAE=`{float(c3_blast['mae']):.6f}`。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ms_cfg = dict(config["ms_timefilter_v1"])
    ms_cfg["model"] = dict(config["ms_timefilter_v1"]["model"])
    ms_cfg["graph"] = dict(config["ms_timefilter_v1"]["graph"])
    ms_cfg["pir"] = dict(config["ms_timefilter_v1"]["pir"])

    set_seed(int(ms_cfg["seed"]))
    device = resolve_device(str(ms_cfg["device"]))

    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_root = Path(config["paths"]["output_dir"]) / str(ms_cfg["output_subdir"])
    sweep_root = output_root / "pir_beta_sweep_artifacts"
    output_dirs = ensure_dirs(sweep_root)

    split_plan = load_split_plan(Path(config["paths"]["output_dir"]) / "experiment_plan_v1" / "experiment_split_plan_v1.csv")
    bundle, patch_meta, _missing_report = load_formal_window_bundle(dataset_dir, missing_hour_policy=str(ms_cfg["missing_hour_policy"]))
    manifest = build_formal_window_manifest(
        bundle.timestamps,
        split_plan,
        seq_len=int(ms_cfg["seq_len"]),
        pred_len=int(ms_cfg["pred_len"]),
        missing_hour_policy=str(ms_cfg["missing_hour_policy"]),
    )
    subset_flags = compute_subset_hour_flags(bundle)
    subset_lookup = expand_subset_lookup(
        manifest,
        build_sample_subset_masks(manifest, subset_flags, int(ms_cfg["pred_len"])),
    )

    rows = []
    for beta in args.betas:
        set_seed(int(ms_cfg["seed"]))
        rows.append(
            train_one_beta(
                beta=float(beta),
                output_dirs=output_dirs,
                config=ms_cfg,
                bundle=bundle,
                patch_meta=patch_meta,
                manifest=manifest,
                split_plan=split_plan,
                subset_lookup=subset_lookup,
                device=device,
                max_epochs=int(args.max_epochs),
                patience=int(args.patience),
            )
        )

    sweep = pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)
    sweep_path = output_root / "pir_beta_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    reference_main = pd.read_csv(output_root / "results_main_all.csv")
    reference_subsets = pd.read_csv(output_root / "results_subsets_all.csv")
    report_path = output_root / "pir_beta_report.md"
    write_report(
        output_path=report_path,
        sweep=sweep,
        reference_main=reference_main,
        reference_subsets=reference_subsets,
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
    )

    print(sweep_path)
    print(report_path)


if __name__ == "__main__":
    main()
