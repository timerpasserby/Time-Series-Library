#!/usr/bin/env python3
"""这个脚本负责在正式 split 下搜索最佳 seq_len / pred_len 基础配置。
相关文件：config_v2.toml、run_patch_baselines_v2.py、experiment_split_plan_v1.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from run_patch_baselines_v2 import (  # noqa: E402
    EXPERIMENTS,
    build_feature_tensor,
    compute_metrics,
    fit_ridge,
    load_bundle,
    load_config,
    predict_ridge,
    prepare_targets,
    summarize_windows,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Tune seq_len / pred_len under the formal split plan.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/slopemine_v2/config_v2.toml"),
        help="Path to the TOML config file.",
    )
    return parser.parse_args()


def ensure_output_dir(config: dict[str, Any]) -> Path:
    """创建参数搜索结果目录。"""

    output_dir = Path(config["paths"]["output_dir"]) / "tuning_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_effective_bundle(data: dict[str, Any]) -> dict[str, Any]:
    """把 720 小时整轴压缩成 712 个有效时间步。"""

    bundle = data["bundle"]
    keep_mask = bundle["global_missing"].astype(np.uint8) == 0
    effective: dict[str, Any] = {
        "timestamps": bundle["timestamps"][keep_mask],
        "patch_ids": bundle["patch_ids"],
        "internal_feature_names": bundle["internal_feature_names"],
        "weather_feature_names": bundle["weather_feature_names"],
        "blast_v2_feature_names": bundle["blast_v2_feature_names"],
        "blast_v3_feature_names": bundle["blast_v3_feature_names"],
        "internal": bundle["internal"][keep_mask],
        "weather": bundle["weather"][keep_mask],
        "blast_v2": bundle["blast_v2"][keep_mask],
        "blast_v3": bundle["blast_v3"][keep_mask],
        "target": bundle["target"][keep_mask],
        "target_mask": bundle["target_mask"][keep_mask],
        "input_mask": bundle["input_mask"][keep_mask],
        "global_missing": bundle["global_missing"][keep_mask],
    }
    return {
        "bundle": effective,
        "patch_meta": data["patch_meta"],
    }


def load_split_plan(dataset_dir: Path, output_dir: Path) -> pd.DataFrame:
    """读取正式 split 计划。"""

    split_path = output_dir / "experiment_plan_v1" / "experiment_split_plan_v1.csv"
    split_plan = pd.read_csv(split_path, parse_dates=["start_timestamp", "end_timestamp"])
    required = {"train", "val", "test"}
    if set(split_plan["split_name"]) != required:
        raise ValueError("experiment_split_plan_v1.csv 缺少 train/val/test 三段正式切分。")
    return split_plan.sort_values("effective_start_index").reset_index(drop=True)


def build_sample_split_masks(
    split_plan: pd.DataFrame,
    total_steps: int,
    seq_len: int,
    pred_len: int,
) -> dict[str, np.ndarray]:
    """根据正式 split 规则构造样本级 train / val / test 掩码。"""

    num_samples = total_steps - seq_len - pred_len + 1
    if num_samples <= 0:
        raise ValueError(f"无可用窗口：seq_len={seq_len}, pred_len={pred_len}, total_steps={total_steps}")

    sample_start = np.arange(num_samples, dtype=np.int32)
    target_start = sample_start + seq_len
    target_end_exclusive = target_start + pred_len

    masks: dict[str, np.ndarray] = {}
    for row in split_plan.itertuples(index=False):
        masks[str(row.split_name)] = (
            (target_start >= int(row.effective_start_index))
            & (target_end_exclusive <= int(row.effective_end_index_exclusive))
        )
    return masks


def run_single_experiment_formal(
    *,
    experiment_key: str,
    seq_len: int,
    pred_len: int,
    data: dict[str, Any],
    config: dict[str, Any],
    split_plan: pd.DataFrame,
) -> dict[str, Any]:
    """在正式 split 下运行单个实验配置。"""

    feature_tensor, feature_names = build_feature_tensor(data, experiment_key)
    summary = summarize_windows(
        feature_tensor=feature_tensor,
        seq_len=seq_len,
        recent_window=int(config["baseline"]["recent_window"]),
    )

    target = data["bundle"]["target"].astype(np.float32)
    input_mask = data["bundle"]["input_mask"].astype(np.uint8)
    y, y_mask, x_valid = prepare_targets(target, input_mask, pred_len, seq_len)
    num_samples = y.shape[0]
    summary = summary[:num_samples]
    patch_count = summary.shape[1]
    summary_dim = summary.shape[2]

    sample_masks = build_sample_split_masks(
        split_plan=split_plan,
        total_steps=target.shape[0],
        seq_len=seq_len,
        pred_len=pred_len,
    )

    x_flat = summary.reshape(num_samples * patch_count, summary_dim)
    y_flat = np.moveaxis(y, 2, 1).reshape(num_samples * patch_count, pred_len)
    mask_flat = np.moveaxis(y_mask, 2, 1).reshape(num_samples * patch_count, pred_len)
    row_valid = (mask_flat.min(axis=1) == 1) & x_valid.reshape(-1)
    sample_index = np.repeat(np.arange(num_samples), patch_count)

    split_row_masks = {
        split_name: row_valid & np.repeat(mask[:, None], patch_count, axis=1).reshape(-1)
        for split_name, mask in sample_masks.items()
    }

    train_rows = split_row_masks["train"]
    val_rows = split_row_masks["val"]
    test_rows = split_row_masks["test"]
    if not train_rows.any() or not val_rows.any() or not test_rows.any():
        raise ValueError(
            f"正式 split 下窗口不足：experiment={experiment_key}, seq_len={seq_len}, pred_len={pred_len}"
        )

    weights, x_mean, x_std, y_mean = fit_ridge(
        x_train=x_flat[train_rows],
        y_train=y_flat[train_rows],
        alpha=float(config["baseline"]["ridge_alpha"]),
    )

    pred_val = predict_ridge(x_flat[val_rows], weights, x_mean, x_std, y_mean)
    pred_test = predict_ridge(x_flat[test_rows], weights, x_mean, x_std, y_mean)
    true_val = y_flat[val_rows]
    true_test = y_flat[test_rows]

    return {
        "experiment": experiment_key,
        "description": EXPERIMENTS[experiment_key],
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "feature_count": len(feature_names),
        "train_rows": int(train_rows.sum()),
        "val_rows": int(val_rows.sum()),
        "test_rows": int(test_rows.sum()),
        "val_metrics": compute_metrics(true_val, pred_val),
        "test_metrics": compute_metrics(true_test, pred_test),
        "val_samples": int(sample_masks["val"].sum()),
        "test_samples": int(sample_masks["test"].sum()),
    }


def build_metrics_table(
    *,
    effective_data: dict[str, Any],
    config: dict[str, Any],
    split_plan: pd.DataFrame,
) -> pd.DataFrame:
    """扫全部窗口组合并汇总 val/test 指标。"""

    rows: list[dict[str, Any]] = []
    for seq_len in config["windows"]["seq_lens"]:
        for pred_len in config["windows"]["pred_lens"]:
            for experiment_key in ["A1", "A2", "A3", "A4"]:
                result = run_single_experiment_formal(
                    experiment_key=experiment_key,
                    seq_len=int(seq_len),
                    pred_len=int(pred_len),
                    data=effective_data,
                    config=config,
                    split_plan=split_plan,
                )
                for split_name in ["val", "test"]:
                    metrics = result[f"{split_name}_metrics"]
                    rows.append(
                        {
                            "experiment": result["experiment"],
                            "description": result["description"],
                            "seq_len": result["seq_len"],
                            "pred_len": result["pred_len"],
                            "split": split_name,
                            "feature_count": result["feature_count"],
                            "sample_count": result[f"{split_name}_samples"],
                            "row_count": result[f"{split_name}_rows"],
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["experiment", "seq_len", "pred_len", "split"]).reset_index(drop=True)


def select_best_configs(metrics: pd.DataFrame) -> dict[str, Any]:
    """按正式 val 指标挑选最佳配置。"""

    val_metrics = metrics.loc[metrics["split"] == "val"].copy()
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()

    a4_val = val_metrics.loc[val_metrics["experiment"] == "A4"].sort_values(["mae", "rmse", "pred_len", "seq_len"]).reset_index(drop=True)
    a4_test = test_metrics.loc[test_metrics["experiment"] == "A4"].copy()

    best_overall = a4_val.iloc[0]
    best_overall_test = a4_test.loc[
        (a4_test["seq_len"] == best_overall["seq_len"]) & (a4_test["pred_len"] == best_overall["pred_len"])
    ].iloc[0]

    a4_pred12_val = a4_val.loc[a4_val["pred_len"] == 12].sort_values(["mae", "rmse", "seq_len"]).reset_index(drop=True)
    best_pred12 = a4_pred12_val.iloc[0]
    best_pred12_test = a4_test.loc[
        (a4_test["seq_len"] == best_pred12["seq_len"]) & (a4_test["pred_len"] == best_pred12["pred_len"])
    ].iloc[0]

    experiment_best_rows = []
    for experiment_key in ["A1", "A2", "A3", "A4"]:
        frame = val_metrics.loc[val_metrics["experiment"] == experiment_key].sort_values(["mae", "rmse", "pred_len", "seq_len"]).reset_index(drop=True)
        best_row = frame.iloc[0]
        test_row = test_metrics.loc[
            (test_metrics["experiment"] == experiment_key)
            & (test_metrics["seq_len"] == best_row["seq_len"])
            & (test_metrics["pred_len"] == best_row["pred_len"])
        ].iloc[0]
        experiment_best_rows.append(
            {
                "experiment": experiment_key,
                "seq_len": int(best_row["seq_len"]),
                "pred_len": int(best_row["pred_len"]),
                "val_mae": float(best_row["mae"]),
                "val_rmse": float(best_row["rmse"]),
                "test_mae": float(test_row["mae"]),
                "test_rmse": float(test_row["rmse"]),
            }
        )

    comparison_48_12_val = a4_val.loc[(a4_val["seq_len"] == 48) & (a4_val["pred_len"] == 12)].iloc[0]
    comparison_48_12_test = a4_test.loc[(a4_test["seq_len"] == 48) & (a4_test["pred_len"] == 12)].iloc[0]

    return {
        "best_overall": {
            "seq_len": int(best_overall["seq_len"]),
            "pred_len": int(best_overall["pred_len"]),
            "val_mae": float(best_overall["mae"]),
            "val_rmse": float(best_overall["rmse"]),
            "test_mae": float(best_overall_test["mae"]),
            "test_rmse": float(best_overall_test["rmse"]),
        },
        "best_pred_len_12": {
            "seq_len": int(best_pred12["seq_len"]),
            "pred_len": int(best_pred12["pred_len"]),
            "val_mae": float(best_pred12["mae"]),
            "val_rmse": float(best_pred12["rmse"]),
            "test_mae": float(best_pred12_test["mae"]),
            "test_rmse": float(best_pred12_test["rmse"]),
        },
        "compare_48_12": {
            "seq_len": 48,
            "pred_len": 12,
            "val_mae": float(comparison_48_12_val["mae"]),
            "val_rmse": float(comparison_48_12_val["rmse"]),
            "test_mae": float(comparison_48_12_test["mae"]),
            "test_rmse": float(comparison_48_12_test["rmse"]),
        },
        "experiment_best_rows": experiment_best_rows,
    }


def build_summary_markdown(
    *,
    output_path: Path,
    selection: dict[str, Any],
    metrics: pd.DataFrame,
) -> None:
    """输出窗口搜索摘要报告。"""

    val_metrics = metrics.loc[(metrics["split"] == "val") & (metrics["experiment"] == "A4")].copy()
    val_metrics = val_metrics.sort_values(["mae", "rmse", "pred_len", "seq_len"]).reset_index(drop=True)
    top_rows = val_metrics.head(6)

    lines = [
        "# window_tuning_summary_v1",
        "",
        "- 本次搜索基于冻结后的正式 split：712 个有效时间步，train / val / test 按 `experiment_split_plan_v1.csv` 划分。",
        "- 搜索网格：`seq_len ∈ {24, 48, 96}`，`pred_len ∈ {1, 3, 6, 12}`。",
        "- 推荐主依据：A4 在验证集上的 `mae`，并同时记录测试集表现。",
        "",
        "## A4 验证集 Top 6",
        "",
        "| rank | seq_len | pred_len | val_mae | val_rmse |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top_rows.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {int(row.seq_len)} | {int(row.pred_len)} | {float(row.mae):.10f} | {float(row.rmse):.10f} |"
        )

    best_overall = selection["best_overall"]
    best_pred12 = selection["best_pred_len_12"]
    compare_48_12 = selection["compare_48_12"]

    lines.extend(
        [
            "",
            "## 推荐结论",
            "",
            f"- 如果只看正式验证集误差，当前最佳基础配置是 `seq_len={best_overall['seq_len']}, pred_len={best_overall['pred_len']}`；对应 val/test MAE 为 `{best_overall['val_mae']:.10f} / {best_overall['test_mae']:.10f}`。",
            f"- 如果限定做 `12` 步预测，当前最佳是 `seq_len={best_pred12['seq_len']}, pred_len=12`；对应 val/test MAE 为 `{best_pred12['val_mae']:.10f} / {best_pred12['test_mae']:.10f}`。",
            f"- 你提到的 `seq_len=48, pred_len=12` 也已单独核过；它的 val/test MAE 为 `{compare_48_12['val_mae']:.10f} / {compare_48_12['test_mae']:.10f}`。",
            "",
            "## 各实验最佳窗口",
            "",
            "| experiment | seq_len | pred_len | val_mae | test_mae |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selection["experiment_best_rows"]:
        lines.append(
            f"| {row['experiment']} | {row['seq_len']} | {row['pred_len']} | {row['val_mae']:.10f} | {row['test_mae']:.10f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """执行正式窗口参数搜索。"""

    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_output_dir(config)
    data = load_bundle(Path(config["paths"]["dataset_dir"]))
    effective_data = build_effective_bundle(data)
    split_plan = load_split_plan(Path(config["paths"]["dataset_dir"]), Path(config["paths"]["output_dir"]))

    metrics = build_metrics_table(
        effective_data=effective_data,
        config=config,
        split_plan=split_plan,
    )
    metrics.to_csv(output_dir / "window_tuning_metrics_v1.csv", index=False)

    selection = select_best_configs(metrics)
    (output_dir / "best_window_config_v1.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_summary_markdown(
        output_path=output_dir / "window_tuning_summary_v1.md",
        selection=selection,
        metrics=metrics,
    )

    print(output_dir / "window_tuning_metrics_v1.csv")
    print(output_dir / "best_window_config_v1.json")
    print(output_dir / "window_tuning_summary_v1.md")


if __name__ == "__main__":
    main()
