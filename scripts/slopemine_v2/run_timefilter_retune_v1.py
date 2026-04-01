#!/usr/bin/env python3
"""这个脚本负责在冻结后的正式 A4 数据版本上做 TimeFilter 小规模公平性复核。
相关文件：config_v2.toml、run_formal_patch_experiments_v1.py、results_main_A1_A4.csv
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
    build_sample_subset_masks,
    compute_subset_hour_flags,
    load_formal_window_bundle,
    load_split_plan,
)
from models.slopemine_formal_wrappers import FormalPatchModel, build_model_configs  # noqa: E402
from run_formal_patch_experiments_v1 import (  # noqa: E402
    build_dataloaders,
    evaluate_model,
    expand_subset_lookup,
    load_config,
    resolve_device,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retune TimeFilter under the frozen formal A4 setup.")
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
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def format_lr_tag(value: float) -> str:
    text = f"{value:.0e}"
    return text.replace("+", "").replace("-", "m")


def load_reference_rows(formal_root: Path) -> dict[str, dict[str, float]]:
    main = pd.read_csv(formal_root / "results_main_A1_A4.csv")
    subsets = pd.read_csv(formal_root / "results_subsets_A1_A4.csv")

    default_main = (
        main.loc[(main["experiment"] == "A4") & (main["model_name"] == "TimeFilter")]
        .iloc[0]
        .to_dict()
    )
    lstm_main = (
        main.loc[(main["experiment"] == "A4") & (main["model_name"] == "LSTM")]
        .iloc[0]
        .to_dict()
    )

    default_blast = (
        subsets.loc[
            (subsets["experiment"] == "A4")
            & (subsets["model_name"] == "TimeFilter")
            & (subsets["split"] == "test")
            & (subsets["subset"] == "blast_hours")
        ]
        .iloc[0]
        .to_dict()
    )
    lstm_blast = (
        subsets.loc[
            (subsets["experiment"] == "A4")
            & (subsets["model_name"] == "LSTM")
            & (subsets["split"] == "test")
            & (subsets["subset"] == "blast_hours")
        ]
        .iloc[0]
        .to_dict()
    )

    return {
        "default_main": default_main,
        "lstm_main": lstm_main,
        "default_blast": default_blast,
        "lstm_blast": lstm_blast,
    }


def train_single_config(
    *,
    seq_len: int,
    learning_rate: float,
    dropout: float,
    feature_array: np.ndarray,
    bundle,
    loaders,
    target_scaler: dict[str, np.ndarray],
    subset_lookup: dict[str, np.ndarray],
    split_sizes: dict[str, int],
    device: torch.device,
    retune_cfg: dict[str, Any],
    output_dirs: dict[str, Path],
) -> dict[str, Any]:
    model_cfg = dict(retune_cfg["model"])
    model_cfg["dropout"] = float(dropout)
    model = FormalPatchModel(
        model_name=str(retune_cfg["model_name"]),
        feature_dim=feature_array.shape[-1],
        num_nodes=feature_array.shape[1],
        seq_len=int(seq_len),
        pred_len=int(retune_cfg["pred_len"]),
        model_cfg=model_cfg,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(retune_cfg["weight_decay"]),
    )

    model_configs = build_model_configs(
        seq_len=int(seq_len),
        pred_len=int(retune_cfg["pred_len"]),
        num_nodes=int(feature_array.shape[1]),
        model_cfg=model_cfg,
    )

    file_stem = f"tf_sl{seq_len}_lr{format_lr_tag(float(learning_rate))}_do{str(dropout).replace('.', 'p')}"
    checkpoint_path = output_dirs["checkpoints"] / f"{file_stem}_best.pt"
    history_path = output_dirs["histories"] / f"{file_stem}_history.csv"

    history_rows: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = 0
    for epoch in range(1, int(retune_cfg["train_epochs"]) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            grad_clip=float(retune_cfg["grad_clip"]),
        )
        val_eval = evaluate_model(
            model=model,
            loader=loaders["val"],
            device=device,
            target_scaler=target_scaler,
            subset_lookup=subset_lookup,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "seq_len": int(seq_len),
                "learning_rate": float(learning_rate),
                "dropout": float(dropout),
                "train_loss": float(train_loss),
                "val_mae": float(val_eval["total_metrics"]["mae"]),
                "val_rmse": float(val_eval["total_metrics"]["rmse"]),
                "val_mse": float(val_eval["total_metrics"]["mse"]),
            }
        )
        if history_rows[-1]["val_mae"] < best_val_mae:
            best_val_mae = history_rows[-1]["val_mae"]
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seq_len": int(seq_len),
                    "pred_len": int(retune_cfg["pred_len"]),
                    "learning_rate": float(learning_rate),
                    "dropout": float(dropout),
                    "resolved_patch_len": int(model_configs.patch_len),
                    "resolved_stride": int(model_configs.stride),
                },
                checkpoint_path,
            )

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_eval = evaluate_model(
        model=model,
        loader=loaders["val"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
    )
    test_eval = evaluate_model(
        model=model,
        loader=loaders["test"],
        device=device,
        target_scaler=target_scaler,
        subset_lookup=subset_lookup,
    )

    return {
        "seq_len": int(seq_len),
        "pred_len": int(retune_cfg["pred_len"]),
        "learning_rate": float(learning_rate),
        "dropout": float(dropout),
        "resolved_patch_len": int(model_configs.patch_len),
        "resolved_stride": int(model_configs.stride),
        "best_epoch": int(best_epoch),
        "train_samples": int(split_sizes["train"]),
        "val_samples": int(split_sizes["val"]),
        "test_samples": int(split_sizes["test"]),
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
    }


def build_ranked_table(frame: pd.DataFrame, metric: str, top_k: int = 5) -> str:
    columns = ["seq_len", "learning_rate", "dropout", "train_samples", "val_samples", "test_samples", metric]
    top = frame.sort_values([metric, "seq_len", "learning_rate", "dropout"]).head(top_k).copy()
    lines = [
        f"| rank | seq_len | learning_rate | dropout | train_samples | val_samples | test_samples | {metric} |",
        f"| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {int(row.seq_len)} | {float(row.learning_rate):.0e} | {float(row.dropout):.1f} | {int(row.train_samples)} | {int(row.val_samples)} | {int(row.test_samples)} | {float(getattr(row, metric)):.6e} |"
        )
    return "\n".join(lines)


def build_augmented_reference_rank(results: pd.DataFrame, reference_row: dict[str, float], metric: str) -> tuple[int, int]:
    augmented = results[[metric]].copy()
    augmented["source"] = "retune"
    reference_value = pd.DataFrame([{metric: float(reference_row[metric]), "source": "formal_default"}])
    augmented = pd.concat([augmented, reference_value], ignore_index=True)
    augmented = augmented.sort_values([metric, "source"]).reset_index(drop=True)
    default_rank = int(augmented.index[augmented["source"] == "formal_default"][0]) + 1
    total = int(len(augmented))
    return default_rank, total


def determine_conclusion(
    best_row: pd.Series,
    references: dict[str, dict[str, float]],
    retune_cfg: dict[str, Any],
) -> dict[str, Any]:
    default_test = float(references["default_main"]["test_mae"])
    default_blast = float(references["default_blast"]["mae"])
    lstm_test = float(references["lstm_main"]["test_mae"])
    lstm_blast = float(references["lstm_blast"]["mae"])

    improvement_test = (default_test - float(best_row["test_mae"])) / default_test
    improvement_blast = (default_blast - float(best_row["blast_test_mae"])) / default_blast
    close_to_lstm_test = float(best_row["test_mae"]) <= lstm_test * (1.0 + float(retune_cfg["close_to_lstm_ratio"]))
    close_to_lstm_blast = float(best_row["blast_test_mae"]) <= lstm_blast * (1.0 + float(retune_cfg["close_to_lstm_ratio"]))

    meaningfully_better = improvement_test >= float(retune_cfg["meaningful_improvement_ratio"])
    blast_supported = improvement_blast >= float(retune_cfg["require_blast_improvement_ratio"])

    if meaningfully_better and blast_supported and (close_to_lstm_test or close_to_lstm_blast):
        verdict = "默认配置存在明显不公平"
    else:
        verdict = "更偏 backbone 不适配"

    return {
        "improvement_test_ratio": float(improvement_test),
        "improvement_blast_ratio": float(improvement_blast),
        "gap_to_lstm_test_ratio": float((float(best_row["test_mae"]) - lstm_test) / lstm_test),
        "gap_to_lstm_blast_ratio": float((float(best_row["blast_test_mae"]) - lstm_blast) / lstm_blast),
        "verdict": verdict,
    }


def write_report(
    *,
    output_path: Path,
    results: pd.DataFrame,
    references: dict[str, dict[str, float]],
    split_counts: dict[int, dict[str, int]],
    retune_cfg: dict[str, Any],
    device: torch.device,
) -> None:
    best_by_val = results.sort_values(["val_mae", "test_mae", "blast_test_mae"]).iloc[0]
    best_by_test = results.sort_values(["test_mae", "val_mae", "blast_test_mae"]).iloc[0]
    conclusion = determine_conclusion(best_by_test, references, retune_cfg)

    default_val_rank, total_rank_count = build_augmented_reference_rank(results, references["default_main"], "val_mae")
    default_test_rank, _ = build_augmented_reference_rank(results, references["default_main"], "test_mae")

    lines = [
        "# timefilter_retune_report",
        "",
        "## Run Context",
        "",
        "- 本轮只复核 `TimeFilter`，实验对象固定为正式 `A4` 数据版本。",
        "- split 固定为 `split_cand_01`，`patch_size=10`，`stable_background` 继续排除。",
        f"- 扫描网格：`seq_len={list(retune_cfg['seq_lens'])}`，`learning_rate={list(retune_cfg['learning_rates'])}`，`dropout={list(retune_cfg['dropouts'])}`。",
        f"- 其余设置固定：`pred_len={int(retune_cfg['pred_len'])}`、`epochs={int(retune_cfg['train_epochs'])}`、`batch_size={int(retune_cfg['batch_size'])}`、`weight_decay={float(retune_cfg['weight_decay'])}`、`grad_clip={float(retune_cfg['grad_clip'])}`。",
        f"- 训练设备：`{device}`。",
        "- 默认参考配置来自 `formal_round1`：`seq_len=24, learning_rate=1e-3, dropout=0.1`。",
        "",
        "## Split Check",
        "",
        "| seq_len | train_samples | val_samples | test_samples |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seq_len in sorted(split_counts):
        stats = split_counts[seq_len]
        lines.append(
            f"| {int(seq_len)} | {int(stats['train'])} | {int(stats['val'])} | {int(stats['test'])} |"
        )
    lines.extend(
        [
            "",
            "- 所有配置都使用同一正式 split 源和同一 `A4` 特征定义；`val/test` 样本数固定不变。",
            "- `train_samples` 会随着 `seq_len` 变长而自然减少，这是因果窗口在不做左侧 padding 的前提下的正式行为，不属于 split 漂移。",
            "",
            "## Reference Comparison",
            "",
            f"- 当前正式默认 `A4-TimeFilter`：`val_mae={float(references['default_main']['val_mae']):.6f}`，`test_mae={float(references['default_main']['test_mae']):.6f}`，`blast_test_mae={float(references['default_blast']['mae']):.6f}`。",
            f"- 当前正式 `A4-LSTM`：`val_mae={float(references['lstm_main']['val_mae']):.6f}`，`test_mae={float(references['lstm_main']['test_mae']):.6f}`，`blast_test_mae={float(references['lstm_blast']['mae']):.6f}`。",
            f"- 默认 `TimeFilter` 在“27 组 retune + 1 个正式默认参考”的扩展排序里：`val_mae` 排名 `{default_val_rank}/{total_rank_count}`，`test_mae` 排名 `{default_test_rank}/{total_rank_count}`。",
            "",
            "## Best Configs",
            "",
            f"- 按 `val_mae` 最优：`seq_len={int(best_by_val['seq_len'])}`、`learning_rate={float(best_by_val['learning_rate']):.0e}`、`dropout={float(best_by_val['dropout']):.1f}`，`val_mae={float(best_by_val['val_mae']):.6f}`，`test_mae={float(best_by_val['test_mae']):.6f}`。",
            f"- 按 `test_mae` 最优：`seq_len={int(best_by_test['seq_len'])}`、`learning_rate={float(best_by_test['learning_rate']):.0e}`、`dropout={float(best_by_test['dropout']):.1f}`，`val_mae={float(best_by_test['val_mae']):.6f}`，`test_mae={float(best_by_test['test_mae']):.6f}`，`blast_test_mae={float(best_by_test['blast_test_mae']):.6f}`。",
            "",
            f"- 相对默认 `TimeFilter` 的 `test_mae` 改善：`{conclusion['improvement_test_ratio'] * 100:.2f}%`。",
            f"- 相对默认 `TimeFilter` 的 `blast_test_mae` 改善：`{conclusion['improvement_blast_ratio'] * 100:.2f}%`。",
            f"- 相对 `A4-LSTM` 的 `test_mae` 差距：`{conclusion['gap_to_lstm_test_ratio'] * 100:.2f}%`。",
            f"- 相对 `A4-LSTM` 的 `blast_test_mae` 差距：`{conclusion['gap_to_lstm_blast_ratio'] * 100:.2f}%`。",
            "",
            "## Ranking by val_mae",
            "",
            build_ranked_table(results, "val_mae"),
            "",
            "## Ranking by test_mae",
            "",
            build_ranked_table(results, "test_mae"),
            "",
            "## Final Answer",
            "",
            f"- 是否明显优于当前正式默认 `TimeFilter`：{'是' if conclusion['improvement_test_ratio'] >= float(retune_cfg['meaningful_improvement_ratio']) else '否'}。",
            f"- 是否能接近或超过当前正式 `A4-LSTM`：{'是' if conclusion['gap_to_lstm_test_ratio'] <= float(retune_cfg['close_to_lstm_ratio']) else '否'}。",
            f"- 综合判断：`{conclusion['verdict']}`。",
            "",
            "## Decision Rule",
            "",
            f"- 本报告把“默认配置存在明显不公平”定义为：`test_mae` 改善至少 `{float(retune_cfg['meaningful_improvement_ratio']) * 100:.0f}%`，`blast_test_mae` 改善至少 `{float(retune_cfg['require_blast_improvement_ratio']) * 100:.0f}%`，且最优配置与 `A4-LSTM` 的 `test_mae` 或 `blast_test_mae` 差距不超过 `{float(retune_cfg['close_to_lstm_ratio']) * 100:.0f}%`。",
            "- 若上述条件不满足，则判定更偏 `backbone` 不适配，而不是单纯默认配置不公平。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    retune_cfg = dict(config["timefilter_retune_v1"])
    retune_cfg["model"] = dict(config["formal_round1"]["model"])

    set_seed(int(retune_cfg["seed"]))
    device = resolve_device(str(retune_cfg["device"]))

    output_root = Path(config["paths"]["output_dir"]) / str(retune_cfg["output_subdir"])
    output_dirs = ensure_output_dirs(output_root)
    formal_root = Path(config["paths"]["output_dir"]) / str(config["formal_round1"]["output_subdir"])

    bundle, _patch_meta, _missing_report = load_formal_window_bundle(
        Path(config["paths"]["dataset_dir"]),
        missing_hour_policy=str(retune_cfg["missing_hour_policy"]),
    )
    split_plan = load_split_plan(Path(config["paths"]["output_dir"]) / "experiment_plan_v1" / "experiment_split_plan_v1.csv")
    subset_flags = compute_subset_hour_flags(bundle)
    references = load_reference_rows(formal_root)

    feature_array, _feature_names = build_feature_tensor_by_experiment(bundle, str(retune_cfg["experiment"]))
    train_end_index = int(split_plan.loc[split_plan["split_name"] == "train", "effective_end_index_exclusive"].iloc[0])

    result_rows: list[dict[str, Any]] = []
    split_counts: dict[int, dict[str, int]] = {}

    for seq_len in [int(value) for value in retune_cfg["seq_lens"]]:
        manifest = build_formal_window_manifest(
            bundle.timestamps,
            split_plan,
            seq_len=seq_len,
            pred_len=int(retune_cfg["pred_len"]),
            missing_hour_policy=str(retune_cfg["missing_hour_policy"]),
        )
        subset_lookup = expand_subset_lookup(
            manifest,
            build_sample_subset_masks(manifest, subset_flags, int(retune_cfg["pred_len"])),
        )
        loaders, datasets, _feature_scaler, target_scaler = build_dataloaders(
            manifest=manifest,
            feature_array=feature_array,
            bundle=bundle,
            train_end_index=train_end_index,
            batch_size=int(retune_cfg["batch_size"]),
            num_workers=int(retune_cfg["num_workers"]),
        )
        split_sizes = {split_name: len(dataset) for split_name, dataset in datasets.items()}
        split_counts[seq_len] = dict(split_sizes)

        for learning_rate in [float(value) for value in retune_cfg["learning_rates"]]:
            for dropout in [float(value) for value in retune_cfg["dropouts"]]:
                row = train_single_config(
                    seq_len=seq_len,
                    learning_rate=learning_rate,
                    dropout=dropout,
                    feature_array=feature_array,
                    bundle=bundle,
                    loaders=loaders,
                    target_scaler=target_scaler,
                    subset_lookup=subset_lookup,
                    split_sizes=split_sizes,
                    device=device,
                    retune_cfg=retune_cfg,
                    output_dirs=output_dirs,
                )
                result_rows.append(row)

    results = pd.DataFrame(result_rows).sort_values(["seq_len", "learning_rate", "dropout"]).reset_index(drop=True)
    results.to_csv(output_root / "timefilter_retune_results.csv", index=False)

    write_report(
        output_path=output_root / "timefilter_retune_report.md",
        results=results,
        references=references,
        split_counts=split_counts,
        retune_cfg=retune_cfg,
        device=device,
    )

    print(output_root / "timefilter_retune_results.csv")
    print(output_root / "timefilter_retune_report.md")


if __name__ == "__main__":
    main()
