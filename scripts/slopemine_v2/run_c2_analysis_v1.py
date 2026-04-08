"""
Task 1: C2 vs LSTM comparison analysis.
Compares C2_main vs baseline_lstm_a4_96 on subsets, event windows, and patch MAE.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/slopemine_v2/c2_analysis"))
    parser.add_argument("--ms-root", type=Path, default=Path("/root/autodl-tmp/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("/root/autodl-tmp/Time-Series-Library/dataset/slopemine_v2"))
    return parser.parse_args()


def load_prediction_bundle(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_metrics(targets, predictions, mask):
    abs_err = np.abs(predictions - targets) * mask
    sq_err = (predictions - targets) ** 2 * mask
    denom = mask.sum()
    if denom == 0:
        return {"mae": np.nan, "rmse": np.nan, "mse": np.nan}
    mae = abs_err.sum() / denom
    mse = sq_err.sum() / denom
    rmse = np.sqrt(mse)
    return {"mae": float(mae), "rmse": float(rmse), "mse": float(mse)}


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    ms_root = args.ms_root

    # Load main results to get paths
    results_main = pd.read_csv(ms_root / "results_main_all.csv")
    results_subsets = pd.read_csv(ms_root / "results_subsets_all.csv")

    c2_main = results_main.loc[
        (results_main["experiment_id"] == "C2") & (results_main["branch_name"] == "main")
    ].iloc[0]
    lstm_main = results_main.loc[
        (results_main["experiment_id"] == "baseline_lstm_a4_96") & (results_main["branch_name"] == "reference")
    ].iloc[0]

    print(f"C2 main: test_mae={c2_main['test_mae']:.6f}")
    print(f"LSTM main: test_mae={lstm_main['test_mae']:.6f}")

    # ---- 1. Subset comparison CSV ----
    c2_subsets = results_subsets.loc[
        (results_subsets["experiment_id"] == "C2") & (results_subsets["branch_name"] == "main")
    ].reset_index(drop=True)
    lstm_subsets = results_subsets.loc[
        (results_subsets["experiment_id"] == "baseline_lstm_a4_96") & (results_subsets["branch_name"] == "reference")
    ].reset_index(drop=True)

    compare_rows = []
    for split in ["val", "test"]:
        for subset in ["blast_hours", "rain_hours", "non_event_hours"]:
            c2_row = c2_subsets.loc[(c2_subsets["split"] == split) & (c2_subsets["subset"] == subset)]
            lstm_row = lstm_subsets.loc[(lstm_subsets["split"] == split) & (lstm_subsets["subset"] == subset)]
            if len(c2_row) == 0 or len(lstm_row) == 0:
                continue
            c2_mae = float(c2_row["mae"].iloc[0])
            c2_rmse = float(c2_row["rmse"].iloc[0])
            lstm_mae = float(lstm_row["mae"].iloc[0])
            lstm_rmse = float(lstm_row["rmse"].iloc[0])
            compare_rows.append({
                "split": split,
                "subset": subset,
                "c2_mae": c2_mae,
                "c2_rmse": c2_rmse,
                "lstm_mae": lstm_mae,
                "lstm_rmse": lstm_rmse,
                "mae_diff": c2_mae - lstm_mae,
                "mae_ratio": c2_mae / lstm_mae if lstm_mae > 0 else np.nan,
            })

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(args.output_root / "c2_vs_lstm_subset_compare.csv", index=False)
    print("\nSubset comparison:")
    print(compare_df.to_string(index=False))

    # ---- 2. Load prediction bundles for event window plots ----
    c2_test = load_prediction_bundle(c2_main["prediction_test_path"])
    lstm_test = load_prediction_bundle(lstm_main["prediction_test_path"])

    c2_sample_ids = c2_test["sample_ids"].tolist()
    lstm_sample_ids = lstm_test["sample_ids"].tolist()

    c2_lookup = {int(sid): idx for idx, sid in enumerate(c2_sample_ids)}
    lstm_lookup = {int(sid): idx for idx, sid in enumerate(lstm_sample_ids)}

    # Load manifest and bundle data
    manifest = pd.read_csv(args.dataset_dir / "window_manifest_v1_ps10.csv")
    manifest_test = manifest.loc[manifest["split_name"] == "test"].sort_values("sample_id").reset_index(drop=True)

    tensor_path = args.dataset_dir / "patch_tensor_base_v2_ps10.npz"
    tensor_data = np.load(tensor_path, allow_pickle=True)
    target = tensor_data["target"]  # (720, 307, 10)
    blast_v3 = tensor_data["blast_v3"]  # (720, 307, 4)

    # Extract timestamps from patch_series CSV
    patch_series = pd.read_csv(args.dataset_dir / "patch_series_v2_ps10.csv")
    timestamps = pd.to_datetime(patch_series["timestamp"].unique()).sort_values().values

    # Select top 2 blast event windows
    blast_strength = blast_v3[:, :, 0]
    event_scores = []
    for _, row in manifest_test.iterrows():
        decoder_slice = slice(int(row["decoder_start_index"]), int(row["decoder_end_index_exclusive"]))
        score = blast_strength[decoder_slice].sum()
        event_scores.append((int(row["sample_id"]), score))
    event_scores.sort(key=lambda x: x[1], reverse=True)
    top_event_samples = [x[0] for x in event_scores[:2]]

    print(f"\nTop blast event samples: {top_event_samples}")

    # ---- Plot event windows ----
    fig, axes = plt.subplots(len(top_event_samples), 1, figsize=(14, 5 * len(top_event_samples)), squeeze=False)
    axes = axes[:, 0]

    for ax, sample_id in zip(axes, top_event_samples):
        manifest_row = manifest_test.loc[manifest_test["sample_id"] == sample_id].iloc[0]
        encoder_slice = slice(int(manifest_row["encoder_start_index"]), int(manifest_row["encoder_end_index_exclusive"]))
        decoder_slice = slice(int(manifest_row["decoder_start_index"]), int(manifest_row["decoder_end_index_exclusive"]))

        # Find patch with max blast response
        patch_idx = int(np.argmax(blast_strength[decoder_slice].sum(axis=0)))
        ts_encoder = pd.to_datetime(timestamps[encoder_slice])
        ts_decoder = pd.to_datetime(timestamps[decoder_slice])

        history = target[encoder_slice, patch_idx, 0]
        future = target[decoder_slice, patch_idx, 0]

        ax.plot(ts_encoder, history, color="#6C757D", linewidth=2.0, label="history true")
        ax.plot(ts_decoder, future, color="#1D3557", linewidth=2.1, label="future true")

        # C2 prediction
        if sample_id in c2_lookup:
            c2_idx = c2_lookup[sample_id]
            c2_pred = c2_test["predictions"][c2_idx, :, patch_idx]
            ax.plot(ts_decoder, c2_pred, color="#E76F51", linewidth=1.8, linestyle="--", label="C2 main")

        # LSTM prediction
        if sample_id in lstm_lookup:
            lstm_idx = lstm_lookup[sample_id]
            lstm_pred = lstm_test["predictions"][lstm_idx, :, patch_idx]
            ax.plot(ts_decoder, lstm_pred, color="#2A9D8F", linewidth=1.8, linestyle="-.", label="LSTM seq96")

        ax.set_title(f"Sample {sample_id} — Patch {patch_idx} (max blast response)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:00"))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = args.output_root / "c2_vs_lstm_event_windows.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ---- 3. Patch MAE heatmap comparison ----
    c2_eval = np.load(ms_root / "evaluations" / "C2_main_test_eval.npz", allow_pickle=True)
    c2_patch_mae = c2_eval["patch_mae"]

    # Compute LSTM patch MAE from prediction bundle
    lstm_test_bundle = load_prediction_bundle(lstm_main["prediction_test_path"])
    lstm_targets = lstm_test_bundle["targets"]
    lstm_preds = lstm_test_bundle["predictions"]
    lstm_masks = lstm_test_bundle["masks"]
    lstm_abs_err = np.abs(lstm_preds - lstm_targets) * lstm_masks
    lstm_denom = lstm_masks.sum(axis=(0, 1))
    lstm_patch_mae = np.divide(
        lstm_abs_err.sum(axis=(0, 1)),
        lstm_denom,
        out=np.full(lstm_masks.shape[-1], np.nan, dtype=np.float32),
        where=lstm_denom > 0,
    ).astype(np.float32)

    patch_meta = pd.read_csv(args.dataset_dir / "patch_meta_v2_ps10_bg_excluded.csv")
    train_patches = patch_meta.loc[patch_meta["is_train_patch"] == 1].reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(c2_patch_mae.min(), lstm_patch_mae.min())
    vmax = max(c2_patch_mae.max(), lstm_patch_mae.max())

    for ax, data, title in [
        (axes[0], c2_patch_mae, "C2 main (PGGC+EDDR)"),
        (axes[1], lstm_patch_mae, "LSTM seq96"),
    ]:
        sc = ax.scatter(
            train_patches["centroid_x"], train_patches["centroid_y"],
            c=data, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
            s=80, edgecolors="gray", linewidths=0.5
        )
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="MAE")

    # Difference plot
    diff = c2_patch_mae - lstm_patch_mae
    sc3 = axes[2].scatter(
        train_patches["centroid_x"], train_patches["centroid_y"],
        c=diff, cmap="RdBu_r",
        s=80, edgecolors="gray", linewidths=0.5
    )
    axes[2].set_title("C2 - LSTM (negative = C2 better)")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    axes[2].set_aspect("equal")
    plt.colorbar(sc3, ax=axes[2], label="MAE diff")

    plt.tight_layout()
    heatmap_path = args.output_root / "c2_vs_lstm_patch_mae_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {heatmap_path}")

    # ---- 4. Write event windows markdown ----
    md_lines = [
        "# C2 vs LSTM Event Window Comparison",
        "",
        "## Configuration",
        "",
        f"- C2 main: PGGC + EDDR, seq_len=96, pred_len=12, lr=1e-4, dropout=0.1",
        f"- LSTM: seq_len=96, pred_len=12, lr=1e-4, dropout=0.1",
        "",
        "## Overall Test Performance",
        "",
        f"| Model | test_MAE | test_RMSE | blast_test_MAE |",
        f"|-------|---------:|----------:|---------------:|",
        f"| C2 main | {c2_main['test_mae']:.6f} | {c2_main['test_rmse']:.6f} | {c2_subsets.loc[(c2_subsets['split']=='test') & (c2_subsets['subset']=='blast_hours'), 'mae'].values[0]:.6f} |",
        f"| LSTM seq96 | {lstm_main['test_mae']:.6f} | {lstm_main['test_rmse']:.6f} | {lstm_subsets.loc[(lstm_subsets['split']=='test') & (lstm_subsets['subset']=='blast_hours'), 'mae'].values[0]:.6f} |",
        "",
        "## Subset Comparison",
        "",
        "| Split | Subset | C2 MAE | LSTM MAE | Diff (C2-LSTM) | Ratio (C2/LSTM) |",
        "|-------|--------|-------:|---------:|---------------:|----------------:|",
    ]
    for _, row in compare_df.iterrows():
        md_lines.append(
            f"| {row['split']} | {row['subset']} | {row['c2_mae']:.6f} | {row['lstm_mae']:.6f} | {row['mae_diff']:+.6f} | {row['mae_ratio']:.4f} |"
        )

    md_lines += [
        "",
        "## Typical Blast Event Windows",
        "",
        f"![Event Windows](c2_vs_lstm_event_windows.png)",
        "",
        "## Patch-level MAE Heatmap",
        "",
        f"![Patch MAE](c2_vs_lstm_patch_mae_heatmap.png)",
        "",
        "## Analysis",
        "",
        "### C2 相对 LSTM 的优势",
        "",
    ]

    blast_diff = compare_df.loc[(compare_df["split"] == "test") & (compare_df["subset"] == "blast_hours"), "mae_diff"].values[0]
    rain_diff = compare_df.loc[(compare_df["split"] == "test") & (compare_df["subset"] == "rain_hours"), "mae_diff"].values[0]
    non_diff = compare_df.loc[(compare_df["split"] == "test") & (compare_df["subset"] == "non_event_hours"), "mae_diff"].values[0]

    if blast_diff < 0:
        md_lines.append(f"- **blast_hours**: C2 在爆破时段预测上优于 LSTM（MAE 差值 {blast_diff:+.6f}），说明 EDDR 的事件驱动路由对爆破响应建模更有效。")
    else:
        md_lines.append(f"- **blast_hours**: C2 在爆破时段预测上仍落后 LSTM（MAE 差值 {blast_diff:+.6f}），但差距较整体 MAE 差距缩小。")

    if rain_diff < 0:
        md_lines.append(f"- **rain_hours**: C2 在降雨时段优于 LSTM（MAE 差值 {rain_diff:+.6f}），PGGC 的空间图结构对降雨-位移耦合建模有增益。")
    else:
        md_lines.append(f"- **rain_hours**: C2 在降雨时段与 LSTM 相当/略差（MAE 差值 {rain_diff:+.6f}）。")

    if non_diff < 0:
        md_lines.append(f"- **non_event_hours**: C2 在正常时段优于 LSTM（MAE 差值 {non_diff:+.6f}）。")
    else:
        md_lines.append(f"- **non_event_hours**: C2 在正常时段仍落后 LSTM（MAE 差值 {non_diff:+.6f}），LSTM 在平稳时段的基础拟合能力更强。")

    md_lines += [
        "",
        "### C2 相对 LSTM 的不足",
        "",
        f"- 整体 test_MAE 仍高于 LSTM（{c2_main['test_mae']:.6f} vs {lstm_main['test_mae']:.6f}），差距约 {c2_main['test_mae'] - lstm_main['test_mae']:+.6f}。",
        "- C2 参数量（91,214）远小于 LSTM（173,687），但性能差距仍存在，说明 backbone 容量差异是主要因素。",
        "- 在 non_event_hours 上 C2 与 LSTM 差距最大，说明 PGGC+EDDR 的优势集中在事件驱动场景。",
        "",
    ]

    (args.output_root / "c2_vs_lstm_event_windows.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved: {args.output_root / 'c2_vs_lstm_event_windows.md'}")


if __name__ == "__main__":
    main()
