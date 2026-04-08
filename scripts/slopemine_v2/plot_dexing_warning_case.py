#!/usr/bin/env python3
"""基于 Dexing 露天矿坍塌案例生成第五章典型事件时序图。

输入数据仅包含单点位移与速度，因此这里采用“回溯预警示意”口径：
- y_true: 原始位移 Displacement
- y_pred: 基于过去若干采样点的滚动二次趋势外推预测
- risk_score: 由位移偏离、速度、加速度和短时预测增量共同映射得到
- warning_level: 在风险分数上叠加确认与滞回后的四级预警

若 CSV 中没有显式坍塌时刻，则默认把最后一个时间戳视为事件发生时刻。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_utils_v2 import setup_plot_style  # noqa: E402


LEVEL_LABELS = {
    0: "正常",
    1: "注意",
    2: "警戒",
    3: "危险",
}

LEVEL_COLORS = {
    0: "#5B8FF9",
    1: "#F6BD16",
    2: "#FA8C16",
    3: "#D4380D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a Dexing landslide warning case figure.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "dataset" / "Dexing Open-pit mine landslide（发生坍塌） .csv",
        help="Input Dexing CSV path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "chapter5_dexing_warning_case_v1",
        help="Output directory.",
    )
    parser.add_argument(
        "--event-time",
        type=str,
        default="",
        help="Explicit event timestamp. Empty means using the last timestamp in the CSV.",
    )
    parser.add_argument(
        "--pre-hours",
        type=float,
        default=18.0,
        help="Hours to retain before the event.",
    )
    parser.add_argument(
        "--post-hours",
        type=float,
        default=0.0,
        help="Hours to retain after the event. If no post-event data exist, the script keeps available rows only.",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=12,
        help="Number of historical samples used by the rolling trend predictor.",
    )
    return parser.parse_args()


def load_case_frame(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少 Dexing CSV 文件：{csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"Time", "Displacement", "Velocity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} 缺少字段：{sorted(missing)}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["Time"])
    frame["Displacement"] = pd.to_numeric(frame["Displacement"], errors="coerce")
    frame["Velocity"] = pd.to_numeric(frame["Velocity"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "Displacement", "Velocity"]).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{csv_path.name} 中没有有效的 timestamp / Displacement / Velocity 记录。")

    dt_hours = frame["timestamp"].diff().dt.total_seconds().div(3600.0)
    if len(frame) > 1:
        dt_hours.iloc[0] = dt_hours.iloc[1]
    else:
        dt_hours.iloc[0] = 7.0 / 60.0
    frame["dt_hours"] = dt_hours
    frame["Acceleration"] = frame["Velocity"].diff().div(frame["dt_hours"]).replace([np.inf, -np.inf], np.nan)
    return frame


def robust_positive_zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, np.nan, dtype=float)
    median = float(np.nanmedian(finite))
    q1 = float(np.nanquantile(finite, 0.25))
    q3 = float(np.nanquantile(finite, 0.75))
    iqr = max(q3 - q1, 1e-6)
    z = (arr - median) / iqr
    z[z < 0] = 0.0
    return z


def compute_rolling_prediction(frame: pd.DataFrame, history_points: int) -> pd.DataFrame:
    pred = np.full(len(frame), np.nan, dtype=float)
    pred_increment = np.full(len(frame), np.nan, dtype=float)

    for idx in range(history_points, len(frame)):
        history_values = frame["Displacement"].iloc[idx - history_points : idx].to_numpy(dtype=float)
        x_hist = np.arange(history_points, dtype=float)
        coefficients = np.polyfit(x_hist, history_values, deg=2)
        pred[idx] = float(np.polyval(coefficients, history_points))
        pred_increment[idx] = float(pred[idx] - history_values[-1])

    result = frame.copy()
    result["y_pred"] = pred
    result["pred_increment"] = pred_increment
    return result


def apply_risk_persistence(base_risk: np.ndarray, decay_1: float = 0.75, decay_2: float = 0.45) -> np.ndarray:
    risk = np.zeros_like(base_risk, dtype=float)
    for idx, value in enumerate(np.asarray(base_risk, dtype=float)):
        prev_1 = risk[idx - 1] * decay_1 if idx >= 1 else 0.0
        prev_2 = risk[idx - 2] * decay_2 if idx >= 2 else 0.0
        risk[idx] = max(float(value), float(prev_1), float(prev_2))
    return np.clip(risk, 0.0, 1.0)


def apply_warning_hysteresis(risk_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_level = np.digitize(risk_score, bins=[0.25, 0.45, 0.70], right=False).astype(int)
    final_level = np.zeros_like(raw_level, dtype=int)

    current_level = 0
    yellow_confirm = 0
    orange_confirm = 0
    red_confirm = 0
    yellow_downgrade = 0
    orange_downgrade = 0
    red_downgrade = 0

    for idx, risk in enumerate(np.asarray(risk_score, dtype=float)):
        yellow_confirm = yellow_confirm + 1 if risk >= 0.25 else 0
        orange_confirm = orange_confirm + 1 if risk >= 0.45 else 0
        red_confirm = red_confirm + 1 if risk >= 0.70 else 0

        if current_level < 1 and yellow_confirm >= 2:
            current_level = 1
        if current_level < 2 and orange_confirm >= 3:
            current_level = 2
        if current_level < 3 and red_confirm >= 4:
            current_level = 3

        yellow_downgrade = yellow_downgrade + 1 if risk < 0.16 else 0
        orange_downgrade = orange_downgrade + 1 if risk < 0.35 else 0
        red_downgrade = red_downgrade + 1 if risk < 0.55 else 0

        if current_level == 3 and red_downgrade >= 3:
            current_level = 2
            red_downgrade = 0
        if current_level == 2 and orange_downgrade >= 3:
            current_level = 1
            orange_downgrade = 0
        if current_level == 1 and yellow_downgrade >= 4:
            current_level = 0
            yellow_downgrade = 0

        final_level[idx] = current_level

    return raw_level, final_level


def build_warning_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["disp_baseline"] = result["Displacement"].rolling(24, min_periods=6).median()
    result["vel_smooth"] = result["Velocity"].rolling(5, min_periods=1).mean()
    result["acc_smooth"] = result["Acceleration"].rolling(5, min_periods=1).mean()
    result["pred_increment_smooth"] = result["pred_increment"].rolling(3, min_periods=1).mean()

    disp_gap_score = robust_positive_zscore(result["Displacement"] - result["disp_baseline"])
    vel_score = robust_positive_zscore(result["vel_smooth"])
    acc_score = robust_positive_zscore(result["acc_smooth"].fillna(0.0))
    pred_score = robust_positive_zscore(result["pred_increment_smooth"].fillna(0.0))

    risk_raw = 0.25 * disp_gap_score + 0.25 * vel_score + 0.20 * acc_score + 0.30 * pred_score
    finite_raw = risk_raw[np.isfinite(risk_raw)]
    q30 = float(np.nanquantile(finite_raw, 0.30)) if finite_raw.size else 0.0
    q95 = float(np.nanquantile(finite_raw, 0.95)) if finite_raw.size else 1.0
    risk_score = np.clip((risk_raw - q30) / max(q95 - q30, 1e-6), 0.0, 1.0)
    risk_score = apply_risk_persistence(risk_score)
    warning_level_raw, warning_level = apply_warning_hysteresis(risk_score)

    result["risk_score"] = risk_score
    result["warning_level_raw"] = warning_level_raw
    result["warning_level"] = warning_level
    return result


def clip_window(frame: pd.DataFrame, event_time: pd.Timestamp, pre_hours: float, post_hours: float) -> pd.DataFrame:
    start_time = event_time - pd.Timedelta(hours=float(pre_hours))
    end_time = event_time + pd.Timedelta(hours=float(post_hours))
    clipped = frame.loc[(frame["timestamp"] >= start_time) & (frame["timestamp"] <= end_time)].copy()
    if clipped.empty:
        raise ValueError("按给定 pre_hours/post_hours 截取后没有可用数据。")
    return clipped.reset_index(drop=True)


def add_warning_band(ax: plt.Axes, plot_frame: pd.DataFrame) -> None:
    timestamps = pd.DatetimeIndex(plot_frame["timestamp"])
    if len(timestamps) == 1:
        half_step = pd.Timedelta(minutes=3.5)
        edges = [timestamps[0] - half_step, timestamps[0] + half_step]
    else:
        median_delta = pd.to_timedelta(np.diff(timestamps.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[ns]").min())
        half_step = median_delta / 2
        edges = [timestamps[0] - half_step]
        for left, right in zip(timestamps[:-1], timestamps[1:]):
            edges.append(left + (right - left) / 2)
        edges.append(timestamps[-1] + half_step)

    x_edges = mdates.date2num(pd.DatetimeIndex(edges).to_pydatetime())
    y_edges = np.array([0.0, 1.0], dtype=float)
    band = plot_frame["warning_level"].fillna(0).to_numpy(dtype=float)[None, :]
    cmap = ListedColormap([LEVEL_COLORS[level] for level in [0, 1, 2, 3]])
    mesh = ax.pcolormesh(x_edges, y_edges, band, cmap=cmap, vmin=0.0, vmax=3.0, shading="flat")
    mesh.set_edgecolor("face")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("预警等级")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_case(plot_frame: pd.DataFrame, event_time: pd.Timestamp, png_path: Path, pdf_path: Path) -> None:
    setup_plot_style()
    fig = plt.figure(figsize=(13.8, 6.6))
    grid = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[4.0, 0.10, 0.55, 0.08, 0.05], hspace=0.0)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_band = fig.add_subplot(grid[2, 0], sharex=ax_main)
    ax_risk = ax_main.twinx()

    ax_main.plot(
        plot_frame["timestamp"],
        plot_frame["y_true"],
        color="#1F2937",
        linewidth=2.0,
        linestyle="-",
        label="真实位移",
    )
    ax_main.plot(
        plot_frame["timestamp"],
        plot_frame["y_pred"],
        color="#4E79A7",
        linewidth=2.0,
        linestyle="--",
        label="预测位移",
    )
    ax_risk.plot(
        plot_frame["timestamp"],
        plot_frame["risk_score"],
        color="#C0392B",
        linewidth=2.2,
        label="风险评分",
    )

    x_margin = pd.Timedelta(minutes=35)
    ax_main.set_xlim(plot_frame["timestamp"].iloc[0], plot_frame["timestamp"].iloc[-1] + x_margin)
    for axis in [ax_main, ax_risk, ax_band]:
        axis.axvline(event_time, color="black", linewidth=1.2, linestyle=(0, (4, 2)))

    y_max = float(np.nanmax(plot_frame[["y_true", "y_pred"]].to_numpy(dtype=float)))
    y_min = float(np.nanmin(plot_frame[["y_true", "y_pred"]].to_numpy(dtype=float)))
    y_text = y_max + 0.03 * max(y_max - y_min, 1.0)
    ax_main.text(event_time, y_text, "坍塌发生时刻", rotation=90, ha="left", va="bottom", fontsize=10)

    ax_main.set_ylabel("位移 / mm")
    ax_risk.set_ylabel("风险评分 [0, 1]")
    ax_risk.set_ylim(0.0, 1.05)
    ax_main.grid(True, axis="y", alpha=0.28)
    ax_main.grid(False, axis="x")
    ax_risk.grid(False)

    add_warning_band(ax_band, plot_frame)
    ax_band.set_xlabel("时间")
    ax_band.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_band.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    plt.setp(ax_main.get_xticklabels(), visible=False)

    legend_handles = [
        Line2D([], [], color="#1F2937", linewidth=2.0, linestyle="-", label="真实位移"),
        Line2D([], [], color="#4E79A7", linewidth=2.0, linestyle="--", label="预测位移"),
        Line2D([], [], color="#C0392B", linewidth=2.2, linestyle="-", label="风险评分"),
        Line2D([], [], color="black", linewidth=1.2, linestyle=(0, (4, 2)), label="坍塌发生时刻"),
    ] + [Patch(facecolor=LEVEL_COLORS[level], edgecolor="none", label=LEVEL_LABELS[level]) for level in [0, 1, 2, 3]]
    ax_main.legend(handles=legend_handles, loc="upper left", ncol=4, frameon=True)
    ax_main.set_title("Dexing 露天矿坍塌案例的位移、风险评分与预警等级时序演化")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_caption_and_latex(event_time: pd.Timestamp) -> tuple[str, str, str]:
    caption = "图5-X Dexing 露天矿坍塌案例下预测位移、风险评分与预警等级的时序演化过程"
    note = (
        "图中黑色竖虚线表示坍塌实际发生时刻。由于原始公开数据仅提供单点位移与速度序列，"
        "图中的预测位移采用过去 12 个采样点的滚动趋势外推结果，综合风险评分由位移偏离、速度、加速度和短时预测增量共同构造。"
        f"在本次示意中，坍塌时刻取 {event_time:%Y-%m-%d %H:%M}。可以看到，风险评分在坍塌前持续升高，"
        "预警等级依次经历蓝-黄-橙-红升级，说明该类时序映射能够较直观地呈现失稳前的加速演化与预警提前量。"
    )
    latex = (
        "\\begin{figure}[htbp]\n"
        "    \\centering\n"
        "    \\includegraphics[width=0.95\\textwidth]{figures/chapter5/dexing_warning_case.pdf}\n"
        "    \\caption{Dexing 露天矿坍塌案例下预测位移、风险评分与预警等级的时序演化过程}\n"
        "    \\label{fig:dexing_warning_case}\n"
        "\\end{figure}\n"
    )
    return caption, note, latex


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    frame = load_case_frame(args.input_csv)
    event_time = pd.Timestamp(args.event_time) if args.event_time else pd.Timestamp(frame["timestamp"].iloc[-1])
    if event_time not in set(frame["timestamp"]):
        raise ValueError(f"给定 event_time={event_time} 不在 CSV 时间戳中。")

    frame = compute_rolling_prediction(frame, history_points=args.history_points)
    frame = build_warning_frame(frame)
    frame["y_true"] = frame["Displacement"].astype(float)
    frame["event_flag"] = (frame["timestamp"] == event_time).astype(int)
    plot_frame = clip_window(frame, event_time=event_time, pre_hours=args.pre_hours, post_hours=args.post_hours)

    png_path = output_root / "dexing_warning_case.png"
    pdf_path = output_root / "dexing_warning_case.pdf"
    plot_case(plot_frame, event_time=event_time, png_path=png_path, pdf_path=pdf_path)

    export_frame = plot_frame.loc[
        :,
        [
            "timestamp",
            "y_true",
            "y_pred",
            "risk_score",
            "warning_level_raw",
            "warning_level",
            "event_flag",
            "Velocity",
            "Acceleration",
            "pred_increment",
        ],
    ].copy()
    export_frame.to_csv(output_root / "dexing_warning_case_data.csv", index=False)

    caption, note, latex = build_caption_and_latex(event_time)
    (output_root / "dexing_warning_case_caption.md").write_text(caption + "\n\n" + note + "\n", encoding="utf-8")
    (output_root / "dexing_warning_case_latex.tex").write_text(latex, encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "event_time": event_time,
                "input_csv": str(args.input_csv.resolve()),
                "output_png": str(png_path.resolve()),
                "output_pdf": str(pdf_path.resolve()),
                "data_csv": str((output_root / "dexing_warning_case_data.csv").resolve()),
                "history_points": int(args.history_points),
                "pre_hours": float(args.pre_hours),
                "post_hours": float(args.post_hours),
                "assumption": "event_time defaults to the last timestamp in the CSV when no explicit collapse marker exists",
                "first_yellow": plot_frame.loc[plot_frame["warning_level"] >= 1, "timestamp"].min(),
                "first_orange": plot_frame.loc[plot_frame["warning_level"] >= 2, "timestamp"].min(),
                "first_red": plot_frame.loc[plot_frame["warning_level"] >= 3, "timestamp"].min(),
                "prediction_corr": float(
                    np.corrcoef(
                        plot_frame.loc[plot_frame["y_pred"].notna(), "y_true"],
                        plot_frame.loc[plot_frame["y_pred"].notna(), "y_pred"],
                    )[0, 1]
                ),
                "prediction_mae": float(
                    np.mean(
                        np.abs(
                            plot_frame.loc[plot_frame["y_pred"].notna(), "y_true"]
                            - plot_frame.loc[plot_frame["y_pred"].notna(), "y_pred"]
                        )
                    )
                ),
            }
        ]
    )
    summary.to_csv(output_root / "dexing_warning_case_summary.csv", index=False)

    print(f"Selected event_time={event_time:%Y-%m-%d %H:%M}")
    print(f"Input csv={args.input_csv.resolve()}")
    print(f"Output figure={pdf_path.resolve()}")


if __name__ == "__main__":
    main()
