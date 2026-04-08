"""生成边坡数据集的论文级多源可视化图件。

图件设计目标：
1. 覆盖全部内部监测特征（patch 级 10 个指标）。
2. 覆盖天气 4 个特征与全局爆破 7 个特征。
3. 给出内外因子耦合热力图，辅助论文中的机理解释。
4. 给出 patch 空间敏感性图，展示不同区域对降雨/爆破的响应差异。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/time_series_matplotlib_cache")

from matplotlib import colors
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


INTERNAL_FEATURE_SPECS = [
    {"key": "disp_mean", "label": "位移均值", "color": "#0F4C5C"},
    {"key": "disp_std", "label": "位移标准差", "color": "#0F4C5C"},
    {"key": "disp_p95", "label": "位移 95 分位", "color": "#0F4C5C"},
    {"key": "disp_max", "label": "位移最大值", "color": "#0F4C5C"},
    {"key": "vel_mean", "label": "速度均值", "color": "#4C6E91"},
    {"key": "vel_p95", "label": "速度 95 分位", "color": "#4C6E91"},
    {"key": "acc_mean", "label": "加速度均值", "color": "#8C5E34"},
    {"key": "acc_p95", "label": "加速度 95 分位", "color": "#8C5E34"},
    {"key": "active_ratio", "label": "活跃点比例", "color": "#2A7F62"},
    {"key": "valid_ratio", "label": "有效点比例", "color": "#2A7F62"},
]

WEATHER_FEATURE_SPECS = [
    {"key": "temperature", "label": "气温", "color": "#D1495B"},
    {"key": "humidity", "label": "湿度", "color": "#577590"},
    {"key": "wind_speed", "label": "风速", "color": "#277DA1"},
    {"key": "rainfall", "label": "降雨量", "color": "#4D908E"},
]

BLAST_FEATURE_SPECS = [
    {"key": "blast_count", "label": "爆破次数", "color": "#C97B2A"},
    {"key": "total_charge_kg", "label": "总装药量", "color": "#E76F51"},
    {"key": "mean_charge_kg", "label": "平均装药量", "color": "#F4A261"},
    {"key": "max_ppv_est_mm_s", "label": "最大 PPV 估计", "color": "#B56576"},
    {"key": "disturbance_index", "label": "扰动指数", "color": "#9C6644"},
    {"key": "blast_count_6h", "label": "近 6h 爆破次数", "color": "#BC6C25"},
    {"key": "charge_24h_kg", "label": "近 24h 累积装药量", "color": "#CA6702"},
]

RAIN_SPAN_COLOR = "#9CC4E4"
BLAST_SPAN_COLOR = "#F7B267"
INTERQUARTILE_COLOR = "#B8D8E3"
MISSING_FACE_COLOR = "#ECECEC"
ZONE_TEXT_COLOR = "#263238"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready slopemine dataset figures.")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="paper_dataset_figures_v1",
        help="Sub-directory under outputs/slopemine_v2/ for generated figures.",
    )
    return parser.parse_args()


def setup_plot_style() -> None:
    """按用户要求固定中文字体，并统一论文风格。"""

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.12
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["grid.color"] = "#D0D0D0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def resolve_paths(output_subdir: str) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_v1_dir = repo_root / "dataset" / "slopemine"
    dataset_v2_dir = repo_root / "dataset" / "slopemine_v2"
    output_root = repo_root / "outputs" / "slopemine_v2" / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": repo_root,
        "dataset_v1_dir": dataset_v1_dir,
        "dataset_v2_dir": dataset_v2_dir,
        "output_root": output_root,
    }


def make_event_spans(timestamps: pd.Series, flags: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """把小时级布尔事件序列压缩成连续时间段。"""

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None
    hour_delta = pd.Timedelta(hours=1)

    for timestamp, flag in zip(timestamps.tolist(), flags.tolist()):
        timestamp = pd.Timestamp(timestamp)
        if bool(flag):
            if start is None:
                start = timestamp
                prev = timestamp
                continue
            if prev is not None and timestamp - prev == hour_delta:
                prev = timestamp
                continue
            spans.append((start, prev + hour_delta if prev is not None else start + hour_delta))
            start = timestamp
            prev = timestamp
            continue

        if start is not None:
            spans.append((start, prev + hour_delta if prev is not None else start + hour_delta))
            start = None
            prev = None

    if start is not None:
        spans.append((start, prev + hour_delta if prev is not None else start + hour_delta))

    return spans


def format_time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=0)
    ax.margins(x=0)


def safe_spearman(x_values: pd.Series, y_values: pd.Series) -> float:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    if len(frame) < 3:
        return float("nan")
    if frame["x"].nunique() <= 1 or frame["y"].nunique() <= 1:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="spearman"))


def load_datasets(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    dataset_v1_dir = paths["dataset_v1_dir"]
    dataset_v2_dir = paths["dataset_v2_dir"]

    patch_meta_path = dataset_v2_dir / "patch_meta_v2_ps10_bg_excluded.csv"
    if not patch_meta_path.exists():
        raise FileNotFoundError(f"未找到 patch 元数据文件：{patch_meta_path}")

    patch_meta = pd.read_csv(patch_meta_path)
    patch_series = pd.read_csv(dataset_v2_dir / "patch_series_v2_ps10.csv", parse_dates=["timestamp"])
    blast_patch = pd.read_csv(dataset_v2_dir / "patch_blast_features_v3_ps10.csv", parse_dates=["timestamp"])
    external = pd.read_csv(dataset_v1_dir / "slopemine_full.csv", parse_dates=["date"]).rename(columns={"date": "timestamp"})

    patch_series = patch_series.merge(
        patch_meta[
            [
                "patch_id",
                "zone_id_v2",
                "zone_label_v2",
                "cell_x_min",
                "cell_x_max",
                "cell_y_min",
                "cell_y_max",
                "cell_width",
                "cell_height",
                "centroid_x",
                "centroid_y",
            ]
        ],
        on=["patch_id", "zone_id_v2"],
        how="inner",
    )
    blast_patch = blast_patch.merge(
        patch_meta[["patch_id", "zone_id_v2", "zone_label_v2"]],
        on=["patch_id", "zone_id_v2", "zone_label_v2"],
        how="inner",
    )

    external = external.sort_values("timestamp").reset_index(drop=True)
    effective_timestamps = set(external["timestamp"].tolist())

    patch_series = patch_series.loc[
        (patch_series["is_global_missing"] == 0) & (patch_series["timestamp"].isin(effective_timestamps))
    ].copy()
    blast_patch = blast_patch.loc[blast_patch["timestamp"].isin(effective_timestamps)].copy()

    patch_series = patch_series.sort_values(["timestamp", "patch_id"]).reset_index(drop=True)
    blast_patch = blast_patch.sort_values(["timestamp", "patch_id"]).reset_index(drop=True)

    return {
        "patch_meta": patch_meta,
        "patch_series": patch_series,
        "blast_patch": blast_patch,
        "external": external,
    }


def build_internal_summary(patch_series: pd.DataFrame, timestamps: pd.DataFrame) -> pd.DataFrame:
    feature_keys = [spec["key"] for spec in INTERNAL_FEATURE_SPECS]
    grouped = patch_series.groupby("timestamp", sort=True)[feature_keys]
    q25 = grouped.quantile(0.25).rename(columns=lambda col: f"{col}_q25")
    median = grouped.median().rename(columns=lambda col: f"{col}_median")
    q75 = grouped.quantile(0.75).rename(columns=lambda col: f"{col}_q75")
    summary = pd.concat([q25, median, q75], axis=1).reset_index()
    return timestamps[["timestamp"]].merge(summary, on="timestamp", how="left")


def add_event_spans(ax: plt.Axes, rain_spans: list[tuple[pd.Timestamp, pd.Timestamp]], blast_spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    for start, end in rain_spans:
        ax.axvspan(start, end, color=RAIN_SPAN_COLOR, alpha=0.10, zorder=0)
    for start, end in blast_spans:
        ax.axvspan(start, end, color=BLAST_SPAN_COLOR, alpha=0.10, zorder=0)


def plot_internal_overview(
    internal_summary: pd.DataFrame,
    external: pd.DataFrame,
    output_root: Path,
) -> Path:
    rain_spans = make_event_spans(external["timestamp"], external["rainfall"] > 0)
    blast_spans = make_event_spans(external["timestamp"], external["blast_count"] > 0)

    fig, axes = plt.subplots(5, 2, figsize=(15, 16), sharex=True, constrained_layout=True)
    axes_flat = axes.flatten()
    timestamps = internal_summary["timestamp"]
    panel_labels = list("abcdefghij")

    for index, spec in enumerate(INTERNAL_FEATURE_SPECS):
        ax = axes_flat[index]
        add_event_spans(ax, rain_spans, blast_spans)
        feature_key = spec["key"]
        ax.fill_between(
            timestamps,
            internal_summary[f"{feature_key}_q25"],
            internal_summary[f"{feature_key}_q75"],
            color=INTERQUARTILE_COLOR,
            alpha=0.55,
            linewidth=0,
        )
        ax.plot(
            timestamps,
            internal_summary[f"{feature_key}_median"],
            color=spec["color"],
            linewidth=1.8,
        )
        ax.set_title(f"({panel_labels[index]}) {spec['label']}", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.35)
        if feature_key.endswith("ratio"):
            ax.set_ylim(-0.03, 1.03)
        format_time_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=INTERNAL_FEATURE_SPECS[0]["color"], linewidth=1.8, label="中位数"),
        Patch(facecolor=INTERQUARTILE_COLOR, alpha=0.55, label="25%~75% 分位带"),
        Patch(facecolor=RAIN_SPAN_COLOR, alpha=0.35, label="降雨时段"),
        Patch(facecolor=BLAST_SPAN_COLOR, alpha=0.35, label="爆破时段"),
    ]
    axes_flat[0].legend(handles=legend_handles, loc="upper left", ncol=2, frameon=True)
    fig.suptitle("边坡内部监测指标时序总览（307 个活动 patch 汇总）", fontsize=15, fontweight="bold")

    png_path = output_root / "fig_paper_internal_monitoring_overview.png"
    pdf_path = output_root / "fig_paper_internal_monitoring_overview.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def draw_series_panel(ax: plt.Axes, timestamps: pd.Series, values: pd.Series, label: str, color: str) -> None:
    ax.plot(timestamps, values, color=color, linewidth=1.6)
    ax.fill_between(timestamps, values, 0, color=color, alpha=0.16)
    ax.set_title(label, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.35)
    format_time_axis(ax)


def plot_external_overview(external: pd.DataFrame, output_root: Path, summary_text: str) -> Path:
    feature_specs = WEATHER_FEATURE_SPECS + BLAST_FEATURE_SPECS
    fig, axes = plt.subplots(6, 2, figsize=(15, 18), sharex=True, constrained_layout=True)
    axes_flat = axes.flatten()
    timestamps = external["timestamp"]
    panel_labels = list("abcdefghijkl")

    for index, spec in enumerate(feature_specs):
        ax = axes_flat[index]
        draw_series_panel(ax, timestamps, external[spec["key"]], f"({panel_labels[index]}) {spec['label']}", spec["color"])

    info_ax = axes_flat[len(feature_specs)]
    info_ax.set_xlim(0.0, 1.0)
    info_ax.set_ylim(0.0, 1.0)
    info_ax.set_xticks([])
    info_ax.set_yticks([])
    for spine in info_ax.spines.values():
        spine.set_visible(False)
    info_ax.text(
        0.02,
        0.98,
        summary_text,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.7,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F8F9FA", "edgecolor": "#D9D9D9"},
        transform=info_ax.transAxes,
    )

    fig.suptitle("天气与爆破因子时序总览", fontsize=15, fontweight="bold")

    png_path = output_root / "fig_paper_weather_blast_overview.png"
    pdf_path = output_root / "fig_paper_weather_blast_overview.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def plot_coupling_heatmap(combined: pd.DataFrame, output_root: Path) -> tuple[Path, pd.DataFrame]:
    internal_cols = [f"{spec['key']}_median" for spec in INTERNAL_FEATURE_SPECS]
    external_cols = [spec["key"] for spec in WEATHER_FEATURE_SPECS + BLAST_FEATURE_SPECS]
    corr_matrix = combined[internal_cols + external_cols].corr(method="spearman").loc[internal_cols, external_cols]

    row_labels = [spec["label"] for spec in INTERNAL_FEATURE_SPECS]
    col_labels = [spec["label"] for spec in WEATHER_FEATURE_SPECS + BLAST_FEATURE_SPECS]

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title("内部监测指标与天气/爆破因子的 Spearman 相关系数", fontweight="bold")
    ax.axvline(len(WEATHER_FEATURE_SPECS) - 0.5, color="#555555", linewidth=1.0)

    for row_idx in range(corr_matrix.shape[0]):
        for col_idx in range(corr_matrix.shape[1]):
            value = float(corr_matrix.iat[row_idx, col_idx])
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value) >= 0.55 else "#222222"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Spearman 相关系数")

    png_path = output_root / "fig_paper_internal_external_coupling_heatmap.png"
    pdf_path = output_root / "fig_paper_internal_external_coupling_heatmap.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    export_matrix = corr_matrix.copy()
    export_matrix.index = row_labels
    export_matrix.columns = col_labels
    export_matrix.to_csv(output_root / "internal_external_spearman_matrix.csv", encoding="utf-8-sig")
    return png_path, export_matrix


def build_spatial_summary(
    patch_meta: pd.DataFrame,
    patch_series: pd.DataFrame,
    blast_patch: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    joined = patch_series[["timestamp", "patch_id", "disp_mean", "active_ratio"]].merge(
        external[["timestamp", "rainfall"]],
        on="timestamp",
        how="left",
    )
    joined = joined.merge(
        blast_patch[["timestamp", "patch_id", "blast_patch_decay_24h"]],
        on=["timestamp", "patch_id"],
        how="left",
    )

    rows: list[dict[str, float | str]] = []
    for patch_id, frame in joined.groupby("patch_id", sort=True):
        rows.append(
            {
                "patch_id": str(patch_id),
                "mean_disp": float(frame["disp_mean"].mean()),
                "mean_active_ratio": float(frame["active_ratio"].mean()),
                "disp_rain_spearman": safe_spearman(frame["disp_mean"], frame["rainfall"]),
                "disp_blast_spearman": safe_spearman(frame["disp_mean"], frame["blast_patch_decay_24h"]),
            }
        )

    summary = pd.DataFrame(rows)
    merged = patch_meta.merge(summary, on="patch_id", how="left")
    return merged


def get_patch_axes_limits(patch_meta: pd.DataFrame, padding: float = 6.0) -> tuple[float, float, float, float]:
    x_min = float(patch_meta["cell_x_min"].min()) - padding
    x_max = float(patch_meta["cell_x_max"].max()) + padding
    y_min = float(patch_meta["cell_y_min"].min()) - padding
    y_max = float(patch_meta["cell_y_max"].max()) + padding
    return x_min, x_max, y_min, y_max


def add_zone_labels(ax: plt.Axes, patch_meta: pd.DataFrame) -> None:
    zone_centers = (
        patch_meta.groupby("zone_label_v2", as_index=False)[["centroid_x", "centroid_y"]]
        .median()
        .sort_values("zone_label_v2")
    )
    for row in zone_centers.itertuples(index=False):
        if not np.isfinite(float(row.centroid_x)) or not np.isfinite(float(row.centroid_y)):
            continue
        ax.text(
            float(row.centroid_x),
            float(row.centroid_y),
            str(row.zone_label_v2),
            ha="center",
            va="center",
            fontsize=8.5,
            color=ZONE_TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )


def plot_patch_map(
    ax: plt.Axes,
    patch_frame: pd.DataFrame,
    value_col: str,
    title: str,
    cmap_name: str,
    norm: colors.Normalize,
) -> None:
    cmap = plt.get_cmap(cmap_name)
    for row in patch_frame.itertuples(index=False):
        value = getattr(row, value_col)
        facecolor = MISSING_FACE_COLOR if pd.isna(value) else cmap(norm(float(value)))
        rect = Rectangle(
            (float(row.cell_x_min), float(row.cell_y_min)),
            float(row.cell_width),
            float(row.cell_height),
            facecolor=facecolor,
            edgecolor="white",
            linewidth=0.28,
        )
        ax.add_patch(rect)

    x_min, x_max, y_min, y_max = get_patch_axes_limits(patch_frame)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("X 网格")
    ax.set_ylabel("Y 网格")
    ax.grid(False)
    add_zone_labels(ax, patch_frame)


def plot_spatial_sensitivity_maps(spatial_summary: pd.DataFrame, output_root: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    axes_flat = axes.flatten()

    mean_disp_norm = colors.Normalize(
        vmin=float(spatial_summary["mean_disp"].quantile(0.05)),
        vmax=float(spatial_summary["mean_disp"].quantile(0.95)),
    )
    mean_active_norm = colors.Normalize(
        vmin=float(spatial_summary["mean_active_ratio"].quantile(0.05)),
        vmax=float(spatial_summary["mean_active_ratio"].quantile(0.95)),
    )
    corr_rain_abs = float(np.nanquantile(np.abs(spatial_summary["disp_rain_spearman"]), 0.95))
    corr_blast_abs = float(np.nanquantile(np.abs(spatial_summary["disp_blast_spearman"]), 0.95))
    corr_rain_norm = colors.TwoSlopeNorm(vmin=-max(corr_rain_abs, 0.20), vcenter=0.0, vmax=max(corr_rain_abs, 0.20))
    corr_blast_norm = colors.TwoSlopeNorm(vmin=-max(corr_blast_abs, 0.20), vcenter=0.0, vmax=max(corr_blast_abs, 0.20))

    panel_specs = [
        {
            "col": "mean_disp",
            "title": "(a) patch 平均位移水平",
            "cmap": "YlOrRd",
            "norm": mean_disp_norm,
            "cbar_label": "位移均值",
        },
        {
            "col": "mean_active_ratio",
            "title": "(b) patch 平均活跃比例",
            "cmap": "YlGnBu",
            "norm": mean_active_norm,
            "cbar_label": "活跃比例",
        },
        {
            "col": "disp_rain_spearman",
            "title": "(c) 位移-降雨相关性",
            "cmap": "RdBu_r",
            "norm": corr_rain_norm,
            "cbar_label": "Spearman 相关系数",
        },
        {
            "col": "disp_blast_spearman",
            "title": "(d) 位移-爆破衰减相关性",
            "cmap": "RdBu_r",
            "norm": corr_blast_norm,
            "cbar_label": "Spearman 相关系数",
        },
    ]

    for ax, spec in zip(axes_flat, panel_specs):
        plot_patch_map(ax, spatial_summary, spec["col"], spec["title"], spec["cmap"], spec["norm"])
        scalar_mappable = plt.cm.ScalarMappable(norm=spec["norm"], cmap=plt.get_cmap(spec["cmap"]))
        colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(spec["cbar_label"])

    fig.suptitle("边坡 patch 空间敏感性图", fontsize=15, fontweight="bold")

    png_path = output_root / "fig_paper_patch_spatial_sensitivity_maps.png"
    pdf_path = output_root / "fig_paper_patch_spatial_sensitivity_maps.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def build_summary_text(external: pd.DataFrame, patch_meta: pd.DataFrame) -> str:
    start_time = pd.Timestamp(external["timestamp"].min()).strftime("%Y-%m-%d %H:%M")
    end_time = pd.Timestamp(external["timestamp"].max()).strftime("%Y-%m-%d %H:%M")
    rain_hours = int((external["rainfall"] > 0).sum())
    blast_hours = int((external["blast_count"] > 0).sum())
    total_charge = float(external["total_charge_kg"].sum())

    lines = [
        "数据集摘要",
        f"时间范围：{start_time} 至 {end_time}",
        f"有效小时数：{len(external)}",
        f"活动 patch 数：{patch_meta['patch_id'].nunique()}",
        f"降雨小时数：{rain_hours}",
        f"爆破小时数：{blast_hours}",
        f"累计装药量：{total_charge:.1f} kg",
        "说明：内部监测图以活动 patch 的中位数与四分位带表示。",
    ]
    return "\n".join(lines)


def write_manifest(
    output_root: Path,
    patch_meta: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    start_time = pd.Timestamp(external["timestamp"].min()).strftime("%Y-%m-%d %H:%M")
    end_time = pd.Timestamp(external["timestamp"].max()).strftime("%Y-%m-%d %H:%M")
    rain_hours = int((external["rainfall"] > 0).sum())
    blast_hours = int((external["blast_count"] > 0).sum())

    lines = [
        "# paper_dataset_figures_v1",
        "",
        "## 数据概况",
        f"- 时间范围：`{start_time}` 至 `{end_time}`",
        f"- 有效时间步：`{len(external)}`",
        f"- 活动 patch 数：`{patch_meta['patch_id'].nunique()}`",
        f"- 降雨小时数：`{rain_hours}`",
        f"- 爆破小时数：`{blast_hours}`",
        "",
        "## 输出图件",
        "- `fig_paper_internal_monitoring_overview.png / .pdf`：10 个内部监测指标的时序总览图。",
        "- `fig_paper_weather_blast_overview.png / .pdf`：4 个天气指标与 7 个爆破指标的时序总览图。",
        "- `fig_paper_internal_external_coupling_heatmap.png / .pdf`：内部监测指标与天气/爆破因子的 Spearman 相关系数热力图。",
        "- `fig_paper_patch_spatial_sensitivity_maps.png / .pdf`：patch 级空间敏感性图，展示位移水平、活跃度以及对降雨/爆破的响应相关性。",
        "",
        "## 伴随表格",
        "- `internal_external_spearman_matrix.csv`：论文热力图对应的相关系数矩阵。",
        "- `patch_spatial_sensitivity_metrics.csv`：patch 空间敏感性指标表。",
        "",
        "## 复现命令",
        f"- `/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python {Path(__file__).resolve()}`",
    ]
    (output_root / "figure_manifest_paper_dataset.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_plot_style()
    paths = resolve_paths(args.output_subdir)
    datasets = load_datasets(paths)

    patch_meta = datasets["patch_meta"]
    patch_series = datasets["patch_series"]
    blast_patch = datasets["blast_patch"]
    external = datasets["external"]
    output_root = paths["output_root"]

    internal_summary = build_internal_summary(patch_series, external)
    combined = internal_summary.merge(external, on="timestamp", how="left")

    summary_text = build_summary_text(external, patch_meta)
    plot_internal_overview(internal_summary, external, output_root)
    plot_external_overview(external, output_root, summary_text)
    plot_coupling_heatmap(combined, output_root)

    spatial_summary = build_spatial_summary(
        patch_meta=patch_meta,
        patch_series=patch_series,
        blast_patch=blast_patch,
        external=external,
    )
    spatial_summary.to_csv(output_root / "patch_spatial_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    plot_spatial_sensitivity_maps(spatial_summary, output_root)
    write_manifest(output_root, patch_meta, external)

    print(output_root)


if __name__ == "__main__":
    main()
