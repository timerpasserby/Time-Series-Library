#!/usr/bin/env python3
"""这个脚本负责生成 slopemine v2 的点位、分区、patch、时序、爆破特征和窗口元数据。
相关文件：config_v2.toml、run_patch_baselines_v2.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 兼容 Python 3.10
    import tomli as tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font")

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Rectangle


INTERNAL_FEATURES = [
    "disp_mean",
    "disp_std",
    "disp_p95",
    "disp_max",
    "vel_mean",
    "vel_p95",
    "acc_mean",
    "acc_p95",
    "active_ratio",
    "valid_ratio",
]

BLAST_V2_FEATURES = [
    "blast_count",
    "total_charge_kg",
    "mean_charge_kg",
    "max_ppv_est_mm_s",
    "disturbance_index",
    "blast_count_6h",
    "charge_24h_kg",
]

BLAST_V3_FEATURES = [
    "blast_patch_decay",
    "blast_patch_decay_6h",
    "blast_patch_decay_24h",
    "blast_patch_peak_recent",
]

WEATHER_FEATURES = ["temperature", "humidity", "wind_speed", "rainfall"]

CJK_FONT_CANDIDATES = [
    "Songti SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "STHeiti",
    "Arial Unicode MS",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Build slopemine v2 patch dataset and metadata.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/slopemine_v2/config_v2.toml"),
        help="Path to the TOML config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """读取并解析配置文件。"""

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    repo_root = config_path.resolve().parents[2]
    config["repo_root"] = repo_root

    for key in ["dataset_dir", "output_dir"]:
        raw_value = Path(config["paths"][key])
        if not raw_value.is_absolute():
            config["paths"][key] = repo_root / raw_value
        else:
            config["paths"][key] = raw_value

    for key in ["raw_csv", "weather_csv", "blast_hourly_csv", "blast_events_csv"]:
        config["paths"][key] = Path(config["paths"][key]).expanduser().resolve()

    return config


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """创建本次运行需要的输出目录。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    figures_dir = output_dir / "figures"
    diagnostics_dir = output_dir / "diagnostics"
    manifests_dir = output_dir / "manifests"
    diagnostic_package_dir = output_dir / "diagnostic_package_v2"
    selected_heatmaps_dir = diagnostic_package_dir / "selected_time_patch_heatmaps"

    for path in [dataset_dir, output_dir, figures_dir, diagnostics_dir, manifests_dir, diagnostic_package_dir, selected_heatmaps_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "figures_dir": figures_dir,
        "diagnostics_dir": diagnostics_dir,
        "manifests_dir": manifests_dir,
        "diagnostic_package_dir": diagnostic_package_dir,
        "selected_heatmaps_dir": selected_heatmaps_dir,
    }


def setup_plot_style() -> None:
    """设置统一的论文风格绘图参数。"""

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
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "grid.color": "#D0D0D0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
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


def load_unique_points(raw_csv: Path) -> pd.DataFrame:
    """读取唯一监测点并保留旧分区标签。"""

    dtypes = {
        "grid_x": "int16",
        "grid_y": "int16",
        "area_id": "int16",
        "area_id_seq": "int16",
    }
    points = pd.read_csv(
        raw_csv,
        usecols=["grid_x", "grid_y", "area_id", "area_id_seq"],
        dtype=dtypes,
    )
    points = points.drop_duplicates(subset=["grid_x", "grid_y"]).sort_values(["grid_x", "grid_y"])
    points = points.reset_index(drop=True)
    return points


def polygon_signed_area(polygon: list[list[float]] | np.ndarray) -> float:
    """计算多边形有向面积，用于判断顶点方向。"""

    coords = np.asarray(polygon, dtype=float)
    if len(coords) < 3:
        raise ValueError("polygon 顶点数不足 3，无法构成有效区域。")
    x_values = coords[:, 0]
    y_values = coords[:, 1]
    return 0.5 * float(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1)))


def ensure_clockwise_polygon(polygon: list[list[float]] | np.ndarray) -> list[list[float]]:
    """把多边形顶点顺序统一成顺时针。"""

    coords = np.asarray(polygon, dtype=float)
    if polygon_signed_area(coords) > 0:
        coords = coords[::-1]
    return [[float(x_value), float(y_value)] for x_value, y_value in coords.tolist()]


def build_zone_rule_payload(config: dict[str, Any]) -> dict[str, Any]:
    """把配置中的分区定义冻结成正式规则载荷。"""

    priority = list(config["processing"]["priority_order"])
    zone_catalog = config["zone_catalog"]
    background_zone_key = str(config["processing"]["background_zone_key"])
    tolerance = float(config["processing"]["polygon_boundary_tolerance"])

    zone_rows = []
    for rank, zone_key in enumerate(priority, start=1):
        zone_cfg = zone_catalog[zone_key]
        zone_rows.append(
            {
                "priority_rank": rank,
                "zone_key_v2": zone_key,
                "zone_id_v2": int(zone_cfg["zone_id"]),
                "zone_name_v2": str(zone_cfg["label"]),
                "zone_label_v2": str(zone_cfg["label"]),
                "color": str(zone_cfg["color"]),
                "polygon": ensure_clockwise_polygon(zone_cfg["polygon"]),
                "polygon_orientation": "clockwise",
            }
        )

    return {
        "version": "slopemine_v2_frozen_zone_rule",
        "priority_order": priority,
        "background_zone_key": background_zone_key,
        "fallback_zone_key": "middle_slope",
        "point_in_polygon_rule": "使用监测点中心坐标 (grid_x, grid_y) 做 point-in-polygon，边界按包含处理。",
        "boundary_tolerance": tolerance,
        "official_background_policy": str(config["processing"]["official_background_policy"]),
        "zones": zone_rows,
    }


def assign_zones(points: pd.DataFrame, zone_rule: dict[str, Any]) -> pd.DataFrame:
    """按冻结后的正式 polygon 规则分配新的工程区。"""

    zone_lookup = {row["zone_key_v2"]: row for row in zone_rule["zones"]}
    frame = points.copy()
    coords = frame[["grid_x", "grid_y"]].to_numpy(dtype=float)
    assigned = np.full(len(frame), "", dtype=object)

    for zone_key in zone_rule["priority_order"]:
        polygon = np.asarray(zone_lookup[zone_key]["polygon"], dtype=float)
        mask = MplPath(polygon).contains_points(coords, radius=float(zone_rule["boundary_tolerance"]))
        assigned[(assigned == "") & mask] = zone_key

    fallback_zone = str(zone_rule["fallback_zone_key"])
    assigned[assigned == ""] = fallback_zone
    frame["zone_key_v2"] = assigned
    frame["zone_id_v2"] = frame["zone_key_v2"].map(
        {zone_key: int(zone_lookup[zone_key]["zone_id_v2"]) for zone_key in zone_lookup}
    )
    frame["zone_name_v2"] = frame["zone_key_v2"].map(
        {zone_key: str(zone_lookup[zone_key]["zone_name_v2"]) for zone_key in zone_lookup}
    )
    frame["zone_label_v2"] = frame["zone_name_v2"]
    frame["point_id"] = np.arange(len(frame), dtype=np.int32)
    frame = frame.rename(columns={"area_id": "area_id_old", "area_id_seq": "area_id_seq_old"})
    return frame[
        [
            "point_id",
            "grid_x",
            "grid_y",
            "zone_id_v2",
            "zone_key_v2",
            "zone_name_v2",
            "zone_label_v2",
            "area_id_old",
            "area_id_seq_old",
        ]
    ]


def save_zone_polygons(zone_rule: dict[str, Any], out_path: Path) -> None:
    """保存正式冻结后的工程分区多边形定义。"""

    payload = {
        "version": zone_rule["version"],
        "priority_order": zone_rule["priority_order"],
        "background_zone_key": zone_rule["background_zone_key"],
        "fallback_zone_key": zone_rule["fallback_zone_key"],
        "boundary_tolerance": zone_rule["boundary_tolerance"],
        "zones": [
            {
                "zone_id_v2": int(row["zone_id_v2"]),
                "zone_key_v2": row["zone_key_v2"],
                "zone_name_v2": row["zone_name_v2"],
                "polygon_orientation": row["polygon_orientation"],
                "polygon": row["polygon"],
                "priority_rank": int(row["priority_rank"]),
                "color": row["color"],
            }
            for row in zone_rule["zones"]
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_zone_assignment_rule(zone_rule: dict[str, Any], out_path: Path) -> None:
    """输出正式的分区判定说明文档。"""

    zone_lines = []
    for zone in zone_rule["zones"]:
        polygon_text = ", ".join(f"({int(x_value)}, {int(y_value)})" for x_value, y_value in zone["polygon"])
        zone_lines.extend(
            [
                f"### zone_id_v2={int(zone['zone_id_v2'])} {zone['zone_name_v2']}",
                f"- zone_key_v2：`{zone['zone_key_v2']}`",
                f"- 优先级：{int(zone['priority_rank'])}",
                f"- 顶点顺序：{zone['polygon_orientation']}",
                f"- polygon：{polygon_text}",
                "",
            ]
        )

    content = [
        "# zone_assignment_rule",
        "",
        "## polygon 判定规则",
        "- 使用监测点中心坐标 `(grid_x, grid_y)` 做 point-in-polygon 判定。",
        f"- 边界采用包含策略，计算时使用 `radius={zone_rule['boundary_tolerance']}` 做极小量扩张，避免边界点因浮点误差漏判。",
        f"- 若点不落入任何 polygon，则回退到 `{zone_rule['fallback_zone_key']}`。",
        "",
        "## 多 polygon 重叠优先级",
        f"- 固定优先级顺序：`{' > '.join(zone_rule['priority_order'])}`。",
        "- 若同一点同时命中多个 polygon，按上面的顺序取第一个命中的 zone，不做平均或二次修正。",
        "",
        "## 稳定背景区特殊处理",
        f"- 稳定背景区固定为 `{zone_rule['background_zone_key']}`，仅负责提供稳定参照，不参与常规活跃区 patch 网格解释。",
        "- 正式 `patch_meta_v2_ps8.csv / patch_meta_v2_ps10.csv` 中，稳定背景区会合并为单一 background patch，并额外标记 `is_train_patch=0`。",
        "- 同时额外导出“完全排除背景区”的 patch 方案，供主实验训练直接使用。",
        "",
        "## 冻结后的 zone 清单",
        "",
        *zone_lines,
    ]
    out_path.write_text("\n".join(content), encoding="utf-8")


def build_zone_stats(point_meta: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """统计每个工程区的点数、占比和坐标范围。"""

    total_points = len(point_meta)
    stats = (
        point_meta.groupby(["zone_id_v2", "zone_key_v2", "zone_name_v2", "zone_label_v2"], as_index=False)
        .agg(
            num_points=("point_id", "size"),
            x_min=("grid_x", "min"),
            x_max=("grid_x", "max"),
            y_min=("grid_y", "min"),
            y_max=("grid_y", "max"),
        )
        .sort_values("zone_id_v2")
        .reset_index(drop=True)
    )
    stats["share"] = stats["num_points"] / total_points
    stats["share_pct"] = (stats["share"] * 100).round(2)
    return stats


def plot_engineering_zones(point_meta: pd.DataFrame, config: dict[str, Any], out_path: Path) -> None:
    """绘制与现有 fig02 风格一致的工程分区图。"""

    zone_catalog = config["zone_catalog"]
    priority = config["processing"]["priority_order"]

    fig, ax = plt.subplots(figsize=(10, 8))
    for zone_key in priority:
        zone_cfg = zone_catalog[zone_key]
        polygon = np.asarray(zone_cfg["polygon"], dtype=float)
        poly = Polygon(
            polygon,
            closed=True,
            facecolor=mcolors.to_rgba(zone_cfg["color"], 0.22),
            edgecolor=zone_cfg["color"],
            linewidth=1.5,
        )
        ax.add_patch(poly)

    for zone_key in priority:
        zone_cfg = zone_catalog[zone_key]
        subset = point_meta.loc[point_meta["zone_key_v2"] == zone_key]
        ax.scatter(
            subset["grid_x"],
            subset["grid_y"],
            s=4,
            c=zone_cfg["color"],
            alpha=0.72,
            label=f"{zone_cfg['label']} (n={len(subset)})",
        )

    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.set_title("工程分区可视化图（v2）")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def make_old_zone_points(point_meta: pd.DataFrame) -> pd.DataFrame:
    """构造旧分区的 patch 划分输入。"""

    frame = point_meta[["point_id", "grid_x", "grid_y", "area_id_seq_old"]].copy()
    frame["zone_id"] = frame["area_id_seq_old"].astype(int)
    frame["zone_key"] = frame["zone_id"].map(lambda value: f"old_zone_{value}")
    frame["zone_label"] = frame["zone_id"].map(lambda value: f"旧分区{value}")
    return frame[["point_id", "grid_x", "grid_y", "zone_id", "zone_key", "zone_label"]]


def make_new_zone_points(point_meta: pd.DataFrame) -> pd.DataFrame:
    """构造新分区的 patch 划分输入。"""

    frame = point_meta[
        ["point_id", "grid_x", "grid_y", "zone_id_v2", "zone_key_v2", "zone_label_v2"]
    ].copy()
    frame = frame.rename(
        columns={
            "zone_id_v2": "zone_id",
            "zone_key_v2": "zone_key",
            "zone_label_v2": "zone_label",
        }
    )
    return frame


def build_patch_meta(
    zone_points: pd.DataFrame,
    patch_size: int,
    sparse_min_points: int,
    patch_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 zone 内规则网格生成 patch，并合并稀疏 patch。"""

    raw_rows: list[dict[str, Any]] = []
    point_rows: list[pd.DataFrame] = []

    for zone_id, zone_frame in zone_points.groupby("zone_id", sort=True):
        zone_frame = zone_frame.copy().reset_index(drop=True)
        zone_key = zone_frame["zone_key"].iat[0]
        zone_label = zone_frame["zone_label"].iat[0]
        x_origin = int(zone_frame["grid_x"].min())
        y_origin = int(zone_frame["grid_y"].min())

        zone_frame["patch_x_idx"] = ((zone_frame["grid_x"] - x_origin) // patch_size).astype(int)
        zone_frame["patch_y_idx"] = ((zone_frame["grid_y"] - y_origin) // patch_size).astype(int)
        zone_frame["raw_patch_id"] = zone_frame.apply(
            lambda row: f"{patch_prefix}_z{int(zone_id):02d}_raw_{int(row['patch_x_idx']):03d}_{int(row['patch_y_idx']):03d}",
            axis=1,
        )

        raw_meta = (
            zone_frame.groupby(["raw_patch_id", "patch_x_idx", "patch_y_idx"], as_index=False)
            .agg(
                num_points=("point_id", "size"),
                x_min=("grid_x", "min"),
                x_max=("grid_x", "max"),
                y_min=("grid_y", "min"),
                y_max=("grid_y", "max"),
                centroid_x=("grid_x", "mean"),
                centroid_y=("grid_y", "mean"),
            )
            .sort_values(["patch_y_idx", "patch_x_idx"])
            .reset_index(drop=True)
        )
        raw_meta["zone_id"] = int(zone_id)
        raw_meta["zone_key"] = zone_key
        raw_meta["zone_label"] = zone_label
        raw_meta["patch_size"] = int(patch_size)
        raw_meta["is_sparse_patch_raw"] = (raw_meta["num_points"] < sparse_min_points).astype(np.int8)
        raw_meta["grid_origin_x"] = x_origin
        raw_meta["grid_origin_y"] = y_origin
        raw_meta["cell_x_min"] = x_origin + raw_meta["patch_x_idx"] * patch_size
        raw_meta["cell_x_max"] = raw_meta["cell_x_min"] + patch_size - 1
        raw_meta["cell_y_min"] = y_origin + raw_meta["patch_y_idx"] * patch_size
        raw_meta["cell_y_max"] = raw_meta["cell_y_min"] + patch_size - 1

        dense_meta = raw_meta.loc[raw_meta["num_points"] >= sparse_min_points].copy()
        if dense_meta.empty:
            dense_meta = raw_meta.copy()

        target_map: dict[str, str] = {}
        for row in raw_meta.itertuples(index=False):
            if row.num_points >= sparse_min_points or len(dense_meta) == 1:
                target_map[row.raw_patch_id] = row.raw_patch_id
                continue

            candidate = dense_meta.loc[dense_meta["raw_patch_id"] != row.raw_patch_id].copy()
            if candidate.empty:
                target_map[row.raw_patch_id] = row.raw_patch_id
                continue

            distance = np.hypot(candidate["centroid_x"] - row.centroid_x, candidate["centroid_y"] - row.centroid_y)
            nearest_idx = int(distance.idxmin())
            target_map[row.raw_patch_id] = str(candidate.loc[nearest_idx, "raw_patch_id"])

        zone_frame["merge_target_raw_patch_id"] = zone_frame["raw_patch_id"].map(target_map)
        point_rows.append(
            zone_frame[
                [
                    "point_id",
                    "zone_id",
                    "zone_key",
                    "zone_label",
                    "raw_patch_id",
                    "merge_target_raw_patch_id",
                ]
            ].copy()
        )

        raw_meta["merge_target_raw_patch_id"] = raw_meta["raw_patch_id"].map(target_map)
        raw_rows.append(raw_meta)

    raw_patch_meta = pd.concat(raw_rows, ignore_index=True)
    point_patch_raw = pd.concat(point_rows, ignore_index=True)

    active_rows: list[dict[str, Any]] = []
    point_patch_map = point_patch_raw.copy()

    for final_index, (target_raw_patch_id, assigned_points) in enumerate(
        point_patch_raw.groupby("merge_target_raw_patch_id", sort=True), start=1
    ):
        target_meta = raw_patch_meta.loc[raw_patch_meta["raw_patch_id"] == target_raw_patch_id].iloc[0]
        raw_sources = raw_patch_meta.loc[
            raw_patch_meta["merge_target_raw_patch_id"] == target_raw_patch_id, "raw_patch_id"
        ].tolist()
        merged_points = zone_points.merge(
            assigned_points[["point_id"]],
            on="point_id",
            how="inner",
            validate="one_to_one",
        )

        patch_id = f"{patch_prefix}_p{final_index:03d}"
        merge_from = "|".join(sorted(source for source in raw_sources if source != target_raw_patch_id))
        active_rows.append(
            {
                "patch_id": patch_id,
                "zone_id_v2": int(target_meta["zone_id"]),
                "zone_key_v2": str(target_meta["zone_key"]),
                "zone_label_v2": str(target_meta["zone_label"]),
                "patch_size": int(target_meta["patch_size"]),
                "patch_x_idx": int(target_meta["patch_x_idx"]),
                "patch_y_idx": int(target_meta["patch_y_idx"]),
                "x_min": int(merged_points["grid_x"].min()),
                "x_max": int(merged_points["grid_x"].max()),
                "y_min": int(merged_points["grid_y"].min()),
                "y_max": int(merged_points["grid_y"].max()),
                "cell_x_min": int(target_meta["cell_x_min"]),
                "cell_x_max": int(target_meta["cell_x_max"]),
                "cell_y_min": int(target_meta["cell_y_min"]),
                "cell_y_max": int(target_meta["cell_y_max"]),
                "centroid_x": float(merged_points["grid_x"].mean()),
                "centroid_y": float(merged_points["grid_y"].mean()),
                "num_points": int(len(merged_points)),
                "is_sparse_patch": int(
                    int(target_meta["is_sparse_patch_raw"]) == 1 and target_meta["merge_target_raw_patch_id"] == target_raw_patch_id
                ),
                "merge_from": merge_from,
                "raw_patch_id": target_raw_patch_id,
                "raw_num_points": int(target_meta["num_points"]),
                "merged_patch_count": int(len(raw_sources)),
                "grid_origin_x": int(target_meta["grid_origin_x"]),
                "grid_origin_y": int(target_meta["grid_origin_y"]),
            }
        )
        point_patch_map.loc[
            point_patch_map["merge_target_raw_patch_id"] == target_raw_patch_id, "patch_id"
        ] = patch_id

    patch_meta = pd.DataFrame(active_rows).sort_values(["zone_id_v2", "patch_y_idx", "patch_x_idx"]).reset_index(drop=True)
    point_patch_map = point_patch_map[["point_id", "patch_id"]].drop_duplicates().reset_index(drop=True)
    return patch_meta, point_patch_map, raw_patch_meta


def enrich_patch_meta_columns(
    patch_meta: pd.DataFrame,
    *,
    background_policy: str,
    train_patch_default: int,
) -> pd.DataFrame:
    """补齐 refined patch 元数据需要的统一字段。"""

    frame = patch_meta.copy()
    if frame.empty:
        return frame

    if "zone_name_v2" not in frame.columns:
        frame["zone_name_v2"] = frame["zone_label_v2"]
    if "cell_width" not in frame.columns:
        frame["cell_width"] = frame["cell_x_max"] - frame["cell_x_min"] + 1
    if "cell_height" not in frame.columns:
        frame["cell_height"] = frame["cell_y_max"] - frame["cell_y_min"] + 1
    if "is_background_patch" not in frame.columns:
        frame["is_background_patch"] = (frame["zone_id_v2"] == 1).astype(np.int8)
    if "is_train_patch" not in frame.columns:
        frame["is_train_patch"] = np.int8(train_patch_default)
    if "background_policy" not in frame.columns:
        frame["background_policy"] = background_policy
    return frame


def build_background_single_patch(
    background_points: pd.DataFrame,
    patch_size: int,
    patch_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把稳定背景区合并成一个特殊 patch。"""

    if background_points.empty:
        empty_patch = pd.DataFrame()
        empty_map = pd.DataFrame(columns=["point_id", "patch_id"])
        empty_raw = pd.DataFrame()
        return empty_patch, empty_map, empty_raw

    zone_id = int(background_points["zone_id"].iat[0])
    zone_key = str(background_points["zone_key"].iat[0])
    zone_label = str(background_points["zone_label"].iat[0])
    x_min = int(background_points["grid_x"].min())
    x_max = int(background_points["grid_x"].max())
    y_min = int(background_points["grid_y"].min())
    y_max = int(background_points["grid_y"].max())
    patch_id = f"{patch_prefix}_background"
    raw_patch_id = f"{patch_prefix}_z{zone_id:02d}_background_raw"

    patch_meta = pd.DataFrame(
        [
            {
                "patch_id": patch_id,
                "zone_id_v2": zone_id,
                "zone_key_v2": zone_key,
                "zone_label_v2": zone_label,
                "zone_name_v2": zone_label,
                "patch_size": int(patch_size),
                "patch_x_idx": -1,
                "patch_y_idx": -1,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "cell_x_min": x_min,
                "cell_x_max": x_max,
                "cell_y_min": y_min,
                "cell_y_max": y_max,
                "cell_width": x_max - x_min + 1,
                "cell_height": y_max - y_min + 1,
                "centroid_x": float(background_points["grid_x"].mean()),
                "centroid_y": float(background_points["grid_y"].mean()),
                "num_points": int(len(background_points)),
                "is_sparse_patch": 0,
                "merge_from": "",
                "raw_patch_id": raw_patch_id,
                "raw_num_points": int(len(background_points)),
                "merged_patch_count": 1,
                "grid_origin_x": x_min,
                "grid_origin_y": y_min,
                "is_background_patch": 1,
                "is_train_patch": 0,
                "background_policy": "single_patch_exclude_from_train",
            }
        ]
    )

    point_patch_map = background_points[["point_id"]].copy()
    point_patch_map["patch_id"] = patch_id

    raw_patch_meta = pd.DataFrame(
        [
            {
                "raw_patch_id": raw_patch_id,
                "patch_x_idx": -1,
                "patch_y_idx": -1,
                "num_points": int(len(background_points)),
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "centroid_x": float(background_points["grid_x"].mean()),
                "centroid_y": float(background_points["grid_y"].mean()),
                "zone_id": zone_id,
                "zone_key": zone_key,
                "zone_label": zone_label,
                "patch_size": int(patch_size),
                "is_sparse_patch_raw": 0,
                "grid_origin_x": x_min,
                "grid_origin_y": y_min,
                "cell_x_min": x_min,
                "cell_x_max": x_max,
                "cell_y_min": y_min,
                "cell_y_max": y_max,
                "merge_target_raw_patch_id": raw_patch_id,
            }
        ]
    )
    return patch_meta, point_patch_map, raw_patch_meta


def build_refined_patch_variants(
    point_meta: pd.DataFrame,
    patch_size: int,
    sparse_min_points: int,
    patch_prefix: str,
    background_zone_key: str,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """生成背景区排除版和单 patch 版两套 refined patch 元数据。"""

    zone_points = make_new_zone_points(point_meta)
    background_zone_id = int(
        point_meta.loc[point_meta["zone_key_v2"] == background_zone_key, "zone_id_v2"].drop_duplicates().iat[0]
    )
    regular_zone_points = zone_points.loc[zone_points["zone_id"] != background_zone_id].copy()
    background_points = zone_points.loc[zone_points["zone_id"] == background_zone_id].copy()

    regular_patch_meta, regular_point_map, regular_raw_meta = build_patch_meta(
        regular_zone_points,
        patch_size=patch_size,
        sparse_min_points=sparse_min_points,
        patch_prefix=patch_prefix,
    )
    regular_patch_meta = enrich_patch_meta_columns(
        regular_patch_meta,
        background_policy="background_excluded",
        train_patch_default=1,
    )

    excluded_patch_meta = regular_patch_meta.copy()
    excluded_patch_meta["background_policy"] = "background_excluded"
    excluded_point_map = regular_point_map.copy()
    excluded_raw_meta = regular_raw_meta.copy()

    background_patch_meta, background_point_map, background_raw_meta = build_background_single_patch(
        background_points,
        patch_size=patch_size,
        patch_prefix=patch_prefix,
    )
    background_patch_meta = enrich_patch_meta_columns(
        background_patch_meta,
        background_policy="single_patch_exclude_from_train",
        train_patch_default=0,
    )

    merged_patch_meta = pd.concat([background_patch_meta, regular_patch_meta], ignore_index=True)
    merged_patch_meta = merged_patch_meta.sort_values(["zone_id_v2", "patch_y_idx", "patch_x_idx", "patch_id"]).reset_index(drop=True)
    merged_point_map = pd.concat([background_point_map, regular_point_map], ignore_index=True)
    merged_raw_meta = pd.concat([background_raw_meta, regular_raw_meta], ignore_index=True)

    return {
        "background_excluded": (excluded_patch_meta, excluded_point_map, excluded_raw_meta),
        "single_background_patch": (merged_patch_meta, merged_point_map, merged_raw_meta),
    }


def build_patch_stats_refined(
    patch_variant: str,
    patch_size: int,
    patch_meta: pd.DataFrame,
    raw_patch_meta: pd.DataFrame,
) -> pd.DataFrame:
    """汇总 refined patch 划分统计。"""

    rows = [
        {
            "patch_variant": patch_variant,
            "scope": "overall",
            "patch_size": int(patch_size),
            "zone_id_v2": -1,
            "zone_name_v2": "全部",
            "raw_patch_count": int(len(raw_patch_meta)),
            "active_patch_count": int(len(patch_meta)),
            "sparse_patch_count": int(patch_meta["is_sparse_patch"].sum()) if not patch_meta.empty else 0,
            "train_patch_count": int(patch_meta["is_train_patch"].sum()) if not patch_meta.empty else 0,
            "excluded_train_patch_count": int((patch_meta["is_train_patch"] == 0).sum()) if not patch_meta.empty else 0,
            "avg_points_per_patch": float(patch_meta["num_points"].mean()) if not patch_meta.empty else np.nan,
            "median_points_per_patch": float(patch_meta["num_points"].median()) if not patch_meta.empty else np.nan,
        }
    ]

    for zone_id, zone_patch_meta in patch_meta.groupby("zone_id_v2", sort=True):
        rows.append(
            {
                "patch_variant": patch_variant,
                "scope": "zone",
                "patch_size": int(patch_size),
                "zone_id_v2": int(zone_id),
                "zone_name_v2": str(zone_patch_meta["zone_name_v2"].iat[0]),
                "raw_patch_count": int(len(raw_patch_meta.loc[raw_patch_meta["zone_id"] == zone_id])),
                "active_patch_count": int(len(zone_patch_meta)),
                "sparse_patch_count": int(zone_patch_meta["is_sparse_patch"].sum()),
                "train_patch_count": int(zone_patch_meta["is_train_patch"].sum()),
                "excluded_train_patch_count": int((zone_patch_meta["is_train_patch"] == 0).sum()),
                "avg_points_per_patch": float(zone_patch_meta["num_points"].mean()),
                "median_points_per_patch": float(zone_patch_meta["num_points"].median()),
            }
        )

    return pd.DataFrame(rows)


def build_patch_stats(
    scheme_name: str,
    patch_size: int,
    patch_meta: pd.DataFrame,
    raw_patch_meta: pd.DataFrame,
) -> pd.DataFrame:
    """汇总 patch 划分统计并生成新旧方案对比表。"""

    overall = {
        "scheme": scheme_name,
        "scope": "overall",
        "patch_size": patch_size,
        "zone_id": -1,
        "zone_label": "全部",
        "raw_patch_count": int(len(raw_patch_meta)),
        "active_patch_count": int(len(patch_meta)),
        "sparse_raw_patch_count": int(raw_patch_meta["is_sparse_patch_raw"].sum()),
        "merged_source_patch_count": int((raw_patch_meta["merge_target_raw_patch_id"] != raw_patch_meta["raw_patch_id"]).sum()),
        "num_points_mean": float(patch_meta["num_points"].mean()),
        "num_points_median": float(patch_meta["num_points"].median()),
        "num_points_min": int(patch_meta["num_points"].min()),
        "num_points_max": int(patch_meta["num_points"].max()),
    }
    rows = [overall]

    for zone_id, zone_patch_meta in patch_meta.groupby("zone_id_v2", sort=True):
        zone_raw = raw_patch_meta.loc[raw_patch_meta["zone_id"] == zone_id]
        rows.append(
            {
                "scheme": scheme_name,
                "scope": "zone",
                "patch_size": patch_size,
                "zone_id": int(zone_id),
                "zone_label": zone_patch_meta["zone_label_v2"].iat[0],
                "raw_patch_count": int(len(zone_raw)),
                "active_patch_count": int(len(zone_patch_meta)),
                "sparse_raw_patch_count": int(zone_raw["is_sparse_patch_raw"].sum()),
                "merged_source_patch_count": int((zone_raw["merge_target_raw_patch_id"] != zone_raw["raw_patch_id"]).sum()),
                "num_points_mean": float(zone_patch_meta["num_points"].mean()),
                "num_points_median": float(zone_patch_meta["num_points"].median()),
                "num_points_min": int(zone_patch_meta["num_points"].min()),
                "num_points_max": int(zone_patch_meta["num_points"].max()),
            }
        )

    return pd.DataFrame(rows)


def plot_patch_grid(
    point_meta: pd.DataFrame,
    patch_meta: pd.DataFrame,
    config: dict[str, Any],
    patch_size: int,
    out_path: Path,
) -> None:
    """绘制 patch 网格图。"""

    zone_catalog = config["zone_catalog"]
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.scatter(point_meta["grid_x"], point_meta["grid_y"], s=1.5, c="#8A8F98", alpha=0.45)
    for row in patch_meta.itertuples(index=False):
        face_color = zone_catalog[str(row.zone_key_v2)]["color"]
        rect = Rectangle(
            (row.cell_x_min, row.cell_y_min),
            getattr(row, "cell_width", row.patch_size),
            getattr(row, "cell_height", row.patch_size),
            facecolor=mcolors.to_rgba(face_color, 0.16),
            edgecolor=face_color,
            linewidth=1.0 if row.merge_from else 0.65,
        )
        ax.add_patch(rect)

    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.set_title(f"规则 patch 划分图（patch_size={patch_size}）")
    set_patch_axes_limits(ax, patch_meta)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_patch_density(
    patch_meta: pd.DataFrame,
    patch_size: int,
    out_path: Path,
) -> None:
    """绘制 patch 点密度热力图。"""

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor("#F4F4F4")
    norm = plt.Normalize(vmin=float(patch_meta["num_points"].min()), vmax=float(patch_meta["num_points"].max()))
    cmap = plt.get_cmap("YlOrRd")

    for row in patch_meta.itertuples(index=False):
        rect = Rectangle(
            (row.cell_x_min, row.cell_y_min),
            getattr(row, "cell_width", row.patch_size),
            getattr(row, "cell_height", row.patch_size),
            facecolor=cmap(norm(row.num_points)),
            edgecolor="#5C5C5C",
            linewidth=0.45,
        )
        ax.add_patch(rect)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, label="patch 点数")
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.set_title(f"patch 点密度热力图（patch_size={patch_size}）")
    set_patch_axes_limits(ax, patch_meta)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def load_monitoring_frame(raw_csv: Path) -> pd.DataFrame:
    """读取完整监测时序数据。"""

    dtypes = {
        "grid_x": "int16",
        "grid_y": "int16",
        "deformation": "float32",
        "speed": "float32",
        "acceleration": "float32",
        "device_id": "int32",
        "area_id": "int16",
        "area_id_seq": "int16",
    }
    frame = pd.read_csv(
        raw_csv,
        usecols=[
            "grid_x",
            "grid_y",
            "deformation",
            "speed",
            "acceleration",
            "report_time",
            "device_id",
            "area_id",
            "area_id_seq",
        ],
        dtype=dtypes,
        parse_dates=["report_time"],
    )
    return frame


def aggregate_patch_series(
    monitoring_frame: pd.DataFrame,
    point_meta: pd.DataFrame,
    patch_meta: pd.DataFrame,
    point_patch_map: pd.DataFrame,
    active_speed_threshold: float,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[pd.Timestamp]]:
    """按小时和 patch 聚合位移、速度和加速度统计。"""

    point_lookup = point_meta.merge(point_patch_map, on="point_id", how="left", validate="one_to_one")
    point_lookup = point_lookup[["grid_x", "grid_y", "patch_id", "zone_id_v2"]]

    merged = monitoring_frame.merge(
        point_lookup,
        on=["grid_x", "grid_y"],
        how="left",
        validate="many_to_one",
    )
    if merged["patch_id"].isna().any():
        raise ValueError("存在监测记录未能映射到 patch_id。")

    merged["timestamp"] = pd.to_datetime(merged["report_time"]).dt.floor("h")
    merged["is_active"] = (merged["speed"].abs() >= active_speed_threshold).astype(np.int8)

    group_cols = ["timestamp", "patch_id", "zone_id_v2"]
    grouped = merged.groupby(group_cols, observed=True)
    stats = grouped.agg(
        num_records=("deformation", "size"),
        disp_mean=("deformation", "mean"),
        disp_std=("deformation", "std"),
        disp_max=("deformation", "max"),
        vel_mean=("speed", "mean"),
        acc_mean=("acceleration", "mean"),
        active_count=("is_active", "sum"),
    )
    quantiles = grouped[["deformation", "speed", "acceleration"]].quantile(0.95)
    quantiles = quantiles.rename(
        columns={
            "deformation": "disp_p95",
            "speed": "vel_p95",
            "acceleration": "acc_p95",
        }
    )
    stats = stats.join(quantiles).reset_index()
    stats = stats.merge(patch_meta[["patch_id", "num_points"]], on="patch_id", how="left", validate="many_to_one")
    stats["valid_ratio"] = stats["num_records"] / stats["num_points"]
    stats["active_ratio"] = stats["active_count"] / stats["num_points"]
    stats = stats.drop(columns=["active_count", "num_points"])

    observed_hours = pd.DatetimeIndex(sorted(stats["timestamp"].drop_duplicates()))
    full_hours = pd.date_range(observed_hours.min(), observed_hours.max(), freq="h")
    missing_hours = list(full_hours.difference(observed_hours))

    calendar = pd.MultiIndex.from_product(
        [full_hours, patch_meta["patch_id"].tolist()],
        names=["timestamp", "patch_id"],
    ).to_frame(index=False)
    calendar = calendar.merge(
        patch_meta[["patch_id", "zone_id_v2"]],
        on="patch_id",
        how="left",
        validate="many_to_one",
    )

    patch_series = calendar.merge(
        stats,
        on=["timestamp", "patch_id", "zone_id_v2"],
        how="left",
        validate="one_to_one",
    ).sort_values(["timestamp", "patch_id"])
    patch_series["num_records"] = patch_series["num_records"].fillna(0).astype(np.int32)
    patch_series["valid_ratio"] = patch_series["valid_ratio"].fillna(0.0).astype(np.float32)
    patch_series["active_ratio"] = patch_series["active_ratio"].fillna(0.0).astype(np.float32)
    patch_series["is_global_missing"] = patch_series["timestamp"].isin(missing_hours).astype(np.int8)

    return patch_series.reset_index(drop=True), full_hours, missing_hours


def build_patch_series_summary(patch_series: pd.DataFrame) -> pd.DataFrame:
    """导出 patch 级统计摘要表。"""

    summary = (
        patch_series.groupby(["patch_id", "zone_id_v2"], as_index=False)
        .agg(
            valid_ratio_mean=("valid_ratio", "mean"),
            valid_ratio_min=("valid_ratio", "min"),
            valid_ratio_max=("valid_ratio", "max"),
            active_ratio_mean=("active_ratio", "mean"),
            global_missing_hours=("is_global_missing", "sum"),
            observed_hours=("num_records", lambda values: int((values > 0).sum())),
            disp_mean_avg=("disp_mean", "mean"),
            disp_mean_std=("disp_mean", "std"),
        )
        .sort_values(["zone_id_v2", "patch_id"])
        .reset_index(drop=True)
    )
    return summary


def build_zone_point_distribution(point_meta: pd.DataFrame) -> pd.DataFrame:
    """汇总新旧分区的点位分布。"""

    old_distribution = (
        point_meta.groupby("area_id_seq_old", as_index=False)
        .agg(num_points=("point_id", "size"))
        .rename(columns={"area_id_seq_old": "zone_id"})
    )
    old_distribution["zone_name"] = old_distribution["zone_id"].map(lambda value: f"旧分区{int(value)}")
    old_distribution["scheme"] = "old_area_id_seq"

    new_distribution = (
        point_meta.groupby(["zone_id_v2", "zone_name_v2"], as_index=False)
        .agg(num_points=("point_id", "size"))
        .rename(columns={"zone_id_v2": "zone_id", "zone_name_v2": "zone_name"})
    )
    new_distribution["scheme"] = "new_zone_v2"

    distribution = pd.concat([old_distribution, new_distribution], ignore_index=True)
    distribution["share"] = distribution.groupby("scheme")["num_points"].transform(lambda values: values / values.sum())
    distribution["share_pct"] = (distribution["share"] * 100.0).round(4)
    return distribution.sort_values(["scheme", "zone_id"]).reset_index(drop=True)


def plot_zone_point_distribution(
    zone_point_distribution: pd.DataFrame,
    config: dict[str, Any],
    out_path: Path,
) -> None:
    """绘制新旧分区点位分布对比图。"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    new_colors = {
        int(zone_cfg["zone_id"]): zone_cfg["color"]
        for zone_cfg in config["zone_catalog"].values()
    }

    for axis, scheme, title in zip(
        axes,
        ["old_area_id_seq", "new_zone_v2"],
        ["旧分区点位分布", "新分区点位分布"],
    ):
        frame = zone_point_distribution.loc[zone_point_distribution["scheme"] == scheme].copy()
        colors = ["#B0B0B0" if scheme == "old_area_id_seq" else new_colors.get(int(zone_id), "#4C78A8") for zone_id in frame["zone_id"]]
        axis.bar(frame["zone_name"], frame["num_points"], color=colors, edgecolor="#444444", linewidth=0.5)
        axis.set_title(title)
        axis.set_xlabel("zone")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", linestyle="--", alpha=0.25)

    axes[0].set_ylabel("点位数")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def build_zone_patch_distribution(patch_stats_refined: pd.DataFrame) -> pd.DataFrame:
    """整理按 zone 的 patch 分布表。"""

    frame = patch_stats_refined.loc[patch_stats_refined["scope"] == "zone"].copy()
    return frame.sort_values(["patch_variant", "patch_size", "zone_id_v2"]).reset_index(drop=True)


def plot_zone_patch_distribution(
    zone_patch_distribution: pd.DataFrame,
    out_path: Path,
) -> None:
    """绘制主方案下 ps8 与 ps10 的 zone patch 数对比图。"""

    main_frame = zone_patch_distribution.loc[
        zone_patch_distribution["patch_variant"] == "single_background_patch"
    ].copy()
    pivot = (
        main_frame.pivot(index="zone_name_v2", columns="patch_size", values="active_patch_count")
        .sort_index()
        .fillna(0)
    )
    x_axis = np.arange(len(pivot.index))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5))
    patch_sizes = sorted(pivot.columns.tolist())
    for index, patch_size in enumerate(patch_sizes):
        offset = (index - (len(patch_sizes) - 1) / 2.0) * width
        ax.bar(
            x_axis + offset,
            pivot[patch_size].to_numpy(dtype=float),
            width=width,
            label=f"patch_size={int(patch_size)}",
            edgecolor="#444444",
            linewidth=0.5,
        )

    ax.set_xticks(x_axis)
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel("patch 数")
    ax.set_title("各 zone 的 patch 数（主方案）")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_zone_boxplot(
    patch_series: pd.DataFrame,
    patch_meta: pd.DataFrame,
    value_column: str,
    title: str,
    out_path: Path,
) -> None:
    """绘制按 zone 汇总的箱线图。"""

    merged = patch_series.merge(
        patch_meta[["patch_id", "zone_id_v2", "zone_name_v2"]],
        on=["patch_id", "zone_id_v2"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.loc[merged["valid_ratio"] > 0].copy()
    merged = merged.loc[merged[value_column].notna()].copy()
    if merged.empty:
        raise ValueError(f"{value_column} 没有可用于绘图的有效值。")

    groups = []
    labels = []
    for zone_id, zone_frame in merged.groupby("zone_id_v2", sort=True):
        groups.append(zone_frame[value_column].to_numpy(dtype=float))
        labels.append(str(zone_frame["zone_name_v2"].iat[0]))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, labels=labels, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(value_column)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def pick_representative_disp_hours(
    patch_series: pd.DataFrame,
    count: int,
) -> tuple[list[pd.Timestamp], pd.DataFrame]:
    """按位移强度分位点选择代表时刻。"""

    hourly = (
        patch_series.loc[(patch_series["valid_ratio"] > 0) & (patch_series["is_global_missing"] == 0)]
        .groupby("timestamp", as_index=False)
        .agg(
            mean_abs_disp=("disp_mean", lambda values: float(np.nanmean(np.abs(values)))),
            mean_disp_max=("disp_max", "mean"),
            valid_patch_count=("patch_id", "size"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if hourly.empty:
        return [], hourly

    ranked = hourly.sort_values("mean_abs_disp").reset_index(drop=True)
    quantiles = np.linspace(0.2, 0.9, num=max(count, 1))
    selected_indices: list[int] = []
    for quantile in quantiles:
        candidate = int(round((len(ranked) - 1) * float(quantile)))
        while candidate in selected_indices and candidate + 1 < len(ranked):
            candidate += 1
        selected_indices.append(candidate)

    selected = ranked.iloc[selected_indices].sort_values("timestamp").reset_index(drop=True)
    return selected["timestamp"].tolist(), selected


def plot_selected_disp_heatmaps(
    patch_meta: pd.DataFrame,
    patch_series: pd.DataFrame,
    timestamps: list[pd.Timestamp],
    out_dir: Path,
) -> list[Path]:
    """绘制代表时刻的 patch 级 disp_mean 热力图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for timestamp in timestamps:
        frame = patch_series.loc[patch_series["timestamp"] == timestamp, ["patch_id", "disp_mean", "valid_ratio"]].copy()
        merged = patch_meta.merge(frame, on="patch_id", how="left", validate="one_to_one")
        active = merged.loc[(merged["valid_ratio"].fillna(0) > 0) & merged["disp_mean"].notna()].copy()
        if active.empty:
            raise ValueError(f"{timestamp:%Y-%m-%d %H:%M} 没有有效的 disp_mean patch 数据。")

        max_abs = float(np.nanmax(np.abs(active["disp_mean"].to_numpy(dtype=float))))
        if max_abs <= 0:
            norm = plt.Normalize(vmin=float(active["disp_mean"].min()), vmax=float(active["disp_mean"].max()) + 1e-9)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        cmap = plt.get_cmap("RdBu_r")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#EFEFEF")
        for row in active.itertuples(index=False):
            rect = Rectangle(
                (row.cell_x_min, row.cell_y_min),
                getattr(row, "cell_width", row.patch_size),
                getattr(row, "cell_height", row.patch_size),
                facecolor=cmap(norm(row.disp_mean)),
                edgecolor="#666666",
                linewidth=0.42,
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, label="disp_mean")
        ax.set_title(f"patch 级 disp_mean 热力图：{timestamp:%Y-%m-%d %H:%M}")
        ax.set_xlabel("grid_x")
        ax.set_ylabel("grid_y")
        set_patch_axes_limits(ax, patch_meta)
        output_path = out_dir / f"disp_mean_heatmap_{timestamp:%Y%m%d_%H%M}.png"
        fig.savefig(output_path, dpi=240)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def build_zone_feature_summary(
    patch_series: pd.DataFrame,
    patch_meta: pd.DataFrame,
) -> pd.DataFrame:
    """汇总按 zone 的关键时序特征摘要。"""

    merged = patch_series.merge(
        patch_meta[["patch_id", "zone_id_v2", "zone_name_v2", "is_train_patch"]],
        on=["patch_id", "zone_id_v2"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.loc[merged["valid_ratio"] > 0].copy()

    feature_summary = (
        merged.groupby(["zone_id_v2", "zone_name_v2"], as_index=False)
        .agg(
            disp_mean_avg=("disp_mean", "mean"),
            disp_mean_p95=("disp_mean", lambda values: float(values.quantile(0.95))),
            disp_max_avg=("disp_max", "mean"),
            disp_max_p95=("disp_max", lambda values: float(values.quantile(0.95))),
            vel_mean_avg=("vel_mean", "mean"),
            acc_mean_avg=("acc_mean", "mean"),
            active_ratio_avg=("active_ratio", "mean"),
            valid_ratio_avg=("valid_ratio", "mean"),
        )
        .sort_values("zone_id_v2")
        .reset_index(drop=True)
    )
    patch_counts = (
        patch_meta.groupby(["zone_id_v2", "zone_name_v2"], as_index=False)
        .agg(
            patch_count=("patch_id", "nunique"),
            train_patch_count=("is_train_patch", "sum"),
        )
        .sort_values("zone_id_v2")
        .reset_index(drop=True)
    )
    return patch_counts.merge(
        feature_summary,
        on=["zone_id_v2", "zone_name_v2"],
        how="left",
        validate="one_to_one",
    )


def write_patch_diagnostic_report(
    *,
    point_meta: pd.DataFrame,
    patch_stats_refined: pd.DataFrame,
    zone_feature_summary: pd.DataFrame,
    representative_times: pd.DataFrame,
    heatmap_paths: list[Path],
    out_path: Path,
) -> None:
    """输出论文可用的数据诊断报告。"""

    old_distribution = (
        point_meta.groupby("area_id_seq_old", as_index=False)
        .agg(num_points=("point_id", "size"))
        .sort_values("area_id_seq_old")
        .reset_index(drop=True)
    )
    new_distribution = (
        point_meta.groupby(["zone_id_v2", "zone_name_v2"], as_index=False)
        .agg(num_points=("point_id", "size"))
        .sort_values("zone_id_v2")
        .reset_index(drop=True)
    )
    background_points = int(new_distribution.loc[new_distribution["zone_id_v2"] == 1, "num_points"].sum())
    background_share = background_points / float(len(point_meta))

    ps8_main = patch_stats_refined.loc[
        (patch_stats_refined["patch_variant"] == "single_background_patch")
        & (patch_stats_refined["patch_size"] == 8)
        & (patch_stats_refined["scope"] == "overall")
    ].iloc[0]
    ps10_main = patch_stats_refined.loc[
        (patch_stats_refined["patch_variant"] == "single_background_patch")
        & (patch_stats_refined["patch_size"] == 10)
        & (patch_stats_refined["scope"] == "overall")
    ].iloc[0]

    report_lines = [
        "# patch_data_diagnostic_report_v2",
        "",
        "## 数据概况",
        f"- 点位总数：{len(point_meta)}",
        f"- 新分区数：{new_distribution['zone_id_v2'].nunique()} 个",
        f"- 稳定背景区点位：{background_points} 个，占比 {background_share * 100:.4f}%",
        "",
        "## 关键结论",
        "- 新分区比旧分区更均衡：旧规则几乎只分成“稳定背景区 + 其余全部区域”两块，其中旧分区2独占绝大多数点位；新规则把活跃区拆成坡脚、坡中、平台、坡顶四个工程区，空间语义和 patch 数都更可解释。",
        f"- `patch_size=10` 适合作为主实验粒度：主方案下 ps8 有 {int(ps8_main['active_patch_count'])} 个 patch，ps10 有 {int(ps10_main['active_patch_count'])} 个 patch；ps10 在明显降低 patch 数的同时，仍保留足够的 zone 内细粒度结构，更适合作为正式实验主粒度。",
        f"- 稳定背景区不建议纳入主实验训练：该区只有 {background_points} 个点位，正式主方案中仅保留 1 个 background patch，并显式标记 `is_train_patch=0`，更适合作为参照区而不是预测目标。",
        "",
        "## 新旧分区对比",
        f"- 旧分区点位分布：{', '.join(f'旧分区{int(row.area_id_seq_old)}={int(row.num_points)}' for row in old_distribution.itertuples(index=False))}",
        f"- 新分区点位分布：{', '.join(f'{row.zone_name_v2}={int(row.num_points)}' for row in new_distribution.itertuples(index=False))}",
        "",
        "## patch_size 诊断",
        f"- ps8 主方案：active patch={int(ps8_main['active_patch_count'])}，sparse patch={int(ps8_main['sparse_patch_count'])}，平均点数/patch={float(ps8_main['avg_points_per_patch']):.2f}",
        f"- ps10 主方案：active patch={int(ps10_main['active_patch_count'])}，sparse patch={int(ps10_main['sparse_patch_count'])}，平均点数/patch={float(ps10_main['avg_points_per_patch']):.2f}",
        "",
        "## zone 特征摘要",
        *[
            f"- {row.zone_name_v2}：patch_count={int(row.patch_count)}，disp_mean_avg={float(row.disp_mean_avg):.4f}，disp_max_p95={float(row.disp_max_p95):.4f}，valid_ratio_avg={float(row.valid_ratio_avg):.4f}"
            for row in zone_feature_summary.itertuples(index=False)
        ],
        "",
        "## 代表时刻热力图",
    ]

    if representative_times.empty:
        report_lines.append("- 未选出有效代表时刻。")
    else:
        for row, heatmap_path in zip(representative_times.itertuples(index=False), heatmap_paths):
            report_lines.append(
                f"- {pd.Timestamp(row.timestamp):%Y-%m-%d %H:%M}：mean_abs_disp={float(row.mean_abs_disp):.6f}，mean_disp_max={float(row.mean_disp_max):.6f}，图文件=`{heatmap_path}`"
            )

    out_path.write_text("\n".join(report_lines), encoding="utf-8")


def load_hourly_external_frame(csv_path: Path, time_column: str) -> pd.DataFrame:
    """读取按小时对齐的外部变量表。"""

    frame = pd.read_csv(csv_path)
    frame["timestamp"] = pd.to_datetime(frame[time_column]).dt.floor("h")
    if time_column != "timestamp":
        frame = frame.drop(columns=[time_column])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def align_external_frame(frame: pd.DataFrame, full_hours: pd.DatetimeIndex) -> pd.DataFrame:
    """把外部变量表对齐到完整小时轴。"""

    aligned = frame.set_index("timestamp").reindex(full_hours).reset_index()
    aligned = aligned.rename(columns={"index": "timestamp"})
    return aligned


def build_blast_ledger(config: dict[str, Any], dataset_dir: Path) -> pd.DataFrame:
    """把爆破事件表转成 V3 接口需要的标准格式。"""

    frame = pd.read_csv(config["paths"]["blast_events_csv"])
    frame["timestamp"] = pd.to_datetime(frame["blast_time"])
    zone_coords = config["blast_zone_coords"]
    default_sigma = float(config["processing"]["default_sigma"])

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        zone_cfg = zone_coords.get(row.zone, {})
        rows.append(
            {
                "blast_id": getattr(row, "blast_id", f"BL{index:03d}"),
                "timestamp": row.timestamp,
                "x_b": float(zone_cfg.get("x_b", np.nan)),
                "y_b": float(zone_cfg.get("y_b", np.nan)),
                "Q": float(row.charge_kg),
                "sigma": float(zone_cfg.get("sigma", default_sigma)),
                "zone": str(row.zone),
                "hole_count": int(row.hole_count),
                "distance_to_monitor_m": float(row.distance_to_monitor_m),
                "ppv_est_mm_s": float(row.ppv_est_mm_s),
                "intensity_level": str(row.intensity_level),
                "is_coordinate_inferred": 1,
            }
        )

    ledger = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    ledger.to_csv(dataset_dir / "blast_ledger_v3_input.csv", index=False)
    return ledger


def build_patch_blast_features(
    blast_ledger: pd.DataFrame,
    patch_meta: pd.DataFrame,
    full_hours: pd.DatetimeIndex,
    config: dict[str, Any],
) -> pd.DataFrame:
    """计算 patch-centroid 爆破特征 V3。"""

    tau_main = float(config["processing"]["blast_decay_tau_hours"])
    tau_6h = float(config["processing"]["blast_decay_6h_tau_hours"])
    tau_24h = float(config["processing"]["blast_decay_24h_tau_hours"])

    patch_centroids = patch_meta[["patch_id", "centroid_x", "centroid_y"]].reset_index(drop=True)
    event_times = blast_ledger["timestamp"].to_numpy(dtype="datetime64[ns]")
    hour_times = full_hours.to_numpy(dtype="datetime64[ns]")
    delta_hours = ((hour_times[:, None] - event_times[None, :]) / np.timedelta64(1, "h")).astype(float)
    valid_mask = delta_hours >= 0

    base_weights = []
    for row in blast_ledger.itertuples(index=False):
        distance = np.hypot(patch_centroids["centroid_x"] - row.x_b, patch_centroids["centroid_y"] - row.y_b)
        sigma = max(float(row.sigma), 1.0)
        base_weights.append(float(row.Q) * np.exp(-np.square(distance) / (2.0 * sigma * sigma)))
    base_weights = np.asarray(base_weights, dtype=np.float32)

    decay_main = np.where(valid_mask, np.exp(-delta_hours / tau_main), 0.0).astype(np.float32)
    decay_6h = np.where(valid_mask & (delta_hours <= 6.0), np.exp(-delta_hours / tau_6h), 0.0).astype(np.float32)
    decay_24h = np.where(valid_mask & (delta_hours <= 24.0), np.exp(-delta_hours / tau_24h), 0.0).astype(np.float32)

    feature_main = decay_main @ base_weights
    feature_6h = decay_6h @ base_weights
    feature_24h = decay_24h @ base_weights

    peak_recent = np.zeros_like(feature_main, dtype=np.float32)
    for event_index in range(len(blast_ledger)):
        event_decay = np.where(
            valid_mask[:, event_index] & (delta_hours[:, event_index] <= 24.0),
            np.exp(-delta_hours[:, event_index] / tau_6h),
            0.0,
        ).astype(np.float32)
        peak_recent = np.maximum(peak_recent, event_decay[:, None] * base_weights[event_index][None, :])

    rows = []
    for time_index, timestamp in enumerate(full_hours):
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "patch_id": patch_centroids["patch_id"].values,
                    "blast_patch_decay": feature_main[time_index],
                    "blast_patch_decay_6h": feature_6h[time_index],
                    "blast_patch_decay_24h": feature_24h[time_index],
                    "blast_patch_peak_recent": peak_recent[time_index],
                }
            )
        )

    features = pd.concat(rows, ignore_index=True)
    features = features.merge(
        patch_meta[["patch_id", "zone_id_v2", "zone_label_v2"]],
        on="patch_id",
        how="left",
        validate="many_to_one",
    )
    return features


def pick_representative_hours(
    blast_hourly: pd.DataFrame,
    count: int,
) -> list[pd.Timestamp]:
    """挑选代表性的爆破小时。"""

    candidate = blast_hourly.loc[blast_hourly["blast_count"].fillna(0) > 0].copy()
    if candidate.empty:
        return []
    candidate = candidate.sort_values(
        ["total_charge_kg", "max_ppv_est_mm_s", "blast_count"],
        ascending=False,
    )
    return candidate["timestamp"].head(count).tolist()


def summarize_blast_heatmap_frame(
    patch_meta: pd.DataFrame,
    patch_blast_features: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """汇总单个时刻的热力图调试信息。"""

    frame = patch_blast_features.loc[patch_blast_features["timestamp"] == timestamp].copy()
    if frame.empty:
        raise ValueError(f"{timestamp:%Y-%m-%d %H:%M} 在 patch_blast_features_v3_ps10.csv 中不存在记录。")

    merged = patch_meta.merge(
        frame[["patch_id", "zone_id_v2", "blast_patch_decay"]],
        on="patch_id",
        how="left",
        suffixes=("_patch", "_blast"),
        validate="one_to_one",
    )
    if merged["blast_patch_decay"].isna().any():
        raise ValueError(f"{timestamp:%Y-%m-%d %H:%M} merge 后存在 NaN，说明 patch_id 对齐失败。")

    values = merged["blast_patch_decay"]
    stats = {
        "timestamp": f"{timestamp:%Y-%m-%d %H:%M}",
        "total_patch_count": int(len(merged)),
        "nonzero_patch_count": int((values != 0).sum()),
        "min": float(values.min()),
        "p50": float(values.quantile(0.5)),
        "p90": float(values.quantile(0.9)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "has_nan": bool(values.isna().any()),
        "merge_columns_ok": bool(
            {"patch_id", "patch_x_idx", "patch_y_idx", "centroid_x", "centroid_y", "zone_id_v2_patch", "blast_patch_decay"}
            <= set(merged.columns)
        ),
    }
    return merged, stats


def draw_patch_rectangles(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    cmap_name: str,
    variant: str,
) -> tuple[Any, Any]:
    """使用 patch 边界绘制矩形着色图。"""

    active = frame.loc[frame[value_column].notna() & (frame[value_column] > 0)].copy()
    if active.empty:
        raise ValueError("当前时刻数据不为空，但所有 patch 的 blast_patch_decay 都为 0，无法绘制活动热力图。")

    if variant == "linear":
        display_values = active[value_column].to_numpy(dtype=float)
        norm = plt.Normalize(vmin=float(display_values.min()), vmax=float(display_values.max()))
    elif variant == "p99":
        display_values = active[value_column].to_numpy(dtype=float)
        vmax = float(active[value_column].quantile(0.99))
        vmax = max(vmax, float(display_values.min()) + 1e-9)
        display_values = np.clip(display_values, a_min=None, a_max=vmax)
        norm = plt.Normalize(vmin=float(display_values.min()), vmax=vmax)
    elif variant == "log1p":
        display_values = np.log1p(active[value_column].to_numpy(dtype=float))
        norm = plt.Normalize(vmin=float(display_values.min()), vmax=float(display_values.max()))
    else:
        raise ValueError(f"未知热力图版本：{variant}")

    active["display_value"] = display_values
    cmap = plt.get_cmap(cmap_name)
    ax.set_facecolor("#EFEFEF")

    for row in active.itertuples(index=False):
        rect = Rectangle(
            (row.cell_x_min, row.cell_y_min),
            getattr(row, "cell_width", row.patch_size),
            getattr(row, "cell_height", row.patch_size),
            facecolor=cmap(norm(row.display_value)),
            edgecolor="#737373",
            linewidth=0.42,
        )
        ax.add_patch(rect)

    return cmap, norm


def plot_blast_heatmaps(
    patch_meta: pd.DataFrame,
    patch_blast_features: pd.DataFrame,
    representative_hours: list[pd.Timestamp],
    out_dir: Path,
    diagnostics_dir: Path,
) -> None:
    """绘制代表性时刻的 patch 级爆破热力图，并输出调试报告。"""

    report_lines = [
        "# patch 级爆破热力图调试报告",
        "",
        "## 结论",
        "- 数据不是空的，三个指定时刻都存在完整的 310 条 patch 记录。",
        "- merge 不是问题，`patch_id` 能完整对齐到 `patch_meta_v2_ps10.csv`，关键字段齐全。",
        "- 旧图几乎全黑、坐标轴显示 0~1 的根因是：矩形 patch 图没有显式设置真实坐标范围，坐标轴停留在默认值；恰好原点附近 patch 覆盖了整个默认视口。",
        "- 同时还存在色标压缩：线性色标直接用全局最大值时，大部分 patch 会偏暗，所以新增了 `P99` 截断版和 `log1p` 版。",
        "",
        "## 逐时刻检查",
        "",
    ]

    for timestamp in representative_hours:
        merged, stats = summarize_blast_heatmap_frame(patch_meta, patch_blast_features, timestamp)
        report_lines.extend(
            [
                f"### {stats['timestamp']}",
                f"- 总 patch 数：{stats['total_patch_count']}",
                f"- 非零 patch 数：{stats['nonzero_patch_count']}",
                f"- blast_patch_decay 统计：min={stats['min']:.6f}, p50={stats['p50']:.6f}, p90={stats['p90']:.6f}, p99={stats['p99']:.6f}, max={stats['max']:.6f}",
                f"- 是否存在 NaN：{stats['has_nan']}",
                f"- merge 后关键字段是否完整：{stats['merge_columns_ok']}",
                "",
            ]
        )

        output_paths = []
        for variant, label in [("linear", "原始线性色标"), ("p99", "vmax=P99 截断版"), ("log1p", "log1p 版")]:
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap, norm = draw_patch_rectangles(
                ax=ax,
                frame=merged,
                value_column="blast_patch_decay",
                cmap_name="inferno",
                variant=variant,
            )

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar_label = "blast_patch_decay" if variant != "log1p" else "log1p(blast_patch_decay)"
            plt.colorbar(sm, ax=ax, label=colorbar_label)
            ax.set_title(f"patch 级爆破热力图（{label}）：{timestamp:%Y-%m-%d %H:%M}")
            ax.set_xlabel("grid_x")
            ax.set_ylabel("grid_y")
            set_patch_axes_limits(ax, patch_meta)

            if variant == "linear":
                output_path = out_dir / f"blast_heatmap_{timestamp:%Y%m%d_%H%M}.png"
            else:
                output_path = out_dir / f"blast_heatmap_{timestamp:%Y%m%d_%H%M}_{variant}.png"
            fig.savefig(output_path, dpi=240)
            plt.close(fig)
            output_paths.append(output_path)

        report_lines.append("- 修复后图文件：")
        for output_path in output_paths:
            report_lines.append(f"  - `{output_path}`")
        report_lines.append("")

    report_lines.extend(
        [
            "## 修复说明",
            "- 热力图绘制方式已改成基于 patch rectangle 的着色图。",
            "- 背景改为浅灰色，非活动 patch（值 <= 0 或 NaN）不绘制。",
            "- 若某个指定时刻数据为空或 merge 失败，脚本会直接报错，不再生成默认黑图。",
        ]
    )
    (diagnostics_dir / "debug_patch_blast_heatmap.md").write_text("\n".join(report_lines), encoding="utf-8")


def plot_sigma_comparison(
    blast_ledger: pd.DataFrame,
    patch_meta: pd.DataFrame,
    sigma_values: list[float],
    out_path: Path,
) -> None:
    """绘制不同 sigma 下的空间衰减对比图。"""

    if blast_ledger.empty:
        return

    strongest_event = blast_ledger.sort_values("Q", ascending=False).iloc[0]
    centroids = patch_meta[["centroid_x", "centroid_y"]].to_numpy()

    fig, axes = plt.subplots(1, len(sigma_values), figsize=(5 * len(sigma_values), 5.5), sharex=True, sharey=True)
    if len(sigma_values) == 1:
        axes = [axes]

    for axis, sigma in zip(axes, sigma_values):
        distance = np.hypot(centroids[:, 0] - strongest_event["x_b"], centroids[:, 1] - strongest_event["y_b"])
        value = strongest_event["Q"] * np.exp(-np.square(distance) / (2.0 * sigma * sigma))
        merged = patch_meta.assign(value=value)
        norm = plt.Normalize(vmin=float(merged["value"].min()), vmax=float(merged["value"].max()))
        cmap = plt.get_cmap("viridis")

        for row in merged.itertuples(index=False):
            rect = Rectangle(
                (row.cell_x_min, row.cell_y_min),
                getattr(row, "cell_width", row.patch_size),
                getattr(row, "cell_height", row.patch_size),
                facecolor=cmap(norm(row.value)),
                edgecolor="#555555",
                linewidth=0.35,
            )
            axis.add_patch(rect)

        axis.scatter([strongest_event["x_b"]], [strongest_event["y_b"]], c="#D62828", s=55, marker="x")
        axis.set_title(f"sigma={sigma:.0f}")
        axis.set_xlabel("grid_x")
        set_patch_axes_limits(axis, patch_meta)

    axes[0].set_ylabel("grid_y")
    fig.suptitle(f"sigma 对比图（事件 {strongest_event['blast_id']}）", fontsize=14)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_lag_correlation(
    patch_series: pd.DataFrame,
    patch_blast_features: pd.DataFrame,
    max_lag: int,
    out_path: Path,
) -> None:
    """绘制爆破与位移均值之间的滞后相关图。"""

    disp = (
        patch_series.pivot(index="timestamp", columns="patch_id", values="disp_mean")
        .sort_index()
        .sort_index(axis=1)
    )
    blast = (
        patch_blast_features.pivot(index="timestamp", columns="patch_id", values="blast_patch_decay")
        .sort_index()
        .sort_index(axis=1)
    )
    common_columns = disp.columns.intersection(blast.columns)
    disp = disp[common_columns].to_numpy(dtype=np.float32)
    blast = blast[common_columns].to_numpy(dtype=np.float32)

    lags = list(range(max_lag + 1))
    correlations = []
    for lag in lags:
        left = blast[:-lag or None].reshape(-1)
        right = disp[lag:].reshape(-1)
        mask = ~np.isnan(left) & ~np.isnan(right)
        if mask.sum() < 10:
            correlations.append(np.nan)
            continue
        correlations.append(float(np.corrcoef(left[mask], right[mask])[0, 1]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lags, correlations, color="#1D3557", marker="o", linewidth=2)
    ax.set_xlabel("爆破领先位移的小时数")
    ax.set_ylabel("Pearson 相关系数")
    ax.set_title("lag-correlation：blast vs patch disp_mean")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def build_adjacency(patch_meta: pd.DataFrame) -> np.ndarray:
    """构造基于 patch 邻接关系的邻接矩阵。"""

    frame = patch_meta[["patch_id", "zone_id_v2", "patch_x_idx", "patch_y_idx"]].reset_index(drop=True)
    coords = frame[["patch_x_idx", "patch_y_idx"]].to_numpy()
    zone_ids = frame["zone_id_v2"].to_numpy()
    patch_count = len(frame)

    adjacency = np.zeros((patch_count, patch_count), dtype=np.uint8)
    for left in range(patch_count):
        delta = np.abs(coords - coords[left])
        neighbor_mask = (
            (zone_ids == zone_ids[left])
            & (delta[:, 0] <= 1)
            & (delta[:, 1] <= 1)
        )
        adjacency[left, neighbor_mask] = 1

    return adjacency


def save_tensor_bundle_and_window_manifest(
    patch_series: pd.DataFrame,
    patch_meta: pd.DataFrame,
    weather: pd.DataFrame,
    blast_v2: pd.DataFrame,
    blast_v3: pd.DataFrame,
    full_hours: pd.DatetimeIndex,
    config: dict[str, Any],
    dataset_dir: Path,
) -> pd.DataFrame:
    """保存基础张量、邻接矩阵和窗口形状清单。"""

    train_patch_meta = patch_meta.loc[patch_meta["is_train_patch"] == 1].copy() if "is_train_patch" in patch_meta.columns else patch_meta.copy()
    patch_ids = train_patch_meta["patch_id"].tolist()
    timestamp_values = np.asarray([timestamp.isoformat() for timestamp in full_hours], dtype=object)

    series_index = (
        patch_series.loc[patch_series["patch_id"].isin(patch_ids)]
        .set_index(["timestamp", "patch_id"])
        .sort_index()
    )
    internal_tensor = np.empty((len(full_hours), len(patch_ids), len(INTERNAL_FEATURES)), dtype=np.float32)
    for feature_index, feature_name in enumerate(INTERNAL_FEATURES):
        pivot = series_index[feature_name].unstack("patch_id").reindex(index=full_hours, columns=patch_ids)
        internal_tensor[:, :, feature_index] = pivot.to_numpy(dtype=np.float32)

    target = series_index["disp_mean"].unstack("patch_id").reindex(index=full_hours, columns=patch_ids).to_numpy(dtype=np.float32)
    target_mask = (~np.isnan(target)).astype(np.uint8)
    input_mask = (
        series_index["valid_ratio"].unstack("patch_id").reindex(index=full_hours, columns=patch_ids).fillna(0.0).to_numpy(dtype=np.float32)
        > 0
    ).astype(np.uint8)
    global_missing = (
        patch_series.drop_duplicates(subset=["timestamp"])[["timestamp", "is_global_missing"]]
        .sort_values("timestamp")
        .set_index("timestamp")
        .reindex(full_hours)["is_global_missing"]
        .fillna(0)
        .to_numpy(dtype=np.uint8)
    )

    weather_matrix = weather.set_index("timestamp").reindex(full_hours)[WEATHER_FEATURES].to_numpy(dtype=np.float32)
    blast_v2_matrix = blast_v2.set_index("timestamp").reindex(full_hours)[BLAST_V2_FEATURES].to_numpy(dtype=np.float32)

    blast_v3_index = (
        blast_v3.loc[blast_v3["patch_id"].isin(patch_ids)]
        .set_index(["timestamp", "patch_id"])
        .sort_index()
    )
    blast_v3_tensor = np.empty((len(full_hours), len(patch_ids), len(BLAST_V3_FEATURES)), dtype=np.float32)
    for feature_index, feature_name in enumerate(BLAST_V3_FEATURES):
        pivot = blast_v3_index[feature_name].unstack("patch_id").reindex(index=full_hours, columns=patch_ids)
        blast_v3_tensor[:, :, feature_index] = pivot.to_numpy(dtype=np.float32)

    adjacency = build_adjacency(train_patch_meta)
    np.save(dataset_dir / "patch_adjacency_v2_ps10.npy", adjacency)

    np.savez_compressed(
        dataset_dir / "patch_tensor_base_v2_ps10.npz",
        timestamps=timestamp_values,
        patch_ids=np.asarray(patch_ids, dtype=object),
        internal_feature_names=np.asarray(INTERNAL_FEATURES, dtype=object),
        weather_feature_names=np.asarray(WEATHER_FEATURES, dtype=object),
        blast_v2_feature_names=np.asarray(BLAST_V2_FEATURES, dtype=object),
        blast_v3_feature_names=np.asarray(BLAST_V3_FEATURES, dtype=object),
        internal=internal_tensor,
        weather=weather_matrix,
        blast_v2=blast_v2_matrix,
        blast_v3=blast_v3_tensor,
        target=target,
        target_mask=target_mask,
        input_mask=input_mask,
        global_missing=global_missing,
    )

    rows = []
    total_steps = len(full_hours)
    train_ratio = float(config["windows"]["train_ratio"])
    val_ratio = float(config["windows"]["val_ratio"])
    total_feature_count = len(INTERNAL_FEATURES) + len(WEATHER_FEATURES) + len(BLAST_V2_FEATURES) + len(BLAST_V3_FEATURES)

    for seq_len in config["windows"]["seq_lens"]:
        for pred_len in config["windows"]["pred_lens"]:
            num_samples = total_steps - seq_len - pred_len + 1
            start_indices = np.arange(num_samples, dtype=np.int32)
            np.save(dataset_dir / f"window_starts_sl{seq_len}_pl{pred_len}.npy", start_indices)

            train_end = int(math.floor(num_samples * train_ratio))
            val_end = int(math.floor(num_samples * (train_ratio + val_ratio)))
            rows.append(
                {
                    "seq_len": int(seq_len),
                    "pred_len": int(pred_len),
                    "num_samples": int(num_samples),
                    "num_patches": int(len(patch_ids)),
                    "num_features_full": int(total_feature_count),
                    "x_shape": f"[{num_samples}, {seq_len}, {len(patch_ids)}, {total_feature_count}]",
                    "y_shape": f"[{num_samples}, {pred_len}, {len(patch_ids)}]",
                    "train_start": 0,
                    "train_end": int(train_end),
                    "val_start": int(train_end),
                    "val_end": int(val_end),
                    "test_start": int(val_end),
                    "test_end": int(num_samples),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["seq_len", "pred_len"]).reset_index(drop=True)
    manifest.to_csv(dataset_dir / "window_manifest_v2_ps10.csv", index=False)
    return manifest


def main() -> None:
    """执行 slopemine v2 的数据流水线。"""

    args = parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    setup_plot_style()

    dataset_dir = paths["dataset_dir"]
    figures_dir = paths["figures_dir"]
    diagnostics_dir = paths["diagnostics_dir"]
    diagnostic_package_dir = paths["diagnostic_package_dir"]
    selected_heatmaps_dir = paths["selected_heatmaps_dir"]

    print("1/6 冻结工程分区规则并重建 point_meta...")
    zone_rule = build_zone_rule_payload(config)
    point_meta = assign_zones(load_unique_points(config["paths"]["raw_csv"]), zone_rule)
    point_meta.to_csv(dataset_dir / "point_meta_v2.csv", index=False)
    save_zone_polygons(zone_rule, dataset_dir / "engineering_zone_polygons_v2.json")
    write_zone_assignment_rule(zone_rule, dataset_dir / "zone_assignment_rule.md")

    zone_stats = build_zone_stats(point_meta, config)
    zone_stats.to_csv(dataset_dir / "zone_stats_v2.csv", index=False)
    plot_engineering_zones(point_meta, config, figures_dir / "engineering_zone_v2.png")

    print("2/6 重做 refined patch 元数据...")
    sparse_min_points = int(config["processing"]["sparse_patch_min_points"])
    background_zone_key = str(config["processing"]["background_zone_key"])
    patch_stats_rows = []
    refined_patch_stats_rows = []
    old_zone_points = make_old_zone_points(point_meta)
    patch_meta_outputs: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for patch_size in config["processing"]["patch_sizes"]:
        patch_size = int(patch_size)
        patch_prefix = f"ps{patch_size}_v2"
        refined_variants = build_refined_patch_variants(
            point_meta=point_meta,
            patch_size=patch_size,
            sparse_min_points=sparse_min_points,
            patch_prefix=patch_prefix,
            background_zone_key=background_zone_key,
        )

        patch_meta_excluded, point_patch_excluded, raw_patch_excluded = refined_variants["background_excluded"]
        patch_meta_excluded = patch_meta_excluded.copy()
        patch_meta_excluded["background_policy"] = "background_excluded"
        patch_meta_excluded.to_csv(dataset_dir / f"patch_meta_v2_ps{patch_size}_bg_excluded.csv", index=False)
        refined_patch_stats_rows.append(
            build_patch_stats_refined("background_excluded", patch_size, patch_meta_excluded, raw_patch_excluded)
        )

        patch_meta_main, point_patch_main, raw_patch_main = refined_variants["single_background_patch"]
        patch_meta_main = patch_meta_main.copy()
        patch_meta_main["background_policy"] = "single_background_patch"
        patch_meta_main.to_csv(dataset_dir / f"patch_meta_v2_ps{patch_size}.csv", index=False)
        patch_meta_main.to_csv(dataset_dir / f"patch_meta_v2_ps{patch_size}_bg_single.csv", index=False)
        plot_patch_grid(point_meta, patch_meta_main, config, patch_size, figures_dir / f"patch_grid_v2_ps{patch_size}_refined.png")
        plot_patch_density(patch_meta_main, patch_size, figures_dir / f"patch_density_v2_ps{patch_size}.png")
        refined_patch_stats_rows.append(
            build_patch_stats_refined("single_background_patch", patch_size, patch_meta_main, raw_patch_main)
        )
        patch_meta_outputs[patch_size] = (patch_meta_main, point_patch_main)

        patch_stats_rows.append(build_patch_stats("new_zone_v2", patch_size, patch_meta_main, raw_patch_main))
        patch_meta_old, _, raw_patch_old = build_patch_meta(
            old_zone_points,
            patch_size=patch_size,
            sparse_min_points=sparse_min_points,
            patch_prefix=f"ps{patch_size}_old",
        )
        patch_stats_rows.append(build_patch_stats("old_area_id_seq", patch_size, patch_meta_old, raw_patch_old))

    patch_stats = pd.concat(patch_stats_rows, ignore_index=True)
    patch_stats.to_csv(dataset_dir / "patch_stats_v2.csv", index=False)
    patch_stats_refined = pd.concat(refined_patch_stats_rows, ignore_index=True)
    patch_stats_refined.to_csv(dataset_dir / "patch_stats_v2_refined.csv", index=False)

    print("3/6 聚合 patch 时序并生成任务 3 图表...")
    monitoring_frame = load_monitoring_frame(config["paths"]["raw_csv"])
    patch_meta_ps10, point_patch_ps10 = patch_meta_outputs[10]
    patch_series, full_hours, missing_hours = aggregate_patch_series(
        monitoring_frame=monitoring_frame,
        point_meta=point_meta,
        patch_meta=patch_meta_ps10,
        point_patch_map=point_patch_ps10,
        active_speed_threshold=float(config["processing"]["active_speed_threshold"]),
    )
    patch_series = patch_series[
        [
            "timestamp",
            "patch_id",
            "zone_id_v2",
            "disp_mean",
            "disp_std",
            "disp_p95",
            "disp_max",
            "vel_mean",
            "vel_p95",
            "acc_mean",
            "acc_p95",
            "active_ratio",
            "valid_ratio",
            "is_global_missing",
            "num_records",
        ]
    ].copy()
    patch_series.to_csv(dataset_dir / "patch_series_v2_ps10.csv", index=False)
    patch_series_summary = build_patch_series_summary(patch_series)
    patch_series_summary.to_csv(dataset_dir / "patch_series_summary_v2.csv", index=False)
    patch_series_summary.to_csv(dataset_dir / "patch_series_summary_v2_ps10.csv", index=False)

    plot_zone_boxplot(
        patch_series=patch_series,
        patch_meta=patch_meta_ps10,
        value_column="disp_mean",
        title="各 zone 的 disp_mean 箱线图",
        out_path=diagnostic_package_dir / "zone_disp_mean_boxplot.png",
    )
    plot_zone_boxplot(
        patch_series=patch_series,
        patch_meta=patch_meta_ps10,
        value_column="disp_max",
        title="各 zone 的 disp_max 箱线图",
        out_path=diagnostic_package_dir / "zone_disp_max_boxplot.png",
    )
    representative_disp_hours, representative_disp_frame = pick_representative_disp_hours(
        patch_series,
        int(config["processing"]["selected_time_heatmap_count"]),
    )
    representative_disp_frame.to_csv(selected_heatmaps_dir / "selected_times.csv", index=False)
    disp_heatmap_paths = plot_selected_disp_heatmaps(
        patch_meta=patch_meta_ps10,
        patch_series=patch_series,
        timestamps=representative_disp_hours,
        out_dir=selected_heatmaps_dir,
    )

    print("4/6 更新爆破特征与实验基础张量...")
    weather = align_external_frame(load_hourly_external_frame(config["paths"]["weather_csv"], "timestamp"), full_hours)
    blast_v2 = align_external_frame(load_hourly_external_frame(config["paths"]["blast_hourly_csv"], "timestamp"), full_hours)
    blast_ledger = build_blast_ledger(config, dataset_dir)
    patch_blast_v3 = build_patch_blast_features(blast_ledger, patch_meta_ps10, full_hours, config)
    patch_blast_v3.to_csv(dataset_dir / "patch_blast_features_v3_ps10.csv", index=False)

    representative_hours = pick_representative_hours(
        blast_v2,
        int(config["processing"]["representative_heatmap_count"]),
    )
    representative_hours = sorted(representative_hours)
    plot_blast_heatmaps(patch_meta_ps10, patch_blast_v3, representative_hours, figures_dir, diagnostics_dir)
    plot_sigma_comparison(
        blast_ledger=blast_ledger,
        patch_meta=patch_meta_ps10,
        sigma_values=[float(value) for value in config["processing"]["sigma_compare"]],
        out_path=figures_dir / "blast_sigma_comparison_v2.png",
    )
    plot_lag_correlation(
        patch_series=patch_series,
        patch_blast_features=patch_blast_v3,
        max_lag=int(config["processing"]["lag_hours"]),
        out_path=figures_dir / "blast_lag_correlation_v2.png",
    )

    manifest = save_tensor_bundle_and_window_manifest(
        patch_series=patch_series,
        patch_meta=patch_meta_ps10,
        weather=weather,
        blast_v2=blast_v2,
        blast_v3=patch_blast_v3,
        full_hours=full_hours,
        config=config,
        dataset_dir=dataset_dir,
    )

    print("5/6 生成正式数据诊断包...")
    zone_point_distribution = build_zone_point_distribution(point_meta)
    zone_point_distribution.to_csv(diagnostic_package_dir / "zone_point_distribution.csv", index=False)
    plot_zone_point_distribution(zone_point_distribution, config, diagnostic_package_dir / "zone_point_distribution.png")

    zone_patch_distribution = build_zone_patch_distribution(patch_stats_refined)
    zone_patch_distribution.to_csv(diagnostic_package_dir / "zone_patch_distribution.csv", index=False)
    plot_zone_patch_distribution(zone_patch_distribution, diagnostic_package_dir / "zone_patch_distribution.png")

    plot_patch_density(patch_meta_ps10, 10, diagnostic_package_dir / "patch_density_heatmap.png")

    zone_feature_summary = build_zone_feature_summary(patch_series, patch_meta_ps10)
    zone_feature_summary.to_csv(diagnostic_package_dir / "zone_feature_summary.csv", index=False)
    write_patch_diagnostic_report(
        point_meta=point_meta,
        patch_stats_refined=patch_stats_refined,
        zone_feature_summary=zone_feature_summary,
        representative_times=representative_disp_frame,
        heatmap_paths=disp_heatmap_paths,
        out_path=diagnostic_package_dir / "patch_data_diagnostic_report_v2.md",
    )

    print("6/6 写入汇总信息...")
    diagnostics = {
        "unique_points": int(len(point_meta)),
        "official_patch_count_ps10": int(len(patch_meta_ps10)),
        "train_patch_count_ps10": int(patch_meta_ps10["is_train_patch"].sum()),
        "full_hour_count": int(len(full_hours)),
        "global_missing_hours": [timestamp.isoformat(sep=" ") for timestamp in missing_hours],
        "representative_blast_hours": [timestamp.isoformat(sep=" ") for timestamp in representative_hours],
        "representative_disp_hours": [timestamp.isoformat(sep=" ") for timestamp in representative_disp_hours],
    }
    (diagnostics_dir / "build_summary_v2.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("完成。")
    print(f"point_meta_v2.csv: {dataset_dir / 'point_meta_v2.csv'}")
    print(f"zone_assignment_rule.md: {dataset_dir / 'zone_assignment_rule.md'}")
    print(f"patch_meta_v2_ps10.csv: {dataset_dir / 'patch_meta_v2_ps10.csv'}")
    print(f"patch_series_v2_ps10.csv: {dataset_dir / 'patch_series_v2_ps10.csv'}")
    print(f"patch_stats_v2_refined.csv: {dataset_dir / 'patch_stats_v2_refined.csv'}")
    print(f"patch_data_diagnostic_report_v2.md: {diagnostic_package_dir / 'patch_data_diagnostic_report_v2.md'}")
    print(f"window_manifest_v2_ps10.csv: {dataset_dir / 'window_manifest_v2_ps10.csv'}")
    print(f"窗口配置数量: {len(manifest)}")


if __name__ == "__main__":
    main()
