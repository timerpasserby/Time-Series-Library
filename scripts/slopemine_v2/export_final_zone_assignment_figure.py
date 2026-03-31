#!/usr/bin/env python3
"""这个脚本负责导出论文用的最终工程分区结果图。
相关文件：config_v2.toml、dataset/slopemine_v2/point_meta_v2.csv、dataset/slopemine_v2/zone_assignment_rule.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

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

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_utils_v2 import setup_plot_style


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Export final zone assignment figure for slopemine v2.")
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
        config["paths"][key] = (repo_root / raw_value).resolve() if not raw_value.is_absolute() else raw_value.resolve()

    return config


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取最终点位归属表和冻结分区边界。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    point_meta = pd.read_csv(dataset_dir / "point_meta_v2.csv")
    zone_rule = json.loads((dataset_dir / "engineering_zone_polygons_v2.json").read_text(encoding="utf-8"))
    return point_meta, zone_rule


def set_point_axes(ax: plt.Axes, point_meta: pd.DataFrame, padding: float = 8.0) -> None:
    """按点位真实范围设置坐标轴。"""

    x_min = float(point_meta["grid_x"].min()) - padding
    x_max = float(point_meta["grid_x"].max()) + padding
    y_min = float(point_meta["grid_y"].min()) - padding
    y_max = float(point_meta["grid_y"].max()) + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def build_zone_lookup(zone_rule: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """把冻结规则整理成便于绘图的索引。"""

    return {
        int(zone_row["zone_id_v2"]): {
            "zone_name_v2": str(zone_row["zone_name_v2"]),
            "color": str(zone_row["color"]),
            "polygon": zone_row["polygon"],
            "priority_rank": int(zone_row["priority_rank"]),
        }
        for zone_row in zone_rule["zones"]
    }


def plot_final_zone_assignment(
    point_meta: pd.DataFrame,
    zone_rule: dict[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """绘制仅显示最终 zone 归属的论文版工程分区结果图。"""

    zone_lookup = build_zone_lookup(zone_rule)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_facecolor("white")

    legend_handles: list[Line2D] = []
    for zone_id in sorted(zone_lookup):
        zone_cfg = zone_lookup[zone_id]
        polygon = zone_cfg["polygon"]
        x_values = [vertex[0] for vertex in polygon] + [polygon[0][0]]
        y_values = [vertex[1] for vertex in polygon] + [polygon[0][1]]
        ax.plot(
            x_values,
            y_values,
            color=zone_cfg["color"],
            linewidth=0.95,
            linestyle=(0, (4, 2)),
            alpha=0.75,
            zorder=1,
        )

    for zone_id in sorted(zone_lookup):
        zone_cfg = zone_lookup[zone_id]
        subset = point_meta.loc[point_meta["zone_id_v2"] == zone_id]
        ax.scatter(
            subset["grid_x"],
            subset["grid_y"],
            s=9,
            c=zone_cfg["color"],
            alpha=0.95,
            linewidths=0.0,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=zone_cfg["color"],
                markeredgecolor="none",
                markersize=7,
                label=f"zone_id_v2={zone_id} {zone_cfg['zone_name_v2']}",
            )
        )

    ax.set_title("最终工程分区结果图", pad=10)
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.grid(False)
    set_point_axes(ax, point_meta)
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, title="最终 zone 归属")

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    """执行最终 zone 赋值图导出。"""

    args = parse_args()
    config = load_config(args.config)
    setup_plot_style()

    output_dir = Path(config["paths"]["output_dir"]) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    point_meta, zone_rule = load_inputs(config)
    plot_final_zone_assignment(
        point_meta=point_meta,
        zone_rule=zone_rule,
        png_path=output_dir / "final_zone_assignment_v2.png",
        pdf_path=output_dir / "final_zone_assignment_v2.pdf",
    )

    print(output_dir / "final_zone_assignment_v2.png")
    print(output_dir / "final_zone_assignment_v2.pdf")


if __name__ == "__main__":
    main()
