#!/usr/bin/env python3
"""这个脚本负责生成代理爆破坐标台账和对应的 proxy V3 特征。
相关文件：config_v2.toml、dataset/slopemine_v2/point_meta_v2.csv、dataset/slopemine_v2/patch_meta_v2_ps10.csv
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
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

from matplotlib import pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from plot_utils_v2 import setup_plot_style


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate proxy blast ledgers and proxy V3 features.")
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

    config["paths"]["blast_hourly_csv"] = Path(config["paths"]["blast_hourly_csv"]).expanduser().resolve()
    return config


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """创建代理爆破结果目录。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_dir = Path(config["paths"]["output_dir"]) / "proxy_blast"
    figure_dir = output_dir / "figures"

    for path in [dataset_dir, output_dir, figure_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "figure_dir": figure_dir,
    }


def parse_zone_assignment_rule(rule_path: Path) -> dict[str, dict[str, Any]]:
    """从 markdown 规则文件中解析 zone 名称、key 和 polygon 顶点。"""

    content = rule_path.read_text(encoding="utf-8").splitlines()
    zone_rows: dict[str, dict[str, Any]] = {}
    current_zone_id: int | None = None

    heading_pattern = re.compile(r"^### zone_id_v2=(\d+)\s+(.+)$")
    key_pattern = re.compile(r"^- zone_key_v2：`([^`]+)`$")
    polygon_pattern = re.compile(r"^- polygon：(.+)$")
    point_pattern = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")

    for line in content:
        heading_match = heading_pattern.match(line.strip())
        if heading_match:
            current_zone_id = int(heading_match.group(1))
            zone_rows[str(current_zone_id)] = {
                "zone_id_v2": current_zone_id,
                "zone_name_v2": heading_match.group(2).strip(),
            }
            continue

        if current_zone_id is None:
            continue

        key_match = key_pattern.match(line.strip())
        if key_match:
            zone_rows[str(current_zone_id)]["zone_key_v2"] = key_match.group(1)
            continue

        polygon_match = polygon_pattern.match(line.strip())
        if polygon_match:
            polygon = [
                [float(x_value), float(y_value)]
                for x_value, y_value in point_pattern.findall(polygon_match.group(1))
            ]
            zone_rows[str(current_zone_id)]["polygon"] = polygon

    if len(zone_rows) < 5:
        raise ValueError("zone_assignment_rule.md 解析失败，未识别到完整的 5 个 zone。")
    return zone_rows


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """读取生成代理台账所需的输入表。"""

    dataset_dir = Path(config["paths"]["dataset_dir"])
    point_meta = pd.read_csv(dataset_dir / "point_meta_v2.csv")
    patch_meta = pd.read_csv(dataset_dir / "patch_meta_v2_ps10.csv")
    current_v3 = pd.read_csv(dataset_dir / "patch_blast_features_v3_ps10.csv", parse_dates=["timestamp"])
    blast_hourly = pd.read_csv(config["paths"]["blast_hourly_csv"], parse_dates=["timestamp"])
    zone_rule = parse_zone_assignment_rule(dataset_dir / "zone_assignment_rule.md")
    return {
        "point_meta": point_meta,
        "patch_meta": patch_meta,
        "current_v3": current_v3,
        "blast_hourly": blast_hourly,
        "zone_rule": zone_rule,
    }


def get_variant_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """整理三套代理版本参数。"""

    variants = config["proxy_blast"]["variants"]
    rows = []
    for variant_name, variant_cfg in variants.items():
        weights = {
            "toe": float(variant_cfg["toe_weight"]),
            "middle_slope": float(variant_cfg["middle_slope_weight"]),
            "platform": float(variant_cfg["platform_weight"]),
            "crest": float(variant_cfg["crest_weight"]),
        }
        weight_sum = sum(weights.values())
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{variant_name} 的 zone 权重之和不是 1。")
        rows.append(
            {
                "variant_name": variant_name,
                "variant_rank": int(variant_cfg["variant_rank"]),
                "ledger_filename": str(variant_cfg["ledger_filename"]),
                "feature_filename": str(variant_cfg["feature_filename"]),
                "min_event_distance": float(variant_cfg["min_event_distance"]),
                "sigma_min": float(variant_cfg["sigma_min"]),
                "sigma_max": float(variant_cfg["sigma_max"]),
                "weights": weights,
            }
        )
    return sorted(rows, key=lambda row: row["variant_rank"])


def build_candidate_patch_frame(
    patch_meta: pd.DataFrame,
    variant_spec: dict[str, Any],
) -> pd.DataFrame:
    """构造可用于代理抽样的 patch 候选表。"""

    frame = patch_meta.loc[patch_meta["zone_key_v2"] != "stable_background"].copy()
    frame = frame[
        [
            "patch_id",
            "zone_id_v2",
            "zone_key_v2",
            "zone_name_v2",
            "centroid_x",
            "centroid_y",
        ]
    ].reset_index(drop=True)

    zone_counts = frame["zone_key_v2"].value_counts().to_dict()
    frame["zone_weight"] = frame["zone_key_v2"].map(variant_spec["weights"])
    frame["sampling_weight"] = frame.apply(
        lambda row: float(row["zone_weight"]) / float(zone_counts[row["zone_key_v2"]]),
        axis=1,
    )
    return frame


def select_proxy_patch_rows(
    candidate_frame: pd.DataFrame,
    event_count: int,
    min_event_distance: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """在满足最小间距的前提下抽取代理爆破 patch。"""

    selected_indices: list[int] = []
    coords = candidate_frame[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    weights = candidate_frame["sampling_weight"].to_numpy(dtype=float)

    for _ in range(event_count):
        eligible_mask = np.ones(len(candidate_frame), dtype=bool)
        if selected_indices:
            selected_coords = coords[selected_indices]
            distance_matrix = np.sqrt(((coords[:, None, :] - selected_coords[None, :, :]) ** 2).sum(axis=2))
            eligible_mask &= (distance_matrix >= min_event_distance).all(axis=1)
            eligible_mask[selected_indices] = False

        eligible_indices = np.flatnonzero(eligible_mask)
        if len(eligible_indices) == 0:
            raise ValueError(f"在最小间距 {min_event_distance} 下无法抽出 {event_count} 个代理事件。")

        eligible_weights = weights[eligible_indices]
        eligible_weights = eligible_weights / eligible_weights.sum()
        chosen_index = int(rng.choice(eligible_indices, p=eligible_weights))
        selected_indices.append(chosen_index)

    return candidate_frame.iloc[selected_indices].reset_index(drop=True)


def resolve_charge_value(hour_row: pd.Series) -> float:
    """按要求计算单个代理事件的装药量。"""

    blast_count = max(int(round(float(hour_row["blast_count"]))), 1)
    total_charge = float(hour_row["total_charge_kg"]) if pd.notna(hour_row["total_charge_kg"]) else np.nan
    mean_charge = float(hour_row["mean_charge_kg"]) if pd.notna(hour_row["mean_charge_kg"]) else np.nan

    if pd.notna(total_charge) and total_charge > 0:
        return total_charge / blast_count
    if pd.notna(mean_charge):
        return mean_charge
    raise ValueError(f"{hour_row['timestamp']} 无法从 total_charge_kg 或 mean_charge_kg 计算代理 Q。")


def build_proxy_ledger(
    blast_hourly: pd.DataFrame,
    patch_meta: pd.DataFrame,
    variant_spec: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    """生成单个版本的代理爆破坐标台账。"""

    base_seed = int(config["proxy_blast"]["base_seed"])
    rng = np.random.default_rng(base_seed + variant_spec["variant_rank"] * 1000)
    candidate_frame = build_candidate_patch_frame(patch_meta, variant_spec)

    rows: list[dict[str, Any]] = []
    active_hours = blast_hourly.loc[blast_hourly["blast_count"].fillna(0) > 0].copy()
    active_hours = active_hours.sort_values("timestamp").reset_index(drop=True)

    for hour_row in active_hours.itertuples(index=False):
        event_count = max(int(round(float(hour_row.blast_count))), 1)
        selected = select_proxy_patch_rows(
            candidate_frame=candidate_frame,
            event_count=event_count,
            min_event_distance=float(variant_spec["min_event_distance"]),
            rng=rng,
        )

        charge_value = resolve_charge_value(pd.Series(hour_row._asdict()))
        for event_index, patch_row in enumerate(selected.itertuples(index=False), start=1):
            sigma_value = float(rng.uniform(variant_spec["sigma_min"], variant_spec["sigma_max"]))
            rows.append(
                {
                    "blast_id": f"proxy_{variant_spec['variant_name']}_{pd.Timestamp(hour_row.timestamp):%Y%m%d_%H%M}_{event_index:02d}",
                    "timestamp": pd.Timestamp(hour_row.timestamp),
                    "x_b": float(patch_row.centroid_x),
                    "y_b": float(patch_row.centroid_y),
                    "Q": float(charge_value),
                    "sigma": sigma_value,
                    "sigma_proxy": sigma_value,
                    "zone_id_proxy": int(patch_row.zone_id_v2),
                    "zone_key_proxy": str(patch_row.zone_key_v2),
                    "zone_name_proxy": str(patch_row.zone_name_v2),
                    "patch_id_proxy": str(patch_row.patch_id),
                    "coordinate_source": "proxy_inferred",
                    "is_real_coordinate": 0,
                    "proxy_variant": variant_spec["variant_name"],
                    "event_index_in_hour": event_index,
                    "hourly_blast_count": event_count,
                    "min_event_distance_rule": float(variant_spec["min_event_distance"]),
                }
            )

    ledger = pd.DataFrame(rows).sort_values(["timestamp", "event_index_in_hour"]).reset_index(drop=True)
    return ledger


def build_patch_blast_features_proxy(
    blast_ledger: pd.DataFrame,
    patch_meta: pd.DataFrame,
    full_hours: pd.DatetimeIndex,
    config: dict[str, Any],
) -> pd.DataFrame:
    """按照现有 V3 口径计算代理 patch 爆破特征。"""

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


def summarize_feature_distribution(name: str, features: pd.DataFrame) -> dict[str, Any]:
    """汇总单个特征表的非零比例和分位数。"""

    values = features["blast_patch_decay"].to_numpy(dtype=float)
    positive_mask = values > 0
    return {
        "name": name,
        "row_count": int(len(values)),
        "nonzero_patch_ratio": float(positive_mask.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def compute_same_hour_distance_summary(ledger: pd.DataFrame) -> dict[str, float]:
    """统计同一小时多事件之间的最小距离。"""

    distances = []
    for _, frame in ledger.groupby("timestamp", sort=True):
        if len(frame) <= 1:
            continue
        coords = frame[["x_b", "y_b"]].to_numpy(dtype=float)
        for left in range(len(coords)):
            for right in range(left + 1, len(coords)):
                distances.append(float(np.hypot(coords[left, 0] - coords[right, 0], coords[left, 1] - coords[right, 1])))

    if not distances:
        return {"min": np.nan, "p50": np.nan, "max": np.nan}
    return {
        "min": float(np.min(distances)),
        "p50": float(np.quantile(distances, 0.50)),
        "max": float(np.max(distances)),
    }


def set_point_axes(ax: plt.Axes, point_meta: pd.DataFrame, padding: float = 8.0) -> None:
    """按点位真实范围设置坐标轴。"""

    x_min = float(point_meta["grid_x"].min()) - padding
    x_max = float(point_meta["grid_x"].max()) + padding
    y_min = float(point_meta["grid_y"].min()) - padding
    y_max = float(point_meta["grid_y"].max()) + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def plot_proxy_spatial_distribution(
    *,
    point_meta: pd.DataFrame,
    zone_rule: dict[str, dict[str, Any]],
    ledgers: dict[str, pd.DataFrame],
    config: dict[str, Any],
    out_path: Path,
) -> None:
    """绘制与工程分区叠加的代理爆破点分布图。"""

    proxy_cfg = config["proxy_blast"]
    zone_colors = {
        zone_cfg["zone_key"]: zone_cfg["color"]
        for zone_cfg in (
            {
                "zone_key": zone_key,
                "color": config["zone_catalog"][zone_key]["color"],
            }
            for zone_key in config["zone_catalog"]
        )
    }

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), sharex=True, sharey=True)
    for axis, (variant_name, ledger) in zip(axes, ledgers.items()):
        axis.scatter(
            point_meta["grid_x"],
            point_meta["grid_y"],
            s=3,
            c="#D8D8D8",
            alpha=0.22,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )

        for zone_row in zone_rule.values():
            polygon = zone_row["polygon"]
            x_values = [vertex[0] for vertex in polygon] + [polygon[0][0]]
            y_values = [vertex[1] for vertex in polygon] + [polygon[0][1]]
            axis.plot(
                x_values,
                y_values,
                color=zone_colors.get(zone_row["zone_key_v2"], "#777777"),
                linewidth=0.9,
                linestyle=(0, (4, 2)),
                alpha=0.75,
                zorder=2,
            )

        q_min = float(ledger["Q"].min())
        q_max = float(ledger["Q"].max())
        size_min = float(proxy_cfg["plot_event_size_min"])
        size_max = float(proxy_cfg["plot_event_size_max"])
        if math.isclose(q_min, q_max):
            sizes = np.full(len(ledger), (size_min + size_max) / 2.0)
        else:
            sizes = size_min + (ledger["Q"].to_numpy(dtype=float) - q_min) / (q_max - q_min) * (size_max - size_min)

        colors = [zone_colors.get(zone_key, "#444444") for zone_key in ledger["zone_key_proxy"]]
        axis.scatter(
            ledger["x_b"],
            ledger["y_b"],
            s=sizes,
            c=colors,
            edgecolors="#111111",
            linewidths=0.55,
            alpha=0.95,
            marker="o",
            zorder=3,
        )
        axis.set_title(f"{variant_name}\n代理事件数={len(ledger)}")
        axis.set_xlabel("grid_x")
        set_point_axes(axis, point_meta)

    axes[0].set_ylabel("grid_y")
    fig.suptitle("代理爆破点空间分布图（叠加工程分区参考边界）", fontsize=15)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_proxy_summary_markdown(
    *,
    output_dir: Path,
    ledgers: dict[str, pd.DataFrame],
    feature_compare: pd.DataFrame,
    variant_specs: list[dict[str, Any]],
) -> None:
    """输出代理爆破台账与 proxy V3 的摘要说明。"""

    table_lines = [
        "| name | row_count | nonzero_patch_ratio | p50 | p90 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in feature_compare.itertuples(index=False):
        table_lines.append(
            f"| {row.name} | {int(row.row_count)} | {float(row.nonzero_patch_ratio):.6f} | {float(row.p50):.6f} | {float(row.p90):.6f} | {float(row.p99):.6f} | {float(row.max):.6f} |"
        )

    lines = [
        "# proxy_blast_ledger_summary",
        "",
        "- 本页所有坐标均为代理推断坐标，不是实时真实爆破坐标。",
        "- 主代理版本为 `balanced`，并额外导出 `conservative` 与 `localized` 用于敏感性分析。",
        "",
        "## 版本参数",
    ]

    for variant_spec in variant_specs:
        weights = variant_spec["weights"]
        lines.append(
            f"- {variant_spec['variant_name']}：min_event_distance={variant_spec['min_event_distance']:.1f}，sigma∈[{variant_spec['sigma_min']:.1f}, {variant_spec['sigma_max']:.1f}]，weights=(toe={weights['toe']:.2f}, middle_slope={weights['middle_slope']:.2f}, platform={weights['platform']:.2f}, crest={weights['crest']:.2f})"
        )

    lines.extend(["", "## 台账摘要"])
    for variant_name, ledger in ledgers.items():
        zone_counts = ledger["zone_name_proxy"].value_counts().to_dict()
        distance_stats = compute_same_hour_distance_summary(ledger)
        lines.extend(
            [
                f"### {variant_name}",
                f"- 代理事件数：{len(ledger)}",
                f"- 爆破小时数：{ledger['timestamp'].nunique()}",
                f"- zone 分布：{zone_counts}",
                f"- Q 均值：{float(ledger['Q'].mean()):.3f}",
                f"- 同小时多事件最小距离：{distance_stats['min']:.3f}",
                "",
            ]
        )

    lines.extend(["## proxy V3 对比", "", *table_lines, ""])
    (output_dir / "proxy_blast_ledger_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_disclaimer(output_dir: Path) -> None:
    """输出代理坐标免责声明。"""

    content = [
        "# proxy_blast_coordinate_disclaimer",
        "",
        "- 本目录中的 `proxy_blast_ledger_*` 和 `patch_blast_features_v3_ps10_proxy*` 基于代理推断坐标生成。",
        "- 这些坐标不是实时真实爆破作业坐标，不代表任何真实施工台账、调度记录或现场定位结果。",
        "- 它们仅用于 V3 空间衰减方法开发、参数敏感性分析和稳定性实验。",
        "- 任何涉及真实爆破作业复盘、生产调度、安全追溯或工程决策的场景，都不能使用这些代理坐标替代真实数据。",
        "- 文件中的 `coordinate_source` 固定为 `proxy_inferred`，`is_real_coordinate` 固定为 `0`，用于避免误用。",
    ]
    (output_dir / "proxy_blast_coordinate_disclaimer.md").write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    """执行代理爆破台账与 proxy V3 生成。"""

    args = parse_args()
    config = load_config(args.config)
    setup_plot_style()
    paths = ensure_output_dirs(config)
    inputs = load_inputs(config)

    dataset_dir = paths["dataset_dir"]
    output_dir = paths["output_dir"]
    figure_dir = paths["figure_dir"]

    patch_meta = inputs["patch_meta"]
    point_meta = inputs["point_meta"]
    current_v3 = inputs["current_v3"]
    blast_hourly = inputs["blast_hourly"]
    zone_rule = inputs["zone_rule"]
    full_hours = pd.DatetimeIndex(sorted(current_v3["timestamp"].drop_duplicates()))

    variant_specs = get_variant_specs(config)
    ledgers: dict[str, pd.DataFrame] = {}
    feature_tables: dict[str, pd.DataFrame] = {}

    for variant_spec in variant_specs:
        ledger = build_proxy_ledger(
            blast_hourly=blast_hourly,
            patch_meta=patch_meta,
            variant_spec=variant_spec,
            config=config,
        )
        feature_table = build_patch_blast_features_proxy(
            blast_ledger=ledger,
            patch_meta=patch_meta,
            full_hours=full_hours,
            config=config,
        )

        ledger.to_csv(dataset_dir / variant_spec["ledger_filename"], index=False)
        feature_table.to_csv(dataset_dir / variant_spec["feature_filename"], index=False)
        ledgers[variant_spec["variant_name"]] = ledger
        feature_tables[variant_spec["variant_name"]] = feature_table

    main_variant = str(config["proxy_blast"]["main_variant"])
    main_ledger_path = dataset_dir / "proxy_blast_ledger_v1.csv"
    main_feature_path = dataset_dir / "patch_blast_features_v3_ps10_proxy.csv"
    ledgers[main_variant].to_csv(main_ledger_path, index=False)
    feature_tables[main_variant].to_csv(main_feature_path, index=False)

    compare_rows = [summarize_feature_distribution("current_v3", current_v3)]
    compare_rows.extend(
        summarize_feature_distribution(f"proxy_{variant_name}", feature_table)
        for variant_name, feature_table in feature_tables.items()
    )
    feature_compare = pd.DataFrame(compare_rows)
    feature_compare.to_csv(output_dir / "proxy_blast_feature_compare.csv", index=False)

    build_proxy_summary_markdown(
        output_dir=output_dir,
        ledgers=ledgers,
        feature_compare=feature_compare,
        variant_specs=variant_specs,
    )
    build_disclaimer(output_dir)
    plot_proxy_spatial_distribution(
        point_meta=point_meta,
        zone_rule=zone_rule,
        ledgers=ledgers,
        config=config,
        out_path=figure_dir / "proxy_blast_spatial_distribution.png",
    )

    print(dataset_dir / "proxy_blast_ledger_v1.csv")
    print(dataset_dir / "proxy_blast_ledger_v1_conservative.csv")
    print(dataset_dir / "proxy_blast_ledger_v2_balanced.csv")
    print(dataset_dir / "proxy_blast_ledger_v3_localized.csv")
    print(dataset_dir / "patch_blast_features_v3_ps10_proxy.csv")
    print(output_dir / "proxy_blast_ledger_summary.md")
    print(output_dir / "proxy_blast_coordinate_disclaimer.md")
    print(figure_dir / "proxy_blast_spatial_distribution.png")


if __name__ == "__main__":
    main()
