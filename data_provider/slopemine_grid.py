"""把 ps10 patch 数据聚合成固定规则网格，并提供 SpatialGNN-TimeFilter 所需基础对象。
相关文件：patch_tensor_base_v2_ps10.npz、patch_meta_v2_ps10_bg_excluded.csv、experiment_split_plan_v1.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_provider.slopemine_formal import FormalWindowBundle, load_formal_window_bundle


@dataclass(frozen=True)
class SpatialGridBundle:
    """规则网格级别的时序张量。"""

    timestamps: np.ndarray
    node_ids: np.ndarray
    internal_feature_names: list[str]
    weather_feature_names: list[str]
    blast_v2_feature_names: list[str]
    blast_v3_feature_names: list[str]
    internal: np.ndarray
    weather: np.ndarray
    blast_v2: np.ndarray
    blast_v3: np.ndarray
    target: np.ndarray
    input_mask: np.ndarray
    target_mask: np.ndarray
    adjacency: np.ndarray
    edge_index: np.ndarray


def build_dense_grid_adjacency(grid_height: int, grid_width: int, *, eight_neighbor: bool = True) -> np.ndarray:
    """构造规则网格邻接矩阵。"""

    num_nodes = int(grid_height) * int(grid_width)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight_neighbor:
        neighbor_offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    for node_y in range(int(grid_height)):
        for node_x in range(int(grid_width)):
            src = node_y * int(grid_width) + node_x
            for dy, dx in neighbor_offsets:
                dst_y = node_y + dy
                dst_x = node_x + dx
                if 0 <= dst_y < int(grid_height) and 0 <= dst_x < int(grid_width):
                    dst = dst_y * int(grid_width) + dst_x
                    adjacency[src, dst] = 1.0
    return adjacency


def adjacency_to_edge_index(adjacency: np.ndarray) -> np.ndarray:
    """把 dense adjacency 转成 COO edge_index。"""

    rows, cols = np.where(adjacency > 0)
    return np.stack([rows.astype(np.int64), cols.astype(np.int64)], axis=0)


def build_regular_grid_meta(
    patch_meta: pd.DataFrame,
    *,
    grid_height: int,
    grid_width: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """基于 patch centroid 把现有 ps10 patch 映射到规则网格。"""

    x_min = float(patch_meta["cell_x_min"].min())
    x_max = float(patch_meta["cell_x_max"].max())
    y_min = float(patch_meta["cell_y_min"].min())
    y_max = float(patch_meta["cell_y_max"].max())

    x_edges = np.linspace(x_min, x_max, int(grid_width) + 1, dtype=np.float64)
    y_edges = np.linspace(y_min, y_max, int(grid_height) + 1, dtype=np.float64)

    centroid_x = patch_meta["centroid_x"].to_numpy(dtype=np.float64)
    centroid_y = patch_meta["centroid_y"].to_numpy(dtype=np.float64)
    node_x_idx = np.searchsorted(x_edges[1:-1], centroid_x, side="right").astype(np.int32)
    node_y_idx = np.searchsorted(y_edges[1:-1], centroid_y, side="right").astype(np.int32)
    node_index = node_y_idx * int(grid_width) + node_x_idx

    rows: list[dict[str, Any]] = []
    for gy in range(int(grid_height)):
        for gx in range(int(grid_width)):
            node_id = gy * int(grid_width) + gx
            member_mask = node_index == node_id
            member_patch_ids = patch_meta.loc[member_mask, "patch_id"].astype(str).tolist()
            rows.append(
                {
                    "node_id": f"g{int(grid_height)}x{int(grid_width)}_n{node_id:03d}",
                    "node_index": int(node_id),
                    "node_x_idx": int(gx),
                    "node_y_idx": int(gy),
                    "cell_x_min": float(x_edges[gx]),
                    "cell_x_max": float(x_edges[gx + 1]),
                    "cell_y_min": float(y_edges[gy]),
                    "cell_y_max": float(y_edges[gy + 1]),
                    "centroid_x": float((x_edges[gx] + x_edges[gx + 1]) / 2.0),
                    "centroid_y": float((y_edges[gy] + y_edges[gy + 1]) / 2.0),
                    "source_patch_count": int(member_mask.sum()),
                    "is_empty_node": int(member_mask.sum() == 0),
                    "source_patch_ids": ",".join(member_patch_ids),
                }
            )
    return pd.DataFrame(rows), node_index


def _weighted_patch_aggregate(
    values: np.ndarray,
    valid_mask: np.ndarray,
    patch_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对 patch 维做按 num_points 加权的 masked mean。"""

    weights = valid_mask.astype(np.float32) * patch_weights[None, :]
    denom = weights.sum(axis=1)
    aggregated_mask = (denom > 0).astype(np.uint8)
    denom_safe = np.where(denom > 0, denom, 1.0).astype(np.float32)

    if values.ndim == 2:
        numerator = (values * weights).sum(axis=1)
        aggregated = (numerator / denom_safe).astype(np.float32)
    elif values.ndim == 3:
        numerator = (values * weights[:, :, None]).sum(axis=1)
        aggregated = (numerator / denom_safe[:, None]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported ndim={values.ndim} for weighted aggregation.")

    aggregated = np.where(aggregated_mask[:, None], aggregated, 0.0) if values.ndim == 3 else np.where(aggregated_mask, aggregated, 0.0)
    return aggregated.astype(np.float32), aggregated_mask


def aggregate_bundle_to_regular_grid(
    bundle: FormalWindowBundle,
    patch_meta: pd.DataFrame,
    *,
    grid_height: int = 10,
    grid_width: int = 10,
) -> tuple[SpatialGridBundle, pd.DataFrame]:
    """把 307 patch bundle 聚合成固定 10x10 规则网格。"""

    node_meta, patch_to_node = build_regular_grid_meta(
        patch_meta,
        grid_height=int(grid_height),
        grid_width=int(grid_width),
    )
    num_nodes = int(grid_height) * int(grid_width)
    num_steps = bundle.target.shape[0]
    internal_dim = bundle.internal.shape[-1]
    blast_v3_dim = bundle.blast_v3.shape[-1]

    aggregated_internal = np.zeros((num_steps, num_nodes, internal_dim), dtype=np.float32)
    aggregated_blast_v3 = np.zeros((num_steps, num_nodes, blast_v3_dim), dtype=np.float32)
    aggregated_target = np.zeros((num_steps, num_nodes), dtype=np.float32)
    aggregated_input_mask = np.zeros((num_steps, num_nodes), dtype=np.uint8)
    aggregated_target_mask = np.zeros((num_steps, num_nodes), dtype=np.uint8)

    patch_weights_all = patch_meta["num_points"].fillna(1.0).to_numpy(dtype=np.float32)

    for node_index in range(num_nodes):
        member_idx = np.where(patch_to_node == node_index)[0]
        if member_idx.size == 0:
            continue
        patch_weights = patch_weights_all[member_idx]

        internal_values, internal_mask = _weighted_patch_aggregate(
            bundle.internal[:, member_idx, :],
            bundle.input_mask[:, member_idx],
            patch_weights,
        )
        blast_v3_values, _ = _weighted_patch_aggregate(
            bundle.blast_v3[:, member_idx, :],
            bundle.input_mask[:, member_idx],
            patch_weights,
        )
        target_values, target_mask = _weighted_patch_aggregate(
            bundle.target[:, member_idx],
            bundle.target_mask[:, member_idx],
            patch_weights,
        )

        aggregated_internal[:, node_index, :] = internal_values
        aggregated_blast_v3[:, node_index, :] = blast_v3_values
        aggregated_target[:, node_index] = target_values
        aggregated_input_mask[:, node_index] = internal_mask
        aggregated_target_mask[:, node_index] = target_mask

    adjacency = build_dense_grid_adjacency(int(grid_height), int(grid_width), eight_neighbor=True)
    edge_index = adjacency_to_edge_index(adjacency)
    grid_bundle = SpatialGridBundle(
        timestamps=bundle.timestamps,
        node_ids=node_meta["node_id"].astype(str).to_numpy(),
        internal_feature_names=list(bundle.internal_feature_names),
        weather_feature_names=list(bundle.weather_feature_names),
        blast_v2_feature_names=list(bundle.blast_v2_feature_names),
        blast_v3_feature_names=list(bundle.blast_v3_feature_names),
        internal=aggregated_internal,
        weather=bundle.weather.astype(np.float32),
        blast_v2=bundle.blast_v2.astype(np.float32),
        blast_v3=aggregated_blast_v3,
        target=aggregated_target,
        input_mask=aggregated_input_mask,
        target_mask=aggregated_target_mask,
        adjacency=adjacency.astype(np.float32),
        edge_index=edge_index.astype(np.int64),
    )
    return grid_bundle, node_meta


def save_spatial_grid_bundle(
    path: Path,
    bundle: SpatialGridBundle,
) -> None:
    """把规则网格 bundle 落盘为 npz。"""

    np.savez_compressed(
        path,
        timestamps=bundle.timestamps.astype(str),
        node_ids=bundle.node_ids.astype(str),
        internal_feature_names=np.asarray(bundle.internal_feature_names, dtype=object),
        weather_feature_names=np.asarray(bundle.weather_feature_names, dtype=object),
        blast_v2_feature_names=np.asarray(bundle.blast_v2_feature_names, dtype=object),
        blast_v3_feature_names=np.asarray(bundle.blast_v3_feature_names, dtype=object),
        internal=bundle.internal.astype(np.float32),
        weather=bundle.weather.astype(np.float32),
        blast_v2=bundle.blast_v2.astype(np.float32),
        blast_v3=bundle.blast_v3.astype(np.float32),
        target=bundle.target.astype(np.float32),
        input_mask=bundle.input_mask.astype(np.uint8),
        target_mask=bundle.target_mask.astype(np.uint8),
        adjacency=bundle.adjacency.astype(np.float32),
        edge_index=bundle.edge_index.astype(np.int64),
    )


def load_spatial_grid_bundle(path: Path) -> SpatialGridBundle:
    """从 npz 读取规则网格 bundle。"""

    data = np.load(path, allow_pickle=True)
    return SpatialGridBundle(
        timestamps=pd.to_datetime(data["timestamps"].astype(str)).to_numpy(),
        node_ids=data["node_ids"].astype(str),
        internal_feature_names=[str(item) for item in data["internal_feature_names"].tolist()],
        weather_feature_names=[str(item) for item in data["weather_feature_names"].tolist()],
        blast_v2_feature_names=[str(item) for item in data["blast_v2_feature_names"].tolist()],
        blast_v3_feature_names=[str(item) for item in data["blast_v3_feature_names"].tolist()],
        internal=data["internal"].astype(np.float32),
        weather=data["weather"].astype(np.float32),
        blast_v2=data["blast_v2"].astype(np.float32),
        blast_v3=data["blast_v3"].astype(np.float32),
        target=data["target"].astype(np.float32),
        input_mask=data["input_mask"].astype(np.uint8),
        target_mask=data["target_mask"].astype(np.uint8),
        adjacency=data["adjacency"].astype(np.float32),
        edge_index=data["edge_index"].astype(np.int64),
    )


def build_spatial_grid_dataset_from_ps10(
    dataset_dir: Path,
    *,
    grid_height: int = 10,
    grid_width: int = 10,
) -> tuple[SpatialGridBundle, pd.DataFrame, dict[str, Any]]:
    """从冻结的 ps10 patch bundle 直接生成规则网格数据。"""

    bundle, patch_meta, missing_report = load_formal_window_bundle(
        dataset_dir,
        missing_hour_policy="drop_global_missing",
    )
    grid_bundle, node_meta = aggregate_bundle_to_regular_grid(
        bundle,
        patch_meta,
        grid_height=int(grid_height),
        grid_width=int(grid_width),
    )
    report = {
        "source_patch_count": int(bundle.target.shape[1]),
        "effective_steps": int(bundle.target.shape[0]),
        "grid_height": int(grid_height),
        "grid_width": int(grid_width),
        "grid_node_count": int(grid_height) * int(grid_width),
        "nonempty_node_count": int((node_meta["is_empty_node"] == 0).sum()),
        "empty_node_count": int((node_meta["is_empty_node"] == 1).sum()),
        "missing_hour_policy": str(missing_report["missing_hour_policy"]),
    }
    return grid_bundle, node_meta, report
