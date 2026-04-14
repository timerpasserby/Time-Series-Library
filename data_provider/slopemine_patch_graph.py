"""构造 ps10 307-patch 图结构，供 SpatialGNN-TimeFilter 使用。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_provider.slopemine_formal import FormalWindowBundle, load_formal_window_bundle
from data_provider.slopemine_grid import adjacency_to_edge_index


def build_patch_8nn_adjacency(patch_meta: pd.DataFrame) -> np.ndarray:
    """按 patch_x_idx / patch_y_idx 构造 8 邻接静态 patch 图。"""

    coords = patch_meta[["patch_x_idx", "patch_y_idx"]].to_numpy(dtype=np.int32)
    num_nodes = coords.shape[0]
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for src in range(num_nodes):
        dx = np.abs(coords[:, 0] - coords[src, 0])
        dy = np.abs(coords[:, 1] - coords[src, 1])
        neighbor_mask = (dx <= 1) & (dy <= 1) & ~((dx == 0) & (dy == 0))
        adjacency[src, neighbor_mask] = 1.0

    return adjacency


def load_patch_graph_bundle(
    dataset_dir: Path,
) -> tuple[FormalWindowBundle, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """读取正式 ps10 bundle，并构造 307-patch 邻接图。"""

    bundle, patch_meta, missing_report = load_formal_window_bundle(
        dataset_dir,
        missing_hour_policy="drop_global_missing",
    )
    adjacency = build_patch_8nn_adjacency(patch_meta)
    edge_index = adjacency_to_edge_index(adjacency)
    report = {
        "patch_count": int(bundle.target.shape[1]),
        "effective_steps": int(bundle.target.shape[0]),
        "edge_count_directed": int(edge_index.shape[1]),
        "avg_degree": float(adjacency.sum(axis=1).mean()),
        "missing_hour_policy": str(missing_report["missing_hour_policy"]),
    }
    return bundle, patch_meta, adjacency.astype(np.float32), edge_index.astype(np.int64), report
