#!/usr/bin/env python3
"""基于冻结的 ps10 patch 数据生成 10x10 规则网格数据包。
相关文件：patch_tensor_base_v2_ps10.npz、patch_meta_v2_ps10_bg_excluded.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.slopemine_grid import (  # noqa: E402
    build_spatial_grid_dataset_from_ps10,
    save_spatial_grid_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a regular 10x10 spatial-grid dataset from ps10 patch tensors.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "dataset" / "slopemine_v2")
    parser.add_argument("--grid-height", type=int, default=10)
    parser.add_argument("--grid-width", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    bundle, node_meta, report = build_spatial_grid_dataset_from_ps10(
        dataset_dir,
        grid_height=int(args.grid_height),
        grid_width=int(args.grid_width),
    )

    stem = f"spatial_grid_tensor_v1_ps10_g{int(args.grid_height)}x{int(args.grid_width)}"
    npz_path = dataset_dir / f"{stem}.npz"
    csv_path = dataset_dir / f"spatial_grid_node_meta_v1_ps10_g{int(args.grid_height)}x{int(args.grid_width)}.csv"
    report_path = dataset_dir / f"spatial_grid_build_report_v1_ps10_g{int(args.grid_height)}x{int(args.grid_width)}.md"

    save_spatial_grid_bundle(npz_path, bundle)
    node_meta.to_csv(csv_path, index=False)

    lines = [
        "# spatial_grid_build_report_v1",
        "",
        f"- 来源数据：`{dataset_dir / 'patch_tensor_base_v2_ps10.npz'}`",
        f"- 输入 patch 数：`{report['source_patch_count']}`",
        f"- 有效时间步：`{report['effective_steps']}`",
        f"- 规则网格：`{report['grid_height']} x {report['grid_width']}`，共 `N={report['grid_node_count']}` 个节点",
        f"- 非空节点数：`{report['nonempty_node_count']}`；空节点数：`{report['empty_node_count']}`",
        f"- 缺失小时策略：`{report['missing_hour_policy']}`",
        "",
        "## Output Files",
        "",
        f"- Grid tensor: `{npz_path}`",
        f"- Node meta: `{csv_path}`",
        "",
        "## Tensor Shapes",
        "",
        f"- internal: `{tuple(bundle.internal.shape)}`",
        f"- weather: `{tuple(bundle.weather.shape)}`",
        f"- blast_v2: `{tuple(bundle.blast_v2.shape)}`",
        f"- blast_v3: `{tuple(bundle.blast_v3.shape)}`",
        f"- target: `{tuple(bundle.target.shape)}`",
        f"- adjacency: `{tuple(bundle.adjacency.shape)}`",
        "",
        "## Notes",
        "",
        "- 当前实现按 patch centroid 把 307 个 ps10 patch 映射到固定 10x10 规则网格。",
        "- 节点特征与目标值均按 patch `num_points` 做 masked weighted mean 聚合。",
        "- 空节点保留在图中，以保证 TimeFilter 输入通道固定为 100。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"npz_path": str(npz_path), "csv_path": str(csv_path), "report_path": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
