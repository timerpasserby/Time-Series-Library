#!/usr/bin/env python3
"""对 SpatialGNN + TimeFilter 做一次全长前向烟测。
默认走当前正式数据的 307-patch 图，不再重采样为 100 节点规则网格。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.slopemine_grid import build_spatial_grid_dataset_from_ps10, load_spatial_grid_bundle  # noqa: E402
from data_provider.slopemine_patch_graph import load_patch_graph_bundle  # noqa: E402
from models.spatial_gnn_timefilter import SpatialGNNTimeFilter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the regular-grid SpatialGNN + TimeFilter model.")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "dataset" / "slopemine_v2")
    parser.add_argument("--graph-source", type=str, default="patch", choices=["patch", "grid"])
    parser.add_argument("--grid-height", type=int, default=10)
    parser.add_argument("--grid-width", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patch-len", type=int, default=89)
    parser.add_argument("--gnn-type", type=str, default="graphsage", choices=["graphsage", "gcn"])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    lowered = str(device_arg).lower()
    if lowered == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(lowered)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if args.graph_source == "patch":
        bundle, patch_meta, adjacency, edge_index, graph_report = load_patch_graph_bundle(dataset_dir)
        node_ids = bundle.patch_ids
        input_tensor = bundle.internal
        input_mask = bundle.input_mask.astype(np.float32)
        output_dir_name = "patch307"
    else:
        stem = f"spatial_grid_tensor_v1_ps10_g{int(args.grid_height)}x{int(args.grid_width)}.npz"
        grid_path = dataset_dir / stem
        if grid_path.exists():
            bundle = load_spatial_grid_bundle(grid_path)
        else:
            bundle, _node_meta, _report = build_spatial_grid_dataset_from_ps10(
                dataset_dir,
                grid_height=int(args.grid_height),
                grid_width=int(args.grid_width),
            )
        adjacency = bundle.adjacency
        edge_index = bundle.edge_index
        graph_report = {
            "patch_count": int(bundle.internal.shape[1]),
            "effective_steps": int(bundle.internal.shape[0]),
            "edge_count_directed": int(edge_index.shape[1]),
            "avg_degree": float(adjacency.sum(axis=1).mean()),
            "missing_hour_policy": "drop_global_missing",
        }
        node_ids = bundle.node_ids
        input_tensor = bundle.internal
        input_mask = bundle.input_mask.astype(np.float32)
        output_dir_name = f"grid_g{int(args.grid_height)}x{int(args.grid_width)}"

    device = resolve_device(args.device)
    model = SpatialGNNTimeFilter(
        input_dim=input_tensor.shape[-1],
        num_nodes=input_tensor.shape[1],
        seq_len=input_tensor.shape[0],
        pred_len=int(args.pred_len),
        adjacency=torch.from_numpy(adjacency),
        model_cfg={
            "d_model": int(args.hidden_dim),
            "d_ff": int(args.hidden_dim) * 2,
            "e_layers": 2,
            "dropout": 0.1,
            "n_heads": 4,
            "alpha": 0.1,
            "top_p": 0.5,
            "patch_len": int(args.patch_len),
            "gnn_type": str(args.gnn_type),
            "gnn_layers": 2,
            "gnn_dropout": 0.1,
        },
    ).to(device)

    x = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
    node_mask = torch.from_numpy(input_mask).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x, node_mask=node_mask, return_aux=True)

    out_dir = REPO_ROOT / "outputs" / "slopemine_v2" / "spatial_gnn_timefilter_v1" / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "smoke_report.md"

    pred = outputs["pred"].detach().cpu().numpy()
    spatial_embeddings = outputs["aux"]["spatial_embeddings"].detach().cpu().numpy()
    scalar_series = outputs["aux"]["scalar_series"].detach().cpu().numpy()

    lines = [
        "# smoke_report",
        "",
        f"- graph_source: `{args.graph_source}`",
        f"- device: `{device}`",
        f"- input tensor: `{tuple(x.shape)}`",
        f"- node mask: `{tuple(node_mask.shape)}`",
        f"- spatial embeddings: `{tuple(spatial_embeddings.shape)}`",
        f"- scalar series before TimeFilter: `{tuple(scalar_series.shape)}`",
        f"- prediction output: `{tuple(pred.shape)}`",
        f"- adjacency: `{tuple(adjacency.shape)}`",
        f"- edge_index: `{tuple(edge_index.shape)}`",
        f"- nonempty nodes: `{int(input_mask.any(axis=0).sum())}` / `{input_mask.shape[1]}`",
        "",
        "## Model Config",
        "",
        f"- gnn_type: `{args.gnn_type}`",
        f"- hidden_dim: `{int(args.hidden_dim)}`",
        f"- patch_len: `{int(args.patch_len)}`",
        f"- pred_len: `{int(args.pred_len)}`",
        f"- timefilter_tokens: `{int(model.num_nodes * model.seq_len // model.patch_len)}`",
        "",
        "## Graph Summary",
        "",
        f"- node_count: `{len(node_ids)}`",
        f"- directed_edges: `{graph_report['edge_count_directed']}`",
        f"- avg_degree: `{graph_report['avg_degree']:.2f}`",
        "",
        "## Notes",
        "",
        "- 当前烟测采用全长 `T=712` 作为 encoder 序列。",
        "- 当 `graph_source=patch` 时，TimeFilter 输入通道数固定为当前正式数据的 `C=307` 个 patch。",
        "- 为避免原始 TimeFilter 在 `307 x 712` 上出现过大的 `O(L^2)` 图学习开销，默认把时间 patch 长度提高到 `89`，此时每个 patch 通道对应 `8` 个时间 token。",
        "- 该脚本只验证“逐时间步共享 Spatial GNN + TimeFilter”前向链路，不触发训练。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "input_shape": list(x.shape),
                "spatial_embedding_shape": list(spatial_embeddings.shape),
                "scalar_series_shape": list(scalar_series.shape),
                "pred_shape": list(pred.shape),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
