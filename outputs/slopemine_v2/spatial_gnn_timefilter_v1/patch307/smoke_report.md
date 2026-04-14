# smoke_report

- graph_source: `patch`
- device: `cpu`
- input tensor: `(1, 712, 307, 10)`
- node mask: `(1, 712, 307)`
- spatial embeddings: `(1, 712, 307, 64)`
- scalar series before TimeFilter: `(1, 712, 307)`
- prediction output: `(1, 12, 307)`
- adjacency: `(307, 307)`
- edge_index: `(2, 3898)`
- nonempty nodes: `307` / `307`

## Model Config

- gnn_type: `graphsage`
- hidden_dim: `64`
- patch_len: `89`
- pred_len: `12`
- timefilter_tokens: `2456`

## Graph Summary

- node_count: `307`
- directed_edges: `3898`
- avg_degree: `12.70`

## Notes

- 当前烟测采用全长 `T=712` 作为 encoder 序列。
- 当 `graph_source=patch` 时，TimeFilter 输入通道数固定为当前正式数据的 `C=307` 个 patch。
- 为避免原始 TimeFilter 在 `307 x 712` 上出现过大的 `O(L^2)` 图学习开销，默认把时间 patch 长度提高到 `89`，此时每个 patch 通道对应 `8` 个时间 token。
- 该脚本只验证“逐时间步共享 Spatial GNN + TimeFilter”前向链路，不触发训练。
