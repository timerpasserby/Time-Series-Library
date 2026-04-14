# smoke_report

- device: `cpu`
- input tensor: `(1, 712, 100, 10)`
- node mask: `(1, 712, 100)`
- spatial embeddings: `(1, 712, 100, 64)`
- scalar series before TimeFilter: `(1, 712, 100)`
- prediction output: `(1, 12, 100)`
- adjacency: `(100, 100)`
- edge_index: `(2, 684)`
- nonempty nodes: `62` / `100`

## Model Config

- gnn_type: `graphsage`
- hidden_dim: `64`
- patch_len: `8`
- pred_len: `12`

## Notes

- 当前烟测采用全长 `T=712` 作为 encoder 序列，因此 TimeFilter 输入通道数固定为 `C=100`，序列长度固定为 `L=712`。
- 该脚本只验证“规则网格聚合 + 逐时间步共享 Spatial GNN + TimeFilter”前向链路，不触发训练。
