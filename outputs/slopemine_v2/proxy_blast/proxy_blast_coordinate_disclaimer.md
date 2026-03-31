# proxy_blast_coordinate_disclaimer

- 本目录中的 `proxy_blast_ledger_*` 和 `patch_blast_features_v3_ps10_proxy*` 基于代理推断坐标生成。
- 这些坐标不是实时真实爆破作业坐标，不代表任何真实施工台账、调度记录或现场定位结果。
- 它们仅用于 V3 空间衰减方法开发、参数敏感性分析和稳定性实验。
- 任何涉及真实爆破作业复盘、生产调度、安全追溯或工程决策的场景，都不能使用这些代理坐标替代真实数据。
- 文件中的 `coordinate_source` 固定为 `proxy_inferred`，`is_real_coordinate` 固定为 `0`，用于避免误用。