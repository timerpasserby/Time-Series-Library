# dataset_manifest_v1

- 本次正式主实验固定使用 `patch_size=10`。
- 正式训练 patch 数固定为 307，稳定背景区 `stable_background` 默认排除出主训练。
- 正式 split 基于 712 个有效时间步，不直接复用旧的 720 小时整轴切分。

| file | role | row_count | time_range | patch_count | zone_count | stable_background_excluded | note |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| dataset/slopemine_v2/point_meta_v2.csv | 冻结后的点位工程分区元数据 | 19566 | - | - | 5 | 否 | 分区规则已冻结，可重复生成 zone_id_v2。 |
| dataset/slopemine_v2/patch_meta_v2_ps10.csv | 完整 patch 元数据（含 background patch） | 308 | - | 308 | 5 | 否 | 完整 patch 资产，用于回溯和可视化。 |
| dataset/slopemine_v2/patch_meta_v2_ps10_bg_excluded.csv | 正式主实验 patch 清单 | 307 | - | 307 | 4 | 是 | 正式训练 patch 数固定为 307。 |
| dataset/slopemine_v2/patch_series_v2_ps10.csv | patch 级内部时序主表 | 221760 | 2024-06-01 00:00 ~ 2024-06-30 23:00 | 308 | 5 | 否 | 包含 720 小时整轴和 8 个显式全局缺失小时。 |
| dataset/slopemine/weather.csv | 天气外生输入 | 712 | 2024-06-01 00:00 ~ 2024-06-30 23:00 | - | - | 不适用 | 有效天气时间步共 712 小时。 |
| dataset/slopemine_v2/blast_ledger_v3_input.csv | 真实爆破事件输入台账 | 70 | 2024-06-01 10:23 ~ 2024-06-29 11:21 | - | 4 | 不适用 | 共 70 条事件记录，折算为 57 个爆破小时。 |
| dataset/slopemine_v2/patch_blast_features_v3_ps10.csv | 真实坐标 blast V3 patch 级特征 | 221760 | 2024-06-01 00:00 ~ 2024-06-30 23:00 | 308 | 5 | 否 | A4 正式主实验使用这套 real-coordinate V3 特征。 |
| dataset/slopemine_v2/patch_tensor_base_v2_ps10.npz | 正式训练基础张量 | 720 | 2024-06-01 00:00 ~ 2024-06-30 23:00 | 307 | 4 | 是 | 基础张量 shape: internal=(720, 307, 10), weather=(720, 4), blast_v2=(720, 7), blast_v3=(720, 307, 4) |
| dataset/slopemine_v2/inferred_blast_ledger_internal_v1.csv | 内部反演疑似爆破事件参考表 | 19 | 2024-06-02 03:00 ~ 2024-06-27 17:00 | 13 | 3 | 是 | 仅用于第四章补充说明，不作为真实爆破坐标训练输入。 |
| outputs/slopemine_v2/experiment_plan_v1/experiment_split_plan_v1.csv | 正式 train/val/test 切分基准 | 3 | 2024-06-01 00:00 ~ 2024-06-30 23:00 | 307 | 4 | 是 | 正式实验应以此文件覆盖旧 window_manifest 默认切分。 |