# window_build_report_ms_timefilter

- 本轮窗口严格依据 `experiment_split_plan_v1.csv` 构造，decoder horizon 必须完整落入 split。
- 固定窗口配置：`seq_len=96`、`pred_len=12`。
- 缺失小时处理策略：`drop_global_missing`。
- 整轴 720 小时中显式剔除 `8` 个全局缺失小时，正式有效时间步为 `712`。

| split | effective_steps | sample_count | blast_hours | rain_hours |
| --- | ---: | ---: | ---: | ---: |
| train | 456 | 349 | 41 | 78 |
| val | 128 | 117 | 8 | 18 |
| test | 128 | 117 | 8 | 24 |
