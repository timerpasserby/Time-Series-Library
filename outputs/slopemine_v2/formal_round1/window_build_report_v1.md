# window_build_report_v1

- 正式窗口严格依据 `experiment_split_plan_v1.csv` 构造，样本归属以 decoder horizon 完全落入对应 split 为准。
- 当前正式窗口配置：`seq_len=24`、`pred_len=12`，来源：`outputs/slopemine_v2/tuning_v1/best_window_config_v1.json`。
- 全局缺失小时处理策略：`drop_global_missing`。
- 原始整轴共 `720` 小时，其中显式全局缺失 `8` 小时，正式窗口仅基于 `712` 个有效时间步构造。
- 全局缺失小时时间戳：`2024-06-06T05:00:00, 2024-06-06T06:00:00, 2024-06-06T07:00:00, 2024-06-06T08:00:00, 2024-06-14T00:00:00, 2024-06-14T01:00:00, 2024-06-14T05:00:00, 2024-06-20T08:00:00`。

## Split Summary

| split | effective_steps | sample_count | encoder_crosses_split_start | blast_hours | rain_hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 456 | 421 | 0 | 41 | 78 |
| val | 128 | 117 | 24 | 8 | 18 |
| test | 128 | 117 | 24 | 8 | 24 |

## Notes

- 本轮没有复用旧的 `window_manifest_v2_ps10.csv` 默认切分。
- 由于 split 基于有效时间步，val/test 前几个窗口会使用前一段历史作为 encoder 上下文；这属于只用过去信息的滚动预测，不会读取未来标签。
