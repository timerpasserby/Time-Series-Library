# experiment_log_A1_A4

## Run Context

- 正式 split 文件：`outputs/slopemine_v2/experiment_plan_v1/experiment_split_plan_v1.csv`。
- 选定方案：`split_cand_01`。
- 正式窗口配置：`seq_len=24`、`pred_len=12`，来源：`outputs/slopemine_v2/tuning_v1/best_window_config_v1.json`。
- 缺失小时策略：`drop_global_missing`，显式剔除 `8` 个全局缺失小时。
- 训练设备：`cpu`。
- 统一训练配置：`epochs=8`、`batch_size=8`、`lr=0.001`、`weight_decay=1e-05`。

## Best A4 Run

- 最优 A4 组合：`LSTM`。
- `val_mae=4.410412e-02`，`test_mae=1.006999e-01`。
- checkpoint：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/formal_round1/checkpoints/A4_LSTM_best.pt`。

## Outputs

- 结果总表：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/formal_round1/results_main_A1_A4.csv`。
- 子集结果：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/formal_round1/results_subsets_A1_A4.csv`。
- 图表目录：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/formal_round1/figures`。
