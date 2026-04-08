# c2_vs_lstm_event_windows

- 主对比口径固定为 `C2 main` vs `baseline_lstm_a4_96_reference`。
- split 与窗口固定为 `split_cand_01`、`seq_len=96`、`pred_len=12`。
- blast 窗口样本数：`55`；其中 `C2 main` 在 `3` 个 blast 窗口上的样本级 MAE 优于 LSTM。

## Subset Summary

- `blast_hours`：C2 MAE=`0.041348`，LSTM MAE=`0.033767`，差值=`0.007580`。
- `rain_hours`：C2 MAE=`0.122154`，LSTM MAE=`0.111093`，差值=`0.011061`。
- `non_event_hours`：C2 MAE=`0.107749`，LSTM MAE=`0.097509`，差值=`0.010240`。

## Typical Blast Windows

| sample_id | patch_id | decoder_start | decoder_end | blast_score | C2 window MAE | LSTM window MAE | C2 event-patch MAE | LSTM event-patch MAE |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 500 | ps10_v2_p256 | 2024-06-26T04:00:00 | 2024-06-26T15:00:00 | 1088.20 | 0.039780 | 0.026602 | 0.037580 | 0.014707 |
| 526 | ps10_v2_p155 | 2024-06-27T06:00:00 | 2024-06-27T17:00:00 | 1006.60 | 0.362404 | 0.364723 | 0.383154 | 0.375304 |

## Readout

- 优势：C2 在少数 blast 窗口里能够优于 LSTM，而且相对原始 TimeFilter，局部空间误差分布已经明显更合理。
- 不足：相对 LSTM，C2 当前总体 MAE 与三类子集 MAE 仍然偏高，说明 `PGGC + EDDR` 还没有把机制增益完全转化成更强的整体回归精度。
- 典型窗口图：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/c2_vs_lstm_typical_blast_windows.png`。
- patch 级热力图：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/c2_vs_lstm_patch_mae_heatmap.png`。
