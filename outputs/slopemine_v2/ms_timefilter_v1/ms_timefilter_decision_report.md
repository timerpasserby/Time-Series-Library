# ms_timefilter_decision_report

## Main Table

| experiment_id | branch_name | model_name | val_mae | test_mae | blast_test_mae |
| --- | --- | --- | ---: | ---: | ---: |
| A2 | main | MS-TimeFilter | 0.061450 | 0.127768 | 0.051461 |
| A3 | main | MS-TimeFilter | 0.078841 | 0.141597 | 0.058432 |
| A4 | blast_opt | MS-TimeFilter | 0.063480 | 0.136655 | 0.054253 |
| A4 | main | MS-TimeFilter | 0.073161 | 0.132219 | 0.050830 |
| C1 | blast_opt | MS-TimeFilter | 0.058301 | 0.111800 | 0.042384 |
| C1 | main | MS-TimeFilter | 0.055784 | 0.120769 | 0.044829 |
| C2 | blast_opt | MS-TimeFilter | 0.055841 | 0.120503 | 0.047920 |
| C2 | main | MS-TimeFilter | 0.050130 | 0.108187 | 0.041348 |
| C3 | blast_opt | MS-TimeFilter | 0.052127 | 0.110418 | 0.044074 |
| C3 | main | MS-TimeFilter | 0.052966 | 0.112980 | 0.042585 |
| baseline_lstm_a4_96 | reference | LSTM | 0.051959 | 0.097959 | 0.033767 |
| baseline_naive_weather_concat | control | TimeFilter | 0.044932 | 0.177935 | 0.083312 |
| baseline_raw_timefilter_a4_96 | reference | TimeFilter | 0.055817 | 0.174941 | 0.092875 |

## Answers

1. A2 相比 naive weather concat 是否更优：是。
2. A4 是否优于 A3：是。
3. C1 是否优于 A4 和原始 TimeFilter：是。
4. C2 是否在 blast_hours 子集上体现额外增益：是。
5. C3 是否提升了物理一致性或曲线合理性：请结合 `pir_dynamics_C3.png` 与 `typical_event_windows_C2_vs_C3.png` 人工复核。
6. 完整 MS-TimeFilter 是否逼近或超过当前最强 LSTM：否。
7. 若未超过 LSTM，优先从 backbone、图构建、事件路由和物理损失四个方向解释；最终以当前总表数值为准。
