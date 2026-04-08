# c2_interpretability_report

## PGGC

- 当前 PGGC 解释图来自已经落盘的 `C1` 图结构调试结果。
- 先验图约束保持：同区 4 邻接的空间边与同 patch 相邻时间块的时间边。
- 已记录的结构上界是：空间先验最大度 `4`、时间先验最大度 `2`、learned top-k=`8`。
- 图结构对照图：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/pggc_graph_comparison.png`。

## EDDR

- blast_hours 平均 gate：spatial=`0.2557`、temporal=`0.3881`、spatiotemporal=`0.3562`。
- non_event_hours 平均 gate：spatial=`0.2559`、temporal=`0.3865`、spatiotemporal=`0.3576`。
- blast 相对 non-event 的 gate 变化：spatial=`-0.0002`、temporal=`+0.0016`、spatiotemporal=`-0.0015`。
- blast 强度与 spatiotemporal gate 的样本级相关系数：`-0.2862`。
- 路由解释图：`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/eddr_routing_analysis.png`。

## Interpretation

- 当前样本级证据偏弱：blast 与 non-event 的 gate 均值差异非常小，说明 EDDR 已接入但尚未形成很强的工况分化路由。
- 从方向上看，blast_hours 的 temporal gate 略高于 non-event，但 spatiotemporal gate 没有同步增强，因此暂时不能写成“EDDR 已明显实现复杂工况动态调节”。
- 当前结果里，C2 的 blast_hours MAE 仍高于 LSTM，说明动态路由的潜在作用还没有转化成稳定的最终误差优势。
- 这一点与总指标一致：blast_hours 上 C2-LSTM 的 MAE 差值为 `+0.007580`。
