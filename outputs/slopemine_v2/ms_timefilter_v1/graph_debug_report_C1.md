# graph_debug_report_C1

- 调试模型：`C1_main`。
- 空间先验邻接最大度：`4`。
- 时间先验邻接最大度：`2`。
- learned top-k：`8`。
- time_block_count：`12`。
- 显示说明：原始矩阵保持不变，出图时仅对正值边权额外做 `+0.5` 的显示偏移，用于让稀疏结构更明显。

## Heatmaps

- `A_prior`: `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/graph_A_prior_C1.png`
- `A_learned`: `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/graph_A_learned_C1.png`
- `A_final`: `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/graph_A_final_C1.png`
- `graph_debug_C1.npz`: `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/graph_debug_C1.npz`

## Matrix Stats

### A_prior

- shape：`3684 x 3684`。
- nonzero_ratio：`0.001343`（nonzero_count=`18226`）。
- 正值统计：`p50=0.200000`、`p90=0.250000`、`p99=0.333333`、`max=1.000000`。
- 显示上限：正值 `p99.5=0.500000`。

### A_learned

- shape：`3684 x 3684`。
- nonzero_ratio：`0.002172`（nonzero_count=`29472`）。
- 正值统计：`p50=0.124204`、`p90=0.130577`、`p99=0.138453`、`max=0.152272`。
- 显示上限：正值 `p99.5=0.140460`。

### A_final

- shape：`3684 x 3684`。
- nonzero_ratio：`0.003513`（nonzero_count=`47678`）。
- 正值统计：`p50=0.076226`、`p90=0.089661`、`p99=0.104312`、`max=0.141798`。
- 显示上限：正值 `p99.5=0.108594`。

