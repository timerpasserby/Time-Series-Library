# paper_dataset_figures_v1

## 数据概况
- 时间范围：`2024-06-01 00:00` 至 `2024-06-30 23:00`
- 有效时间步：`712`
- 活动 patch 数：`307`
- 降雨小时数：`120`
- 爆破小时数：`57`

## 输出图件
- `fig_paper_internal_monitoring_overview.png / .pdf`：10 个内部监测指标的时序总览图。
- `fig_paper_weather_blast_overview.png / .pdf`：4 个天气指标与 7 个爆破指标的时序总览图。
- `fig_paper_internal_external_coupling_heatmap.png / .pdf`：内部监测指标与天气/爆破因子的 Spearman 相关系数热力图。
- `fig_paper_patch_spatial_sensitivity_maps.png / .pdf`：patch 级空间敏感性图，展示位移水平、活跃度以及对降雨/爆破的响应相关性。

## 伴随表格
- `internal_external_spearman_matrix.csv`：论文热力图对应的相关系数矩阵。
- `patch_spatial_sensitivity_metrics.csv`：patch 空间敏感性指标表。

## 复现命令
- `/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python /Users/dc/Z研究生/time_series/Time-Series-Library/scripts/slopemine_v2/generate_paper_dataset_figures_v1.py`