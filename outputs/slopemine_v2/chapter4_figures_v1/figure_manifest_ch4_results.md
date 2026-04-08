# figure_manifest_ch4_results

| 文件名 | 数据来源 | 图意 | 建议插入章节位置 |
| --- | --- | --- | --- |
| fig_ch4_main_performance_compare.png / fig_ch4_main_performance_compare.pdf | results_main_all.csv + results_subsets_all.csv | 比较原始 TimeFilter、C1、C2、C3 与 LSTM 在 test_MAE 与 blast_test_MAE 上的总体表现，突出 C2 main 作为当前主模型。 | 第四章 4.4 主结果对比 |
| fig_ch4_subset_compare_c2_vs_lstm.png / fig_ch4_subset_compare_c2_vs_lstm.pdf | results_subsets_all.csv + c2_vs_lstm_subset_compare.csv | 比较 C2 与 LSTM 在 val/test 的爆破、降雨、非事件子集上的 MAE，并标注差值。 | 第四章 4.4 子集误差分析 |
| fig_ch4_event_windows_c2_vs_lstm.png / fig_ch4_event_windows_c2_vs_lstm.pdf | c2_vs_lstm_event_windows.md + prediction bundles + formal 96/12 manifest | 展示两个典型爆破事件在 48h 有效时间步上的连续预测曲线；重叠时刻采用“最早窗口预测”规则拼接，并标记 blast event 时刻。 | 第四章 4.5 典型事件窗口分析 |
| fig_ch4_event_windows_c2_vs_lstm_short12h.png / fig_ch4_event_windows_c2_vs_lstm_short12h.pdf | c2_vs_lstm_event_windows.md + prediction bundles + formal 96/12 manifest | 保留原始 12h 单窗事件预测图，作为第四章长预测主图的备份版本。 | 附录 / 论文图件备份 |
| fig_ch4_patch_mae_heatmap_c2.png / fig_ch4_patch_mae_heatmap_c2.pdf | prediction bundles + patch_meta_v2_ps10_bg_excluded.csv | C2 main patch 级 test MAE。背景区未参与主训练，因此图中保持浅灰空白背景。 | 第四章 4.5 patch 级空间误差分析 |
| fig_ch4_patch_mae_heatmap_lstm.png / fig_ch4_patch_mae_heatmap_lstm.pdf | prediction bundles + patch_meta_v2_ps10_bg_excluded.csv | LSTM patch 级 test MAE。背景区未参与主训练，因此图中保持浅灰空白背景。 | 第四章 4.5 patch 级空间误差分析 |
| fig_ch4_patch_mae_heatmap_diff_c2_minus_lstm.png / fig_ch4_patch_mae_heatmap_diff_c2_minus_lstm.pdf | prediction bundles + patch_meta_v2_ps10_bg_excluded.csv | C2 - LSTM patch 级 MAE 差值。背景区未参与主训练，因此图中保持浅灰空白背景。 | 第四章 4.5 patch 级空间误差分析 |
| fig_ch4_pggc_graph_compare.png / fig_ch4_pggc_graph_compare.pdf | graph_A_prior_C1.png + graph_A_learned_C1.png + graph_A_final_C1.png | 三联图展示 PGGC 的先验图、学习图与融合图；单图渲染时对正值边权额外做了 `+0.5` 显示偏移，以增强稀疏结构可见性。 | 第四章 4.6 PGGC 结构解释 |
| fig_ch4_eddr_routing_compare.png / fig_ch4_eddr_routing_compare.pdf | C2_main_val_eval.npz + C2_main_test_eval.npz (gate_mean) | 比较 blast_hours 与 non_event_hours 下三类专家权重分布，用于检查 EDDR 是否发生了事件驱动路由变化。 | 第四章 4.6 EDDR 路由分析 |
| fig_ch4_prediction_smoothness_compare.png / fig_ch4_prediction_smoothness_compare.pdf | baseline_raw_timefilter_a4_96 + C2 main + baseline_lstm_a4_96 prediction bundles | 比较 raw TimeFilter、C2 与 LSTM 在代表窗口中的平滑性与突发工况响应，用于说明 C2 相比原始骨干更合理。 | 第四章 4.6 物理合理性辅助分析 |
| fig_physical_curve_comparison.png / fig_physical_curve_comparison.pdf | baseline_raw_timefilter_a4_96 + C2 main + C3 main prediction bundles + window_manifest_ms_timefilter_ps10_seq96_pred12.csv | 展示 C2 与 C3(PIR) 在代表性爆后响应窗口中的 12h 单窗预测与恢复段放大，用于强调 PIR 对爆后响应稳定性的局部可视化作用。 | 第四章 4.6 PIR 爆后响应示意（fig:physical_curve_comparison） |
| fig_spatial_error_heatmap.png / fig_spatial_error_heatmap.pdf | C3_main_test_eval.npz + point_meta_v2.csv + patch_meta_v2_ps10_bg_excluded.csv + engineering_zone_polygons_v2.json | 在监测点散点上渲染 C3(PIR) 的 patch-derived test MAE，并叠加冻结工程分区边界，突出高误差团块仍主要被限制在 crest 区内。 | 第四章 4.6 PIR 空间误差分布（fig:spatial_error_heatmap） |
| fig_ch4_weather_eddr_ablation_group.png / fig_ch4_weather_eddr_ablation_group.pdf | ms_timefilter_v1 prediction bundles + window_manifest_ms_timefilter_ps10_seq96_pred12.csv | 纵向组图展示天气分支与 EDDR 的核心消融窗口：上图突出强降雨下的伪波动抑制，下图改为以 2024-06-25 00:00 为主冲击点的连续预测视角，历史段平滑衔接，比较 C2 的滞后与 C5 的同步峰值响应。 | 第四章 4.6 机制消融主图 |
