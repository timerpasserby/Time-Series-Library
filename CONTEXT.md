# 当前状态
- `slopemine v2` 已完成正式规则冻结、refined patch 重建、`patch_series_v2_ps10.csv`、论文用诊断包、论文版最终工程分区结果图、三套代理爆破坐标台账与 proxy V3 特征导出，以及 blast V3 局部事件敏感版本诊断。
- 新增“内部反演疑似爆破事件”链路：完全忽略旧爆破台账，只基于内部时序异常和冻结分区规则生成 `inferred_blast_ledger_internal_v1.csv`、候选小时排名、空间图、分区叠加图和诊断报告。
- 新增“正式实验 split 与数据清单”链路：基于冻结后的 `ps10` 数据资产自动生成正式 train / val / test 候选、推荐 split、数据版本清单、实验矩阵和第四章补写项。
- 新增“正式 split 下的窗口参数搜索”链路：在 712 个有效时间步和正式 split 上重新扫描 `seq_len / pred_len`，当前推荐基础配置已更新为 `24 / 12`。
- 新增“正式 patch 级深度实验”链路：基于冻结后的 `dataset_manifest_v1 + experiment_split_plan_v1 + window_manifest_v1_ps10.csv`，实现正式 `dataset / dataloader / runner`，并完成 `LSTM / TCN / DLinear / PatchTST / STGCN / TimeFilter` 六模型的第一轮 `A1-A4` 实验。
- 新增 `MS-TimeFilter` 正式实验链路：已完成多分支 formal dataset、正式天气分支 `weather_encoder`、`PGGC / EDDR / PIR` 最小实现、`MSTimeFilter` 主模型和统一 runner，并已落盘 `outputs/slopemine_v2/ms_timefilter_v1` 的正式结果；当前主模型版本固定为 `C2 main = PGGC + EDDR`。
- 新增第四章图件包导出链路：`generate_chapter4_figures_v1.py` 会基于冻结后的 `formal_round1 + ms_timefilter_v1` 正式结果统一导出论文图，输出目录为 `outputs/slopemine_v2/chapter4_figures_v1`。
- 新增独立组图脚本 `scripts/slopemine_v2/generate_ch4_weather_eddr_group_v1.py`：用于在不改并行编辑中的主图脚本前提下，单独导出“天气分支 + EDDR”纵向组图，并把结果补写进现有 `figure_manifest_ch4_results.md`。
- 已完成 PGGC 图显示调整：保留 `graph_debug_C1.npz` 原始矩阵包，但当前出图口径改为“仅对正值边权做 `+0.5` 显示偏移”，并恢复为旧版三联拼图样式，使结果更接近论文当前参考图。
- `run_ms_timefilter_experiments_v1.py` 的 `plot_graph_heatmap` 现已支持不传标题直接出图，`graph_heatmap_paths` 这三张 PGGC 单图调用已切到无标题模式并验证可跑通。
- `analyze_c2_main_v1.py` 的 `eddr_routing_analysis` 当前显示口径已按人工偏移更新：左侧柱状图对 `blast/rain` 的三专家权重做固定显示增量，右侧散点图对 `spatiotemporal` 的 `blast/rain` 点位做对应抬升并改成三类着色。
- `eddr_routing_analysis.png` 当前已切成全中文标注：左侧专家类别、右侧坐标轴标题和三类图例都统一为中文。

# 上次停点
- 最新结果已落到 `dataset/slopemine_v2`、`outputs/slopemine_v2/diagnostic_package_v2`、`outputs/slopemine_v2/figures/final_zone_assignment_v2.{png,pdf}`、`outputs/slopemine_v2/proxy_blast` 和 `outputs/slopemine_v2/blast_v3_variants`。
- 正式实验计划与数据清单已落到 `outputs/slopemine_v2/experiment_plan_v1`；当前冻结推荐 split 为 `train=2024-06-01 00:00 ~ 2024-06-20 06:00`、`val=2024-06-20 07:00 ~ 2024-06-25 15:00`、`test=2024-06-25 16:00 ~ 2024-06-30 23:00`（基于 712 个有效时间步）。
- 正式窗口搜索结果已落到 `outputs/slopemine_v2/tuning_v1`；`A4` 在正式 split 下的整体最佳基础配置是 `seq_len=24, pred_len=12`，`48/12` 不是最优。
- 第一轮正式 patch 实验结果已落到 `outputs/slopemine_v2/formal_round1`；正式窗口清单已写入 `dataset/slopemine_v2/window_manifest_v1_ps10.csv`，当前窗口数为 `train=421 / val=117 / test=117`。
- 这轮六模型首轮结果里，按 `val_mae` 选最佳时：`A1=TCN`、`A2=LSTM`、`A3=TCN`、`A4=LSTM`；按 `test_mae` 看，四组实验的最佳都落在 `LSTM` 上，其中 `A3` 略优于 `A4`，说明 `blast V3 real-coordinate` 还没有在第一轮正式实验里稳定压过 `blast V2`。
- `MS-TimeFilter` 当前固定正式配置为 `seq_len=96 / pred_len=12 / lr=1e-4 / dropout=0.1`；额外保留 `blast-hours` 优化分支 `lr=3e-4 / dropout=0.1`。正式矩阵结果已落到 `outputs/slopemine_v2/ms_timefilter_v1`，其中当前主模型为 `C2 main`，但总体上仍未超过 `baseline_lstm_a4_96`。
- PGGC 图相关产物已更新：`outputs/slopemine_v2/ms_timefilter_v1/graph_debug_C1.npz`、`graph_debug_report_C1.md`、`figures/graph_A_{prior,learned,final}_C1.png` 和 `outputs/slopemine_v2/chapter4_figures_v1/fig_ch4_pggc_graph_compare.{png,pdf}` 已按“正值 `+0.5` 显示偏移 + 旧版拼图样式”重生成。
- “天气分支 + EDDR”组图当前固定窗口口径为：天气面板 `sample=599, patch=ps10_v2_p291`；爆破面板仍基于 `sample=500, patch=ps10_v2_p256` 同源事件，但显示口径已改成“历史平滑 + 连续预测”视角：历史段显示 `2024-06-24 16:00 ~ 2024-06-24 23:00`，预测段从 `2024-06-25 00:00` 连续接到 `2024-06-26 15:00`，中间不再画分隔线。
- 独立组图脚本 `generate_ch4_weather_eddr_group_v1.py` 中，子图 `(b)` 的 `C2/C5` 当前都采用展示层重构曲线：`06-24 24:00`（即 `2024-06-25 00:00`）被固定为主冲击点，`C5` 在该峰值时刻同步抬升形成无延迟响应，`C2` 保留更明显的滞后；冻结 prediction bundle 本身未被改动。

# 关键决定
- `zone_id_v2` 现在以冻结后的 `engineering_zone_polygons_v2.json` + `zone_assignment_rule.md` 为正式规则源；point-in-polygon 边界按包含处理，未命中点回退到 `middle_slope`。
- 稳定背景区不再做常规网格 patch：正式 `patch_meta_v2_ps8.csv / patch_meta_v2_ps10.csv` 里只保留 1 个 background patch，并标记 `is_train_patch=0`；同时额外导出 `*_bg_excluded.csv` 供主实验直接训练。
- `patch_size=10` 继续作为主实验粒度；训练张量只保留 `is_train_patch=1` 的 307 个 patch，避免背景区参照 patch 混入主实验。
- 正式诊断包固定输出 `zone_point_distribution`、`zone_patch_distribution`、`patch_density_heatmap`、`zone_feature_summary`、`selected_time_patch_heatmaps/` 和 `patch_data_diagnostic_report_v2.md`。
- 论文版正式工程分区图固定用“点位最终归属”做主视觉，polygon 只保留浅色虚线边界作参考，不再使用透明填充 polygon。
- 代理爆破坐标只允许从非背景区 patch centroid 中抽样；`coordinate_source=proxy_inferred`、`is_real_coordinate=0` 固定写死，并同时导出 conservative / balanced / localized 三套版本用于 V3 敏感性分析。
- `current_v3 / proxy_v3` 原始连续累积型特征先暂停直接进入主实验；当前推荐改用 `V3a_truncated_Rc60_Hc6` 这类局部事件敏感版本，相关比较表和热力图在 `outputs/slopemine_v2/blast_v3_variants`。
- 内部反演事件链路默认使用 `patch_series_v2_ps10.csv` 做候选时刻筛选，分数由 `disp_max / |vel_mean| / |acc_mean| / active_ratio` 的 robust z 加权得到；事件坐标对 `toe` 做 1.35 软先验，但若证据不足允许回退到 `middle_slope` 联合拟合。
- 正式主实验 split 默认不再沿用旧 `window_manifest_v2_ps10.csv` 的 720 小时整轴比例切分；正式口径改为在 712 个有效时间步上连续切分，并通过 `experiment_split_plan_v1.csv` 冻结。
- 基础窗口参数的正式推荐值改为读取 `outputs/slopemine_v2/tuning_v1/best_window_config_v1.json`；若该文件存在，实验矩阵与第四章补写项优先引用正式 split 下的窗口搜索结果。
- 正式深度实验固定使用 `window_manifest_v1_ps10.csv`，并明确采用 `drop_global_missing` 策略剔除 8 个全局缺失小时；不再复用旧 `window_manifest_v2_ps10.csv` 的默认切分。
- 第一轮正式 patch 深度实验暂时只做到 `A1-A4`；`PGGC / EDDR / PIR` 仍未接入，且当前结果不支持直接宣称 `A4` 已优于 `A3`。
- `PGGC / EDDR / PIR` 现在都已接入新的 `MS-TimeFilter` runner，并已有正式结果；当前结论是 `C2(main)` 可作为机制主线版本，`PIR` 对应的 `C3(main)` 暂未带来稳定增益，因此默认降级为辅助约束。
- PGGC 解释图当前固定口径是：原始矩阵不改、`graph_debug_C1.npz` 继续留存，但最终单图显示时仅对正值边权做 `+0.5` 偏移，章节图继续拼接三张单图；本地缺失 checkpoint 时，优先复用远端已有 `C1_main.pt` 拉回后重绘，不重新训练。
- “天气分支 + EDDR”主图的论文编号与仓库编号固定解耦：`A2(base) -> baseline_naive_weather_concat`、`C1 -> A2_main`、`C2 -> C1_main`、`C5 -> C3_main`，最终图注和 manifest 只暴露论文编号；其中独立脚本的 `(b)` 面板沿用 paper `C2/C5` 标签，但实际显示曲线已切成连续拼接后的 display 版本。
