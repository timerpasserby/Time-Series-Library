# 当前状态
- `slopemine v2` 已完成正式规则冻结、refined patch 重建、`patch_series_v2_ps10.csv`、论文用诊断包、论文版最终工程分区结果图、三套代理爆破坐标台账与 proxy V3 特征导出，以及 blast V3 局部事件敏感版本诊断。
- 新增“内部反演疑似爆破事件”链路：完全忽略旧爆破台账，只基于内部时序异常和冻结分区规则生成 `inferred_blast_ledger_internal_v1.csv`、候选小时排名、空间图、分区叠加图和诊断报告。
- 新增“正式实验 split 与数据清单”链路：基于冻结后的 `ps10` 数据资产自动生成正式 train / val / test 候选、推荐 split、数据版本清单、实验矩阵和第四章补写项。
- 新增“正式 split 下的窗口参数搜索”链路：在 712 个有效时间步和正式 split 上重新扫描 `seq_len / pred_len`，当前推荐基础配置已更新为 `24 / 12`。
- 新增“正式 patch 级深度实验”链路：基于冻结后的 `dataset_manifest_v1 + experiment_split_plan_v1 + window_manifest_v1_ps10.csv`，实现正式 `dataset / dataloader / runner`，并完成 `LSTM / TCN / DLinear / PatchTST / STGCN / TimeFilter` 六模型的第一轮 `A1-A4` 实验。
- 新增 `MS-TimeFilter` 正式实验代码链路：已实现多分支 formal dataset、正式天气分支 `weather_encoder`、`PGGC / EDDR / PIR` 最小实现、`MSTimeFilter` 主模型和统一 runner，当前已完成 shape / 单步训练烟测，正式整轮实验待服务器运行。

# 上次停点
- 最新结果已落到 `dataset/slopemine_v2`、`outputs/slopemine_v2/diagnostic_package_v2`、`outputs/slopemine_v2/figures/final_zone_assignment_v2.{png,pdf}`、`outputs/slopemine_v2/proxy_blast` 和 `outputs/slopemine_v2/blast_v3_variants`。
- 正式实验计划与数据清单已落到 `outputs/slopemine_v2/experiment_plan_v1`；当前冻结推荐 split 为 `train=2024-06-01 00:00 ~ 2024-06-20 06:00`、`val=2024-06-20 07:00 ~ 2024-06-25 15:00`、`test=2024-06-25 16:00 ~ 2024-06-30 23:00`（基于 712 个有效时间步）。
- 正式窗口搜索结果已落到 `outputs/slopemine_v2/tuning_v1`；`A4` 在正式 split 下的整体最佳基础配置是 `seq_len=24, pred_len=12`，`48/12` 不是最优。
- 第一轮正式 patch 实验结果已落到 `outputs/slopemine_v2/formal_round1`；正式窗口清单已写入 `dataset/slopemine_v2/window_manifest_v1_ps10.csv`，当前窗口数为 `train=421 / val=117 / test=117`。
- 这轮六模型首轮结果里，按 `val_mae` 选最佳时：`A1=TCN`、`A2=LSTM`、`A3=TCN`、`A4=LSTM`；按 `test_mae` 看，四组实验的最佳都落在 `LSTM` 上，其中 `A3` 略优于 `A4`，说明 `blast V3 real-coordinate` 还没有在第一轮正式实验里稳定压过 `blast V2`。
- `MS-TimeFilter` 当前固定正式配置为 `seq_len=96 / pred_len=12 / lr=1e-4 / dropout=0.1`；额外保留 `blast-hours` 优化分支 `lr=3e-4 / dropout=0.1`。本地 CPU 上已完成 `A2 / C2 / C3` 的 batch 与 1-epoch 小样本烟测，但没有完成全量正式矩阵。

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
- `PGGC / EDDR / PIR` 现在已经有最小侵入式实现，但只接到新的 `MS-TimeFilter` runner，不回写旧 `formal_round1` 链路；正式结论必须等待 `run_ms_timefilter_experiments_v1.py` 在服务器上完整跑完后再写。
