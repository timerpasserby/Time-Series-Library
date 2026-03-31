# 当前状态
- `slopemine v2` 已完成正式规则冻结、refined patch 重建、`patch_series_v2_ps10.csv`、论文用诊断包、论文版最终工程分区结果图、三套代理爆破坐标台账与 proxy V3 特征导出，以及 blast V3 局部事件敏感版本诊断。
- 新增“内部反演疑似爆破事件”链路：完全忽略旧爆破台账，只基于内部时序异常和冻结分区规则生成 `inferred_blast_ledger_internal_v1.csv`、候选小时排名、空间图、分区叠加图和诊断报告。

# 上次停点
- 最新结果已落到 `dataset/slopemine_v2`、`outputs/slopemine_v2/diagnostic_package_v2`、`outputs/slopemine_v2/figures/final_zone_assignment_v2.{png,pdf}`、`outputs/slopemine_v2/proxy_blast` 和 `outputs/slopemine_v2/blast_v3_variants`。
- 如果继续做实验，应直接基于新的 `patch_meta_v2_ps10.csv` 和 `patch_tensor_base_v2_ps10.npz` 往下接。

# 关键决定
- `zone_id_v2` 现在以冻结后的 `engineering_zone_polygons_v2.json` + `zone_assignment_rule.md` 为正式规则源；point-in-polygon 边界按包含处理，未命中点回退到 `middle_slope`。
- 稳定背景区不再做常规网格 patch：正式 `patch_meta_v2_ps8.csv / patch_meta_v2_ps10.csv` 里只保留 1 个 background patch，并标记 `is_train_patch=0`；同时额外导出 `*_bg_excluded.csv` 供主实验直接训练。
- `patch_size=10` 继续作为主实验粒度；训练张量只保留 `is_train_patch=1` 的 307 个 patch，避免背景区参照 patch 混入主实验。
- 正式诊断包固定输出 `zone_point_distribution`、`zone_patch_distribution`、`patch_density_heatmap`、`zone_feature_summary`、`selected_time_patch_heatmaps/` 和 `patch_data_diagnostic_report_v2.md`。
- 论文版正式工程分区图固定用“点位最终归属”做主视觉，polygon 只保留浅色虚线边界作参考，不再使用透明填充 polygon。
- 代理爆破坐标只允许从非背景区 patch centroid 中抽样；`coordinate_source=proxy_inferred`、`is_real_coordinate=0` 固定写死，并同时导出 conservative / balanced / localized 三套版本用于 V3 敏感性分析。
- `current_v3 / proxy_v3` 原始连续累积型特征先暂停直接进入主实验；当前推荐改用 `V3a_truncated_Rc60_Hc6` 这类局部事件敏感版本，相关比较表和热力图在 `outputs/slopemine_v2/blast_v3_variants`。
- 内部反演事件链路默认使用 `patch_series_v2_ps10.csv` 做候选时刻筛选，分数由 `disp_max / |vel_mean| / |acc_mean| / active_ratio` 的 robust z 加权得到；事件坐标对 `toe` 做 1.35 软先验，但若证据不足允许回退到 `middle_slope` 联合拟合。
