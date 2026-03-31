## 任务 1：冻结新分区规则

```text id="nd8vmm"
基于当前已经生成的 engineering_zone_v2.png 对应分区结果，输出一个正式可复现的分区规则文件。

要求：
1. 导出 engineering_zone_polygons_v2.json
2. 每个 zone 保存：
   - zone_id_v2
   - zone_name_v2
   - polygon 顶点列表（按顺时针）
3. 输出 zone_assignment_rule.md，写清：
   - polygon 判定规则
   - 若多个 polygon 重叠时的优先级
   - 稳定背景区的特殊处理策略
4. 基于该规则重新生成 point_meta_v2.csv，确保 zone_id_v2 可重复生成。
```

---

## 任务 2：重做 patch 元数据（稳定背景区特殊处理）

```text id="3j3r8n"
基于 point_meta_v2.csv，重新生成 patch_meta_v2_ps10.csv 和 patch_meta_v2_ps8.csv。

要求：
1. 对 zone_id_v2 in {2,3,4,5} 正常做规则 patch 划分
2. 对 zone_id_v2=1（稳定背景区）采用以下两种方案都导出：
   A. 排除出训练 patch
   B. 合并为单一 background patch
3. 输出 patch_stats_v2_refined.csv
4. 输出：
   - active patch 数
   - 稀疏 patch 数
   - 每个 zone 的 patch 数
   - 每个 zone 的平均点数/patch
5. 重新导出 patch_grid_v2_ps8_refined.png 和 patch_grid_v2_ps10_refined.png
```

---

## 任务 3：生成 patch 时序表

```text id="es6w6f"
读取 device_10001_sorted.csv，并结合 point_meta_v2.csv 和 patch_meta_v2_ps10.csv，生成 patch_series_v2_ps10.csv。

按 (report_time, patch_id) 聚合，输出字段：
timestamp, patch_id, zone_id_v2,
disp_mean, disp_std, disp_p95, disp_max,
vel_mean, vel_p95,
acc_mean, acc_p95,
active_ratio, valid_ratio, is_global_missing

要求：
1. 时间轴补齐为完整小时序列
2. 对 8 小时全局缺失显式补齐
3. 输出 patch_series_summary_v2.csv
4. 生成以下图：
   - 各 zone 的 disp_mean 箱线图
   - 各 zone 的 disp_max 箱线图
   - patch 级 disp_mean 热力图（选 3 个代表时刻）
```

---

## 任务 4：做正式实验前的数据诊断包

```text id="432m8n"
基于 point_meta_v2.csv, patch_meta_v2_ps10.csv, patch_series_v2_ps10.csv，生成论文可用的数据诊断包。

输出：
1. zone_point_distribution.csv / png
2. zone_patch_distribution.csv / png
3. patch_density_heatmap.png
4. zone_feature_summary.csv
5. selected_time_patch_heatmaps/
6. markdown 报告：
   patch_data_diagnostic_report_v2.md

报告中必须回答：
- 新分区是否比旧分区更均衡
- patch_size=10 是否适合作为主实验粒度
- 稳定背景区是否应纳入主实验
```

