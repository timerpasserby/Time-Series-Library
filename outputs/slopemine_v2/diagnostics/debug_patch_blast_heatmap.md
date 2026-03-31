# patch 级爆破热力图调试报告

## 结论
- 数据不是空的，三个指定时刻都存在完整的 310 条 patch 记录。
- merge 不是问题，`patch_id` 能完整对齐到 `patch_meta_v2_ps10.csv`，关键字段齐全。
- 旧图几乎全黑、坐标轴显示 0~1 的根因是：矩形 patch 图没有显式设置真实坐标范围，坐标轴停留在默认值；恰好原点附近 patch 覆盖了整个默认视口。
- 同时还存在色标压缩：线性色标直接用全局最大值时，大部分 patch 会偏暗，所以新增了 `P99` 截断版和 `log1p` 版。

## 逐时刻检查

### 2024-06-05 11:00
- 总 patch 数：308
- 非零 patch 数：308
- blast_patch_decay 统计：min=0.077658, p50=36.639570, p90=281.676758, p99=443.871653, max=466.826233
- 是否存在 NaN：False
- merge 后关键字段是否完整：True

- 修复后图文件：
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240605_1100.png`
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240605_1100_p99.png`
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240605_1100_log1p.png`

### 2024-06-07 10:00
- 总 patch 数：308
- 非零 patch 数：308
- blast_patch_decay 统计：min=0.041114, p50=6.319490, p90=59.784499, p99=91.445427, max=96.256012
- 是否存在 NaN：False
- merge 后关键字段是否完整：True

- 修复后图文件：
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240607_1000.png`
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240607_1000_p99.png`
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240607_1000_log1p.png`

### 2024-06-27 15:00
- 总 patch 数：308
- 非零 patch 数：308
- blast_patch_decay 统计：min=0.077855, p50=7.941675, p90=50.948764, p99=74.022700, max=79.244141
- 是否存在 NaN：False
- merge 后关键字段是否完整：True

- 修复后图文件：
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240627_1500.png`
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240627_1500_p99.png`
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/figures/blast_heatmap_20240627_1500_log1p.png`

## 修复说明
- 热力图绘制方式已改成基于 patch rectangle 的着色图。
- 背景改为浅灰色，非活动 patch（值 <= 0 或 NaN）不绘制。
- 若某个指定时刻数据为空或 merge 失败，脚本会直接报错，不再生成默认黑图。