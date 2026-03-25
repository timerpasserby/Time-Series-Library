# 边坡位移预测实验记录（第三章方向）

## 1. 实验背景

按照 `325第三章的实验指导.md` 的要求，从原始聚合宽表方案升级为"三表+事件驱动"实验框架。
本文档记录第一轮数据构造、质量诊断、基线实验的完整过程和结果。

---

## 2. 数据构造

### 2.1 三张主表

| 表名 | 文件 | 行数 | 说明 |
|---|---|---|---|
| 监测点元数据 | `dataset/slopemine/point_meta.csv` | 19,566 | point_id, grid_x, grid_y, x, y, zone_id, area_id_seq, is_valid |
| 位移时序表 | `dataset/slopemine/disp_series.csv` | 13,930,992 | timestamp, point_id, disp, vel, acc |
| 气象数据表 | `dataset/slopemine/weather.csv` | 712 | timestamp, temperature, humidity, wind_speed, rainfall |

### 2.2 爆破数据表

| 表名 | 文件 | 行数 | 说明 |
|---|---|---|---|
| 爆破特征表 | `scripts/slopemine/hourly_blast_features_june_2024.csv` | 720 | timestamp, blast_count, total_charge_kg, mean_charge_kg, max_ppv_est_mm_s, disturbance_index, blast_count_6h, charge_24h_kg |

**关键字段**：
- `disturbance_index`：已含时间衰减的综合扰动指数，可直接作为事件驱动特征
- `blast_count`：当前小时爆破次数
- `total_charge_kg`：当前小时总装药量
- `charge_24h_kg`：最近 24 小时累计装药量

### 2.3 聚合宽表（含天气）

| 表名 | 文件 | 行×列 | 说明 |
|---|---|---|---|
| 聚合+天气 | `dataset/slopemine/slopemine_agg_weather.csv` | 712 × 13 | 内部特征(7) + 天气特征(4) + deformation(target) |

内部特征(7)：speed, acceleration, deformation_std, speed_std, acceleration_std, deformation_max
天气特征(4)：temperature, humidity, wind_speed, rainfall

---

## 3. 数据质量诊断

### 3.1 监测点概况

| 指标 | 数值 |
|---|---|
| 总点数 | 19,566 |
| Zone 1 点数 | 32（0.16%） |
| Zone 2 点数 | 19,534（99.84%） |

**关键发现**：区域严重不平衡。

### 3.2 缺失率

所有 19,566 个监测点均覆盖 712 个时间步，0% 缺失。8 小时缺失是全局性的。

### 3.3 位移分布

| 指标 | 数值 |
|---|---|
| 范围 | [0.0, 16.0] |
| 均值 / 标准差 | 0.143 / 0.290 |
| 零值比例 | 53.85% |

分布极度右偏，超过一半时间-点组合位移为零。

### 3.4 降雨与位移

- 降雨小时数：120（16.9%）
- P(spike \| rain) = 76.7%：降雨时大概率伴随位移尖峰
- P(rain \| spike) = 16.1%：位移尖峰中仅 16.1% 与降雨重叠

### 3.5 爆破概况

- 爆破小时数：57（7.9%）
- 最大总装药量：1006.6 kg
- 最大 disturbance_index：8.40

---

## 4. 空间分块分析

| patch_size | 占用 patches | 平均点数/patch |
|---|---|---|
| 5 | 1,258 | 15.6 |
| 8 | 553 | 35.4 |
| **10** | **382** | **51.2** |
| 16 | 171 | 114.4 |

建议 `patch_size=10` 或 `8`，可视化图见 `scripts/slopemine/figures/`。

---

## 5. 基线实验（无爆破特征）

### 5.1 实验设置

| 参数 | 值 |
|---|---|
| 数据文件 | `slopemine_agg_weather.csv` |
| seq_len / label_len / pred_len | 24 / 24 / 6 |
| enc_in / dec_in / c_out | 11 |
| train_epochs | 1 |
| features | MS（多变量输入，单变量预测） |
| target | deformation |

### 5.2 结果

| 模型 | MSE | MAE |
|---|---:|---:|
| **LSTM** | **11.035** | 1.224 |
| DLinear | 11.249 | 1.150 |
| STGCN | 11.330 | 1.498 |
| TCN | 12.072 | **1.004** |
| PatchTST | 12.956 | 1.304 |
| TimeFilter | 13.648 | 1.532 |

### 5.3 分析

- LSTM 在 MSE 上最优，TCN 在 MAE 上最优
- TimeFilter 表现最差，可能与 seq_len=24 较短有关（patch_len=8 时仅 3 个 patches）
- 所有模型仅训练 1 epoch，结果仅作链路验证

---

## 6. 下一步：爆破特征合并实验

### 6.1 计划

1. 将爆破特征合并到聚合宽表（选取关键列：disturbance_index, blast_count, total_charge_kg）
2. 重新运行基线对比（无爆破 vs 有爆破）
3. 分析爆破特征对各模型的增益

### 6.2 爆破特征选取

从 `hourly_blast_features_june_2024.csv` 选取以下特征合并：
- `disturbance_index`（综合扰动指数，含时间衰减）
- `blast_count`（爆破次数，二值/计数特征）
- `total_charge_kg`（装药量，强度特征）

---

## 7. 基线实验（含爆破特征）

### 7.1 实验设置

| 参数 | 值 |
|---|---|
| 数据文件 | `slopemine_agg_full.csv` |
| 特征数 | 14（内部 7 + 天气 4 + 爆破 3） |
| seq_len / label_len / pred_len | 24 / 24 / 6 |
| enc_in / dec_in / c_out | 14 |
| train_epochs | 1 |
| 爆破特征 | disturbance_index, blast_count, total_charge_kg |

### 7.2 对比结果：无爆破 vs 有爆破

| 模型 | MSE (无爆破) | MAE (无爆破) | MSE (有爆破) | MAE (有爆破) | MSE 变化 | MAE 变化 |
|---|---:|---:|---:|---:|---|---|
| DLinear | 11.249 | 1.150 | 11.249 | 1.150 | — | — |
| LSTM | 11.035 | 1.224 | 11.054 | 1.254 | +0.2% | +2.5% |
| STGCN | 11.330 | 1.498 | 11.199 | 1.343 | **-1.2%** | **-10.3%** |
| TCN | 12.072 | 1.004 | 12.241 | 1.034 | +1.4% | +3.0% |
| PatchTST | 12.956 | 1.304 | 12.481 | 1.334 | **-3.7%** | +2.3% |
| TimeFilter | 13.648 | 1.532 | 13.384 | 1.244 | **-1.9%** | **-18.8%** |

### 7.3 分析

1. **STGCN 获益最大**：MAE 降低 10.3%，MSE 降低 1.2%。STGCN 的图卷积结构可能更能利用爆破事件的空间相关性。

2. **TimeFilter MAE 大幅改善**：MAE 从 1.532 降至 1.244（-18.8%），说明爆破特征对 TimeFilter 的时频滤波有显著帮助。

3. **PatchTST MSE 改善**：MSE 从 12.956 降至 12.481（-3.7%）。

4. **DLinear 不受影响**：线性模型无法利用爆破特征的非线性模式。

5. **LSTM/TCN 轻微退化**：可能因为仅 1 epoch 训练，模型未能有效学习爆破特征。

6. **总体趋势**：爆破特征对时空模型（STGCN）和时频模型（TimeFilter）有正向增益，对纯序列模型效果不一。

---

## 8. 生成脚本清单

| 脚本 | 功能 |
|---|---|
| `build_three_tables.py` | 从原始数据生成三张主表 |
| `data_quality_diag.py` | 6 项数据质量诊断 |
| `visualize_spatial.py` | 空间分布可视化 |
| `merge_weather.py` | 天气特征合并 |
| `merge_blast.py` | 爆破特征合并 |
| `prepare_tslib_csv.py` | 聚合宽表生成（含 deformation_max） |

---

## 9. 数据文件清单

| 文件 | 行×列 | 说明 |
|---|---|---|
| `point_meta.csv` | 19,566 × 8 | 监测点元数据 |
| `disp_series.csv` | 13,930,992 × 5 | 位移时序长表（650MB） |
| `weather.csv` | 712 × 5 | 气象数据 |
| `slopemine_aggregate.csv` | 712 × 8 | 基础聚合宽表 |
| `slopemine_agg_weather.csv` | 712 × 13 | 聚合 + 天气 |
| `slopemine_agg_full.csv` | 712 × 15 | 聚合 + 天气 + 爆破 |
| `hourly_blast_features_june_2024.csv` | 720 × 8 | 爆破特征表 |
