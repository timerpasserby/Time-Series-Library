# 边坡位移预测实验记录 — 三表构造 + 天气特征 + 爆破特征

## 实验日期

2026-03-25

## 实验目标

按照 `325第三章的实验指导.md` 的要求，从聚合宽表方案升级为三表方案（`point_meta`、`disp_series`、`weather`），并合并爆破事件特征，完成包含多源信息的基线对比实验。

---

## 1. 三张主表构造

### 1.1 监测点元数据 `point_meta.csv`

| 指标 | 数值 |
|---|---|
| 监测点总数 | 19,566 |
| 有效点数 | 19,566（全部有效） |
| X 范围 | [0, 296] |
| Y 范围 | [0, 229] |
| Zone 1 点数 | 32（0.16%） |
| Zone 2 点数 | 19,534（99.84%） |

### 1.2 位移时序表 `disp_series.csv`

| 指标 | 数值 |
|---|---|
| 总行数 | 13,930,992 |
| 文件大小 | 650 MB |
| 时间范围 | 2024-06-01 ~ 2024-06-30（712 小时） |
| 字段 | timestamp, point_id, disp, vel, acc |

### 1.3 气象数据 `weather.csv`

| 指标 | 数值 |
|---|---|
| 行数 | 712 |
| 字段 | timestamp, temperature, humidity, wind_speed, rainfall |
| 降雨小时 | 120（16.9%） |
| 温度范围 | [12.11, 33.67] ℃ |

### 1.4 爆破数据 `hourly_blast_features_june_2024.csv`

| 指标 | 数值 |
|---|---|
| 行数 | 720（完整月） |
| 有爆破的天数 | 24 / 30 |
| 有爆破的小时数 | 57（7.9%） |
| 最大小时装药量 | 1006.6 kg |
| 最大 disturbance_index | 8.40 |
| disturbance_index > 0 的小时 | 710（98.6%） |

字段说明：
- `blast_count`: 当前小时爆破次数
- `total_charge_kg`: 当前小时总装药量
- `mean_charge_kg`: 平均单次装药量
- `max_ppv_est_mm_s`: 预估最大质点振动速度
- `disturbance_index`: 时间衰减累积扰动指数（爆破后持续衰减）
- `blast_count_6h`: 最近 6 小时累计爆破次数
- `charge_24h_kg`: 最近 24 小时累计装药量

---

## 2. 数据质量诊断

### 2.1 缺失率

- 所有点 712 个时间步全覆盖（0% 点级缺失）
- 8 小时缺失是全局性的

### 2.2 位移分布

| 指标 | disp | vel | acc |
|---|---|---|---|
| 零值比例 | 53.85% | 54.84% | 53.01% |
| 均值 | 0.143 | -0.0004 | -0.0001 |
| P99 | 1.0 | 0.039 | 0.002 |
| 最大值 | 16.0 | 1.675 | 0.113 |

### 2.3 位移-降雨重叠

- 降雨小时：120
- 位移尖峰小时（> mean+2std）：570
- 重叠：92 小时（P(spike|rain) = 76.7%，P(rain|spike) = 16.1%）

---

## 3. 空间分块分析

| patch_size | 占用 patches | 平均点数/patch |
|---|---|---|
| 5 | 1,258 | 15.6 |
| 8 | 553 | 35.4 |
| 10 | 382 | 51.2 |
| 16 | 171 | 114.4 |

可视化文件位于 `scripts/slopemine/figures/`。

---

## 4. 实验配置

### 4.1 基础聚合（内部特征 + 天气）

**数据文件**：`slopemine_agg_weather.csv`

| 列 | 说明 |
|---|---|
| speed | 聚合速度均值 |
| acceleration | 聚合加速度均值 |
| deformation_std | 位移空间标准差 |
| speed_std | 速度空间标准差 |
| acceleration_std | 加速度空间标准差 |
| deformation_max | 位移空间最大值 |
| temperature | 温度 |
| humidity | 湿度 |
| wind_speed | 风速 |
| rainfall | 降雨量 |
| **deformation** | **目标列（位移均值）** |

### 4.2 基础聚合 + 爆破特征

**数据文件**：`slopemine_agg_blast.csv`（11 内部 + 4 天气 + 7 爆破 = 22 特征）

| 列 | 说明 |
|---|---|
| （继承上述 11 列） | 内部监测 + 天气 |
| blast_count | 爆破次数 |
| total_charge_kg | 总装药量 |
| disturbance_index | 时间衰减扰动指数 |
| blast_count_6h | 6 小时累计爆破次数 |
| charge_24h_kg | 24 小时累计装药量 |
| max_ppv_est_mm_s | 预估最大 PPV |
| mean_charge_kg | 平均单次装药量 |
| **deformation** | **目标列** |

### 4.3 实验参数

| 参数 | 值 |
|---|---|
| seq_len | 24 |
| label_len | 24 |
| pred_len | 6 |
| train_epochs | 1（链路验证） |
| batch_size | 32 |
| features | MS |
| target | deformation |
| freq | h（小时） |

---

## 5. 基线实验结果

### 5.1 内部特征 + 天气（11 特征，enc_in=11）

| 模型 | MSE | MAE |
|---|---:|---:|
| LSTM | **11.035** | 1.224 |
| DLinear | 11.249 | 1.150 |
| STGCN | 11.330 | 1.498 |
| TCN | 12.072 | **1.004** |
| PatchTST | 12.956 | 1.304 |
| TimeFilter | 13.648 | 1.532 |

### 5.2 内部特征 + 天气 + 爆破（17 特征，enc_in=17）

DLinear 多 pred_len 结果：

| pred_len | MSE | MAE | test samples |
|---:|---:|---:|---:|
| 1 | 11.069 | 1.006 | 142 |
| 3 | 11.034 | 1.074 | 140 |
| 6 | 11.249 | 1.150 | 137 |
| 12 | 11.776 | 1.206 | 131 |

| 模型 | MSE (11特征) | MSE (17特征) | 变化 |
|---|---:|---:|---|
| DLinear | 11.249 | 11.249 | 不变 |
| PatchTST | 12.956 | 12.975 | +0.15% |

**说明**：train_epochs=1 下爆破特征未带来显著提升。可能原因：
1. 仅 1 个 epoch 不足以让模型学习到爆破特征的贡献
2. 爆破事件仅占 7.9% 小时，线性模型难以从少量事件中提取有效模式
3. 需要增加 train_epochs 后重新验证

---

## 6. 事件驱动分析

### 6.1 爆破事件统计

- 30 天中有 24 天有爆破（80%）
- 57 个小时有实际爆破事件（7.9%）
- 单小时最大装药量：1006.6 kg（6月27日 15:00）
- 单日最密集爆破：6月14日（5 小时有爆破）、6月7日/11日/17日/24日（各 4 小时）
- disturbance_index 非零小时：710（98.6%）— 说明爆破扰动效应几乎持续全月

### 6.2 爆破特征与位移关系

| 特征 | 说明 |
|---|---|
| disturbance_index | 时间衰减累积扰动指数，爆破后按指数衰减持续影响，范围 [0, 8.40] |
| blast_count_6h | 6 小时窗口累计爆破次数，最多 6 次 |
| charge_24h_kg | 24 小时窗口累计装药量，最大 3140 kg |
| max_ppv_est_mm_s | 预估最大质点振动速度，最大 26.19 mm/s |

**关键发现**：disturbance_index 在 98.6% 的小时非零，说明爆破扰动效应几乎是连续的。这与位移分布的 53.85% 零值形成对比 — 即使扰动持续存在，大部分时间位移仍为零，说明位移响应具有明显的阈值效应。

---

## 7. 下一步计划

1. 合并爆破特征到聚合 CSV
2. 运行含爆破特征的基线实验
3. 对比 11 特征 vs 22 特征的效果差异
4. 分析爆破事件对位移预测的影响
5. 多 pred_len 实验（T=1/3/6/12）
6. 提升 train_epochs 到 20~30

---

## 生成文件清单

| 文件 | 说明 |
|---|---|
| `dataset/slopemine/point_meta.csv` | 监测点元数据 |
| `dataset/slopemine/disp_series.csv` | 位移时序长表（650MB） |
| `dataset/slopemine/weather.csv` | 天气数据 |
| `dataset/slopemine/slopemine_agg_weather.csv` | 聚合+天气（11 特征） |
| `dataset/slopemine/slopemine_agg_blast.csv` | 聚合+天气+爆破（22 特征） |
| `scripts/slopemine/build_three_tables.py` | 三表生成脚本 |
| `scripts/slopemine/data_quality_diag.py` | 数据质量诊断脚本 |
| `scripts/slopemine/data_quality_report.md` | 数据质量诊断报告 |
| `scripts/slopemine/visualize_spatial.py` | 空间可视化脚本 |
| `scripts/slopemine/merge_weather.py` | 天气合并脚本 |
| `scripts/slopemine/figures/` | 可视化图 |
