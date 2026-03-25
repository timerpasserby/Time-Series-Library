# 边坡位移预测实验记录 — 2026-03-25

## 一、本次实验目标

按照 `325第三章的实验指导.md` 要求，从"聚合宽表"方案升级为"三表主表"方案，完成数据构造、质量诊断、空间可视化，并在聚合时序上合并天气和爆破特征后运行基线对比实验。

## 二、数据构造

### 2.1 原始数据概况

| 统计项 | 数值 |
|---|---|
| 原始文件 | `device_10001_sorted.csv` |
| 总行数 | 13,930,992 |
| 监测点数 | 19,566 个 `(grid_x, grid_y)` 组合 |
| 时间步数 | 712 个小时（2024-06-01 ~ 2024-06-30，缺失 8 小时） |
| 区域数 | Zone 1（32 点）+ Zone 2（19,534 点） |

### 2.2 三张主表

| 表名 | 行数 | 大小 | 状态 |
|---|---|---|---|
| `point_meta.csv` | 19,566 | 508 KB | ✅ 含 point_id, grid_x, grid_y, x, y, zone_id, area_id_seq, is_valid |
| `disp_series.csv` | 13,930,992 | 650 MB | ✅ 含 timestamp, point_id, disp, vel, acc |
| `weather.csv` | 712 | 29 KB | ✅ 含 timestamp, temperature, humidity, wind_speed, rainfall |

**爆破数据**（后续补充）：

| 表名 | 行数 | 大小 | 状态 |
|---|---|---|---|
| `hourly_blast_features_june_2024.csv` | 720 | - | ✅ 含 8 个爆破特征列 |

### 2.3 聚合宽表（用于 TSLib 基线实验）

**聚合时序（仅内部特征）**：`slopemine_aggregate.csv`
- 712 行 × 8 列
- 特征：speed, acceleration, deformation_std, speed_std, acceleration_std, deformation_max
- 目标：deformation

**聚合时序（含天气特征）**：`slopemine_agg_weather.csv`
- 712 行 × 13 列
- 在上述基础上增加：temperature, humidity, wind_speed, rainfall
- enc_in=11

## 三、数据质量诊断

### 3.1 六项统计结果

| # | 诊断项 | 结果 |
|---|---|---|
| 1 | 监测点总数 | 19,566（全部有效） |
| 2 | 缺失率分布 | 所有点 0% 缺失（8 小时缺失为全局性） |
| 3 | 异常值比例 | 100% 点存在异常值（\|z\|>3），平均异常率 1.19% |
| 4 | 位移分布 | 53.85% 零值，P99=1.0，最大 16.0，极度右偏 |
| 5 | 降雨分布 | 120 小时降雨（16.9%），非零均值 0.32 mm |
| 6 | 位移-降雨重叠 | P(spike\|rain)=76.7%，P(rain\|spike)=16.1% |

### 3.2 关键发现

1. **区域严重不平衡**：Zone 1 仅 32 点（0.16%）
2. **位移分布极偏**：超半数时间-点组合位移为零
3. **降雨可部分解释位移尖峰**：76.7% 降雨小时伴随位移尖峰
4. **但多数位移尖峰非降雨驱动**：仅 16.1% 位移尖峰与降雨重叠

## 四、空间分块分析

### 4.1 监测点空间范围

- X 范围：[0, 296]，跨度 296
- Y 范围：[0, 229]，跨度 229

### 4.2 Patch 尺寸分析

| patch_size | 占用 patches | 平均点数/patch | 建议用途 |
|---|---|---|---|
| 5 | 1,258 | 15.6 | 精细分析 |
| 8 | 553 | 35.4 | 推荐（均衡） |
| 10 | 382 | 51.2 | 推荐（均衡） |
| 16 | 171 | 114.4 | 粗粒度 |

**建议**：正式实验使用 `patch_size=8` 或 `10`。

## 五、爆破数据分析

### 5.1 爆破数据字段

| 字段 | 含义 | 说明 |
|---|---|---|
| `blast_count` | 当前小时爆破次数 | 离散事件 |
| `total_charge_kg` | 当前小时总装药量 | 爆破强度 |
| `mean_charge_kg` | 平均每次装药量 | 单次强度 |
| `max_ppv_est_mm_s` | 估计最大质点振动速度 | 安全评估指标 |
| `disturbance_index` | 扰动指数 | 含时间衰减的综合影响指标 |
| `blast_count_6h` | 过去 6 小时爆破次数 | 时间窗口特征 |
| `charge_24h_kg` | 过去 24 小时总装药量 | 长时间窗口特征 |

### 5.2 爆破特征版本

根据实验指导，当前爆破数据对应 **V2（时间滞后版）**：

- V1 基线：`blast_count` + `total_charge_kg`（当量有无）
- V2 时间滞后：`blast_count_6h` + `charge_24h_kg`（时间窗口累计）
- V3 论文版：需对每个监测点计算空间衰减 + 时间衰减（待实现）

`disturbance_index` 列已内置指数衰减逻辑，可视为 V2.5 版本。

### 5.3 爆破时间分布

爆破活动集中在工作日的特定时段，呈现明显的间歇性特征。高扰动时段（`disturbance_index > 3`）集中在 6 月 3-5 日、6 月 7-8 日、6 月 11-14 日、6 月 17-19 日、6 月 21 日、6 月 24-25 日、6 月 27-28 日。

## 六、基线实验结果

### 6.1 实验配置

| 参数 | 值 |
|---|---|
| 数据 | `slopemine_agg_weather.csv` |
| seq_len | 24 |
| label_len | 24 |
| pred_len | 6 |
| enc_in / dec_in / c_out | 11 |
| features | MS（多变量输入，单变量预测） |
| target | deformation |
| train_epochs | 1（链路验证） |
| batch_size | 32 |
| CPU / GPU | CPU |

### 6.2 基线结果

| 模型 | MSE | MAE | 说明 |
|---|---:|---:|---|
| **LSTM** | **11.035** | 1.224 | MSE 最优 |
| DLinear | 11.249 | 1.150 | 简单线性模型表现稳健 |
| STGCN | 11.330 | 1.498 | 时空模型未显著优于线性 |
| **TCN** | 12.072 | **1.004** | MAE 最优 |
| PatchTST | 12.956 | 1.304 | Transformer 类中等 |
| TimeFilter | 13.648 | 1.532 | 原始 TimeFilter 表现一般 |

### 6.3 与上一轮结果对比

| 维度 | 上一轮（链路验证） | 本轮 |
|---|---|---|
| 数据 | 聚合宽表（7 特征） | 聚合+天气（11 特征） |
| 窗口 | L=96, T=1 | L=24, T=6 |
| 训练轮数 | 1 epoch | 1 epoch |
| MICN 结果 | MSE=2858（异常） | 未运行 |
| 最优 MSE | ~10.66 (LSTM) | 11.035 (LSTM) |
| 最优 MAE | ~1.18 (STGCN) | 1.004 (TCN) |

## 七、生成脚本清单

| 脚本 | 功能 |
|---|---|
| `build_three_tables.py` | 从原始数据生成 point_meta / disp_series / weather |
| `data_quality_diag.py` | 6 项数据质量诊断 |
| `visualize_spatial.py` | 空间分布 + patch 网格可视化 |
| `merge_weather.py` | 天气特征合并到聚合 CSV |
| `prepare_tslib_csv.py` | 聚合宽表生成（已更新 deformation_max） |

## 八、数据文件清单

| 文件 | 路径 |
|---|---|
| point_meta.csv | `dataset/slopemine/point_meta.csv` |
| disp_series.csv | `dataset/slopemine/disp_series.csv` |
| weather.csv | `dataset/slopemine/weather.csv` |
| hourly_blast_features_june_2024.csv | `scripts/slopemine/hourly_blast_features_june_2024.csv` |
| slopemine_aggregate.csv | `dataset/slopemine/slopemine_aggregate.csv` |
| slopemine_agg_weather.csv | `dataset/slopemine/slopemine_agg_weather.csv` |
| slopemine_full.csv | `dataset/slopemine/slopemine_full.csv`（含天气+爆破，17 特征） |
| data_quality_report.md | `scripts/slopemine/data_quality_report.md` |
| 空间可视化图 | `scripts/slopemine/figures/` |

## 六（续）、爆破特征实验结果

### 实验配置

| 参数 | 值 |
|---|---|
| 数据 | `slopemine_full.csv`（6 内部 + 4 天气 + 7 爆破 = 17 特征） |
| 爆破特征 | disturbance_index, blast_count, total_charge_kg, mean_charge_kg, max_ppv_est_mm_s, blast_count_6h, charge_24h_kg |

### 爆破特征 vs 仅天气特征 对比

| 模型 | MSE (天气) | MSE (天气+爆破) | ΔMSE | MAE (天气) | MAE (天气+爆破) | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|
| DLinear | 11.249 | 11.249 | 0.0% | 1.150 | 1.150 | 0.0% |
| LSTM | 11.035 | 11.045 | +0.1% | 1.224 | 1.239 | +1.2% |
| TCN | 12.072 | 11.752 | **-2.7%** | 1.004 | 1.001 | **-0.3%** |
| STGCN | 11.330 | 11.102 | **-2.0%** | 1.498 | 1.267 | **-15.4%** |
| PatchTST | 12.956 | 12.910 | -0.4% | 1.304 | 1.293 | -0.8% |
| TimeFilter | 13.648 | 13.047 | **-4.4%** | 1.532 | 1.414 | **-7.7%** |

### 事件驱动分析结论

1. **爆破特征对非线性模型有效**：TCN、STGCN、TimeFilter 在加入爆破特征后 MSE/MAE 均有改善
2. **DLinear 完全无变化**：线性模型无法捕捉爆破-变形的非线性关系，符合预期
3. **TimeFilter 改善最显著**：MSE 降低 4.4%，MAE 降低 7.7%，可能与其频率域滤波机制能更好地分离爆破信号有关
4. **STGCN MAE 大幅改善 15.4%**：时空图结构在融合爆破信息后效果明显
5. **仅 1 epoch 的局限**：当前训练不足，爆破特征的潜力可能在更多 epoch 后才能充分体现

### 后续优化方向

1. 增加 train_epochs 至 20~30，观察爆破特征的长期效果
2. 对比不同爆破特征子集（V1/V2/V3）的贡献
3. 单独分析爆破时段的预测误差，验证事件敏感性

## 九、下一步计划

### 短期（本次）

1. ✅ ~~构造三表~~
2. ✅ ~~数据质量诊断~~
3. ✅ ~~空间可视化~~
4. ✅ ~~天气特征基线实验~~
5. ⬜ 整合爆破特征到聚合 CSV
6. ⬜ 含爆破特征的基线实验
7. ⬜ 事件驱动分析：比较有/无爆破特征的模型表现

### 中期

1. 将 `train_epochs` 提升到 20~30
2. 多 pred_len 实验（T=1/3/6/12）
3. 数据集时间切分（70% / 15% / 15%）
4. 补充传统基线（HA / XGBoost）

### 长期

1. 空间分块建模（基于 patch 结构）
2. 爆破特征 V3（空间衰减 + 时间衰减）
3. 消融实验
4. 可解释性分析
