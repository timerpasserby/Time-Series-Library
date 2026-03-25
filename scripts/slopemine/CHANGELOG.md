# SlopeMine 实验变更记录

## 2026-03-25: 特征替换 `grid_count` → `deformation_max`

### 变更原因

`grid_count`（每个时间步聚合时的监测点数量）在全部 712 个时间步中均为常量 `19566`，作为时序预测特征没有信息增益，浪费了一个输入通道。

### 替换方案

用 `deformation_max`（每个时间步中所有监测点的最大变形量）替换 `grid_count`。

**物理意义**：`deformation_max` 直接反映边坡最危险位置的变形程度，是边坡稳定性评估中的核心指标。相比均值（`deformation` 列），最大值更能捕捉异常变形事件。

### 数据对比

| 指标 | grid_count（旧） | deformation_max（新） |
|---|---|---|
| 唯一值数量 | 1 | 39 |
| 最小值 | 19566 | 0.0 |
| 最大值 | 19566 | 16.0 |
| 均值 | 19566 | 1.37 |
| 标准差 | 0.0 | 0.97 |

### 修改文件清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `scripts/slopemine/prepare_tslib_csv.py` | 修改 | 4 处改动：`make_bucket()` 增加 `deformation_max` 跟踪；`update_bucket()` 增加最大值更新；`finalize_bucket()` 输出 `deformation_max` 替代 `grid_count`；`aggregate_mode()` 表头和输出列同步修改 |
| `dataset/slopemine/slopemine_aggregate.csv` | 重新生成 | 列 `grid_count` → `deformation_max`，行数 712 不变，无缺失值 |
| `scripts/slopemine/EXPERIMENT_RUN_REPORT.md` | 待更新 | 特征列说明 |

### 对实验配置的影响

- `enc_in=7`、`dec_in=7`、`c_out=7` **不变**（替换而非增减）
- `features=MS`、`target=deformation` **不变**
- 模型配置和批量运行脚本 **无需修改**
- 已有的 checkpoint 和实验结果不受影响（使用旧特征训练的模型需重新训练）

### 技术实现细节

`deformation_max` 采用流式跟踪方式：在 `make_bucket()` 中初始化为 `float('-inf')`，在 `update_bucket()` 中与当前值取最大。此方式与现有的 sum/sumsq 聚合逻辑完全兼容，无需存储原始值，内存开销极低。

---

## 2026-03-25: 三表构造 + 天气特征合并 + 新基线实验

### 变更背景

按照 `325第三章的实验指导.md` 的要求，不再仅使用聚合宽表，而是构造实验指导要求的三张主表（`disp_series.csv`、`point_meta.csv`、`weather.csv`），并在聚合宽表基础上合并天气特征，重新运行基线实验。

### 新建文件

| 文件 | 说明 |
|---|---|
| `scripts/slopemine/build_three_tables.py` | 从原始数据生成三张主表 |
| `scripts/slopemine/data_quality_diag.py` | 6 项数据质量诊断 |
| `scripts/slopemine/data_quality_report.md` | 数据质量诊断报告 |
| `scripts/slopemine/visualize_spatial.py` | 空间分布可视化 |
| `scripts/slopemine/merge_weather.py` | 天气特征合并到聚合 CSV |
| `dataset/slopemine/point_meta.csv` | 19,566 个监测点元数据 |
| `dataset/slopemine/disp_series.csv` | 13,930,992 行位移时序（650MB） |
| `dataset/slopemine/weather.csv` | 712 行天气数据 |
| `dataset/slopemine/slopemine_agg_weather.csv` | 含天气特征的聚合宽表（712行×13列） |
| `scripts/slopemine/figures/` | 空间分布图、patch 热力图 |

### 实验配置变更

| 参数 | 旧值 | 新值 |
|---|---|---|
| `seq_len` | 96 | 24 |
| `label_len` | 96 | 24 |
| `pred_len` | 1 | 6 |
| `enc_in/dec_in/c_out` | 7 | 11 |
| `data_path` | slopemine_aggregate.csv | slopemine_agg_weather.csv |

新增 4 个天气特征：`temperature`, `humidity`, `wind_speed`, `rainfall`

### 数据质量诊断关键发现

1. **区域严重不平衡**：Zone 1 仅 32 点（0.16%），Zone 2 有 19,534 点
2. **无点级缺失**：所有点 712 个时间步全覆盖
3. **位移分布极偏**：53.85% 为零值，P99=1.0
4. **降雨-位移相关**：76.7% 的降雨小时伴随位移尖峰

### Patch 尺寸分析

| patch_size | 占用 patches | 平均点数/patch |
|---|---|---|
| 5 | 1,258 | 15.6 |
| 8 | 553 | 35.4 |
| 10 | 382 | 51.2 |
| 16 | 171 | 114.4 |

### 基线实验结果（pred_len=6, train_epochs=1）

| 模型 | MSE | MAE |
|---|---:|---:|
| LSTM | **11.035** | 1.224 |
| DLinear | 11.249 | 1.150 |
| STGCN | 11.330 | 1.498 |
| TCN | 12.072 | **1.004** |
| PatchTST | 12.956 | 1.304 |
| TimeFilter | 13.648 | 1.532 |

### 下一步

1. 数据集时间切分（70% train / 15% val / 15% test）
2. 将 `train_epochs` 提升到 20~30
3. ~~补充爆破事件数据~~ ✅ 已完成
4. 空间分块建模实验

---

## 2026-03-25: 爆破特征合并 + 事件驱动分析

### 新增文件

| 文件 | 说明 |
|---|---|
| `scripts/slopemine/merge_blast.py` | 爆破特征合并脚本 |
| `dataset/slopemine/slopemine_agg_full.csv` | 含天气+爆破的完整聚合宽表（712×15） |
| `scripts/slopemine/experiment_record_ch3.md` | 第三章方向完整实验记录 |

### 爆破数据概况

- 来源：`hourly_blast_features_june_2024.csv`（720 行，完整覆盖 6 月）
- 爆破小时数：57（7.9%）
- 选取特征：`disturbance_index`（含时间衰减）、`blast_count`、`total_charge_kg`

### 有/无爆破特征对比（pred_len=6, train_epochs=1）

| 模型 | MSE (无爆破) | MSE (有爆破) | MAE (无爆破) | MAE (有爆破) | MAE 变化 |
|---|---:|---:|---:|---:|---|
| DLinear | 11.249 | 11.249 | 1.150 | 1.150 | — |
| LSTM | 11.035 | 11.054 | 1.224 | 1.254 | +2.5% |
| **STGCN** | 11.330 | **11.199** | 1.498 | **1.343** | **-10.3%** |
| TCN | 12.072 | 12.241 | 1.004 | 1.034 | +3.0% |
| PatchTST | 12.956 | 12.481 | 1.304 | 1.334 | +2.3% |
| **TimeFilter** | 13.648 | **13.384** | 1.532 | **1.244** | **-18.8%** |

### 关键发现

1. **STGCN 获益最大**：MAE 降低 10.3%，图卷积结构能利用爆破的空间相关性
2. **TimeFilter MAE 大幅改善**：-18.8%，时频滤波与爆破扰动特征互补
3. **DLinear 不受影响**：线性模型无法利用非线性爆破模式
4. 仅 1 epoch 训练，部分模型可能尚未学会利用爆破特征
