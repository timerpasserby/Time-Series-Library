# Experiment Lineage v1

> 本文档记录 slopemine v2 从数据冻结到 MS-TimeFilter 正式实验的完整实验血缘。所有实验口径以本文为准，不同对话中的实验记录不再混用。

---

## 0. 数据与 Split 冻结

| 项目 | 值 |
|------|------|
| 数据版本 | `dataset_manifest_v1` + `patch_tensor_base_v2_ps10.npz` |
| Patch 粒度 | `patch_size=10`，307 个训练 patch（背景区排除） |
| 空间分区 | `engineering_zone_polygons_v2.json` + `zone_assignment_rule.md`（冻结） |
| 有效时间步 | 712 小时（720 自然小时剔除 8 个全局缺失小时） |
| Split 版本 | `experiment_split_plan_v1`，候选 `split_cand_01` |
| 缺失策略 | `drop_global_missing` |

### Split 详情

| Split | 时间范围 | 有效步数 | 窗口样本数 | Blast Hours | Rain Hours |
|-------|----------|---------|-----------|-------------|------------|
| **train** | 2024-06-01 00:00 ~ 2024-06-20 06:00 | 456 | 421 | 41 | 78 |
| **val** | 2024-06-20 07:00 ~ 2024-06-25 15:00 | 128 | 117 | 8 | 18 |
| **test** | 2024-06-25 16:00 ~ 2024-06-30 23:00 | 128 | 117 | 8 | 24 |

---

## Phase A：六模型 Baseline（formal_round1）

| 属性 | 值 |
|------|------|
| **阶段编号** | A |
| **数据版本** | `dataset_manifest_v1` + `patch_tensor_base_v2_ps10` |
| **Split 版本** | `experiment_split_plan_v1`（split_cand_01） |
| **seq_len / pred_len** | **24 / 12**（tuning_v1 搜索结果） |
| **天气分支** | 无（A1）/ naive concat（A2 的 weather 是简单拼接，非正式天气分支） |
| **爆破特征** | V2 hourly（A3）/ V3 real-coordinate（A4） |
| **模型** | LSTM / TCN / DLinear / PatchTST / STGCN / TimeFilter |
| **训练配置** | epochs=8, batch_size=8, lr=0.001, weight_decay=1e-5, device=CPU |
| **输出目录** | `outputs/slopemine_v2/formal_round1/` |

### A 阶段实验矩阵

| 实验 ID | 特征集 | 特征维度 | 爆破版本 | 天气 |
|---------|--------|---------|----------|------|
| **A1** | Patch internal only | 10 | 无 | 无 |
| **A2** | A1 + weather | 14 | 无 | naive concat（4 维直接拼接） |
| **A3** | A2 + blast V2 | 21 | V2 hourly（7 维） | naive concat |
| **A4** | A2 + blast V3 | 18 | V3 real-coordinate（4 维 patch-level） | naive concat |

### A 阶段核心结果（各实验最佳模型）

| 实验 | 最佳模型 | val_MAE | test_MAE | 备注 |
|------|----------|---------|----------|------|
| A1 | TCN (val) / LSTM (test) | 0.0406 | 0.1041 | TCN 过拟合严重 |
| A2 | LSTM | 0.0430 | 0.1025 | 天气特征带来小幅改善 |
| A3 | LSTM | 0.0484 | 0.1002 | Blast V2 最佳 test_MAE |
| A4 | LSTM | 0.0441 | 0.1007 | V3 未稳定压过 V2 |

### A 阶段结论

1. **LSTM 是四组实验中最稳定的模型**，在 A2/A3/A4 上均取得最低 test_MAE。
2. **天气特征（naive concat）有效**：A2-LSTM test_MAE 0.1025 < A1-LSTM 0.1041。
3. **Blast V2 有效**：A3-LSTM test_MAE 0.1002 < A2-LSTM 0.1025。
4. **Blast V3 real-coordinate 未稳定压过 V2**：A4-LSTM 0.1007 > A3-LSTM 0.1002。
5. **TCN 在 A1 上 val 最优但 test 过拟合**，不适合单独作为主模型。

---

## Phase B：TimeFilter 超参搜索（timefilter_retune）

| 属性 | 值 |
|------|------|
| **阶段编号** | B |
| **数据版本** | 同 Phase A |
| **Split 版本** | 同 Phase A |
| **seq_len / pred_len** | 搜索 {24, 48, 96} / 12 |
| **天气分支** | naive concat |
| **爆破特征** | V3 real-coordinate（A4 特征集） |
| **模型** | TimeFilter（仅 A4 实验） |
| **搜索网格** | lr ∈ {1e-4, 3e-4, 5e-4} × dropout ∈ {0.1, 0.2, 0.3} |
| **训练配置** | epochs=8, batch_size=8, device=GPU |
| **输出目录** | `outputs/slopemine_v2/timefilter_retune/` |
| **状态** | **已完成** |

### B 阶段结论

TimeFilter 在 A4 特征集上经过 27 组超参搜索后，**最佳配置仍无法逼近 LSTM**。结论偏向"backbone 不适配"而非超参问题。TimeFilter 作为 open-source 模型在边坡位移数据上的表现受限，需要架构级改造。

---

## Phase C：MS-TimeFilter 正式实验（ms_timefilter_v1）

| 属性 | 值 |
|------|------|
| **阶段编号** | C |
| **数据版本** | `dataset_manifest_v1` + `patch_tensor_base_v2_ps10` |
| **Split 版本** | `experiment_split_plan_v1`（split_cand_01） |
| **seq_len / pred_len** | **96 / 12**（与 Phase A 的 24/12 不同，见下方说明） |
| **天气分支** | **正式天气分支**：Linear Projection + Multi-scale Conv1D + Temporal Attention + Time-block Pooling |
| **爆破特征** | V2 hourly（A3）/ V3 real-coordinate（A4） |
| **模型** | MSTimeFilter（多分支架构） |
| **训练配置** | epochs=30, batch_size=4, main_lr=1e-4, blast_opt_lr=3e-4, weight_decay=1e-5, device=GPU |
| **输出目录** | `outputs/slopemine_v2/ms_timefilter_v1/` |
| **状态** | **已完成** |

### C 阶段实验矩阵

| 实验 ID | 描述 | 特征维度 | 新增模块 |
|---------|------|---------|----------|
| **baseline_lstm_a4_96** | A4-LSTM 参考（seq96） | 18 | 无 |
| **baseline_raw_timefilter_a4_96** | A4 原始 TimeFilter 参考（seq96） | 18 | 无 |
| **baseline_naive_weather_concat** | naive weather concat 对照 | 14 | 无 |
| **A2** | internal + weather 正式分支 | 25 | 多分支架构 |
| **A3** | A2 + blast V2 | 25 | 多分支架构 |
| **A4** | A2 + blast V3 real-coordinate | 25 | 多分支架构 |
| **C1** | A4 + PGGC | 25 | 物理制导图构建 |
| **C2** | C1 + EDDR | 25 | 事件驱动动态增强（3 专家 gate） |
| **C3** | C2 + PIR | 25 | Patch 重要性重标定 |

### C 阶段核心结果（main 分支）

| 实验 | val_MAE | test_MAE | blast_test_MAE | 备注 |
|------|---------|----------|----------------|------|
| baseline_lstm_a4_96 | 0.0520 | **0.0980** | **0.0338** | LSTM 仍是最强 |
| baseline_raw_timefilter_a4_96 | 0.0558 | 0.1749 | 0.0929 | 原始 TimeFilter 表现差 |
| baseline_naive_weather_concat | 0.0449 | 0.1779 | 0.0833 | naive concat 严重过拟合 |
| A2 | 0.0614 | 0.1278 | 0.0515 | 正式天气分支有效 |
| A3 | 0.0788 | 0.1416 | 0.0584 | V2 分支 |
| A4_main | 0.0732 | 0.1322 | 0.0508 | V3 分支 |
| A4_blast_opt | 0.0635 | 0.1367 | 0.0543 | V3 高 lr 分支 |
| C1_main | 0.0558 | 0.1208 | 0.0448 | PGGC 有效 |
| C1_blast_opt | 0.0583 | 0.1118 | 0.0424 | PGGC + 高 lr |
| C2_main | 0.0501 | 0.1082 | 0.0413 | EDDR 有效 |
| C2_blast_opt | 0.0558 | 0.1205 | 0.0479 | EDDR + 高 lr |
| C3_main | 0.0530 | 0.1130 | 0.0426 | PIR 效果有限 |
| C3_blast_opt | 0.0521 | 0.1104 | 0.0441 | PIR + 高 lr |

### C 阶段结论

1. **A2 优于 naive weather concat**：test_MAE 0.1278 < 0.1779，正式天气分支有效。
2. **A4 优于 A3**：test_MAE 0.1322 < 0.1416（与 Phase A 结论相反，说明 seq_len=96 + 多分支架构下 V3 优势显现）。
3. **C1 优于 A4 和原始 TimeFilter**：test_MAE 0.1208 < 0.1322 < 0.1749，PGGC 有效。
4. **C2 在 blast_hours 子集上有额外增益**：blast_test_MAE 0.0413 < C1 0.0448，EDDR gate 对爆破事件有效。
5. **C3（PIR）提升有限**：test_MAE 0.1130 > C2 0.1082，PIR 未带来额外改善。
6. **MS-TimeFilter 未超过 LSTM**：最佳 C2_main test_MAE 0.1082 > LSTM 0.0980。

---

## seq_len 差异说明

| 阶段 | seq_len | 原因 |
|------|---------|------|
| Phase A（formal_round1） | **24** | tuning_v1 在六模型 baseline 上搜索得出 24/12 为最优 |
| Phase C（ms_timefilter_v1） | **96** | MS-TimeFilter 的天气分支需要长窗口（96 小时）来学习多尺度卷积和注意力；这是架构决定的固定配置 |
| weather_branch_debug.md | **96** | 属于 Phase C 的诊断产物，与 Phase A 的 24/12 **不属于同一阶段**，不视为冲突 |

**关键说明**：Phase A 的 24/12 和 Phase C 的 96/12 是**不同架构下的最优配置**，不应直接比较绝对数值。Phase C 的 baseline_lstm_a4_96（seq96 LSTM）test_MAE=0.0980 与 Phase A 的 A4-LSTM（seq24）test_MAE=0.1007 接近，说明 seq_len 差异对 LSTM 影响有限，但对 TimeFilter 类模型影响显著。

---

## 实验血缘总览

```
Phase 0: 数据冻结
  |-- dataset_manifest_v1
  |-- patch_tensor_base_v2_ps10 (307 patches, ps10)
  |-- experiment_split_plan_v1 (split_cand_01)
  |-- window_manifest_v1_ps10 (drop_global_missing)
  |
  +-- Phase A: 六模型 Baseline (seq24/pred12, CPU, 8 epochs)
  |     |-- A1: internal only (10 features)
  |     |-- A2: + weather naive concat (14 features)
  |     |-- A3: + blast V2 hourly (21 features)
  |     |-- A4: + blast V3 real-coordinate (18 features)
  |     +-- 结论: LSTM 最稳定; V3 未稳定压过 V2
  |
  +-- Phase B: TimeFilter 超参搜索 (A4, GPU, 27 组)
  |     +-- 结论: backbone 不适配，需架构改造
  |
  +-- Phase C: MS-TimeFilter 正式实验 (seq96/pred12, GPU, 30 epochs)
        |-- Baselines: LSTM(seq96) / raw TimeFilter(seq96) / naive concat
        |-- A2: 正式天气分支
        |-- A3: + blast V2
        |-- A4: + blast V3 (main + blast_opt)
        |-- C1: + PGGC (main + blast_opt)
        |-- C2: + EDDR (main + blast_opt)
        |-- C3: + PIR (main + blast_opt)
        +-- 结论: C2 最佳但未超过 LSTM; PIR 效果有限
```
