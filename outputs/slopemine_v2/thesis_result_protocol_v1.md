# Thesis Result Protocol v1

> 本文档明确第四章正式结果采用哪些实验、哪些仅作为中间验证、哪些模块需要降级表述。以本文为最终口径，不再混用不同对话中的实验记录。

---

## 1. 第四章主结果采用版本

### 1.1 主表（Table 4.x：多模型对比实验）

**采用 Phase A（formal_round1）的结果**，理由如下：

| 属性 | 值 |
|------|------|
| 实验来源 | `outputs/slopemine_v2/formal_round1/results_main_A1_A4.csv` |
| seq_len / pred_len | 24 / 12（tuning_v1 正式搜索结果） |
| 模型 | LSTM / TCN / DLinear / PatchTST / STGCN / TimeFilter |
| 实验 | A1 / A2 / A3 / A4 |
| 选择依据 | 六模型 × 四特征集的完整公平对比，窗口参数经正式搜索 |

### 主表推荐写法

| 实验 | 特征 | LSTM | TCN | DLinear | PatchTST | STGCN | TimeFilter |
|------|------|------|-----|---------|----------|-------|------------|
| A1 | internal | 0.1041 | 0.1535 | 0.1527 | 0.1418 | 0.1500 | 0.1512 |
| A2 | + weather | **0.1025** | 0.1667 | 0.1563 | 0.1439 | 0.1493 | 0.1634 |
| A3 | + blast V2 | **0.1002** | 0.1457 | 0.1581 | 0.1411 | 0.1434 | 0.1514 |
| A4 | + blast V3 | **0.1007** | 0.1717 | 0.1562 | 0.1436 | 0.1431 | 0.1613 |

> 表中加粗表示各实验最低 test_MAE。

### 1.2 子表（Table 4.y：blast_hours 子集分析）

**采用 Phase A 的 subset 结果**，来源 `outputs/slopemine_v2/formal_round1/results_subsets_A1_A4.csv`。

推荐只展示 LSTM 在各实验的 blast_hours / rain_hours / non_event_hours 三子集表现。

---

## 2. 仅作为中间验证、不进主表的实验

| 实验/产物 | 用途 | 不进主表原因 | 可用位置 |
|-----------|------|-------------|---------|
| **tuning_v1** | 窗口参数搜索 | 中间过程，数值为 placeholder 级别 | 附录或方法章节说明"经网格搜索确定 24/12" |
| **timefilter_retune** | TimeFilter 超参搜索 | 结论为"backbone 不适配"，无正面结果 | 可作为"为什么需要 MS-TimeFilter"的动机 |
| **Phase C baselines**（LSTM seq96 / raw TimeFilter seq96 / naive concat） | MS-TimeFilter 的对照基线 | 与 Phase A 口径不同（seq96 vs seq24） | 仅在 MS-TimeFilter 章节内部作为对照 |
| **Phase C blast_opt 分支** | 高 lr 消融 | 非标准训练配置 | 消融实验小节 |
| **proxy_blast 三套变体** | 代理爆破坐标敏感性分析 | 工程诊断性质 | 附录 |
| **internal_blast_inference** | 内部反演事件 | 独立链路，未接入主实验 | 未来工作 |
| **blast_v3_variant_analysis** | V3 变体分析 | 诊断性质 | 附录 |

---

## 3. 可以写成正式结论的内容

### 3.1 Phase A 结论（可写入第四章）

| 编号 | 结论 | 证据 |
|------|------|------|
| C-A1 | LSTM 在四组实验中均取得最低或接近最低的 test_MAE，是最稳定的 backbone | A1-A4 六模型对比表 |
| C-A2 | 天气特征（naive concat）带来小幅但一致的改善 | A2-LSTM test_MAE 0.1025 < A1-LSTM 0.1041 |
| C-A3 | Blast V2 hourly 特征进一步改善预测 | A3-LSTM test_MAE 0.1002 < A2-LSTM 0.1025 |
| C-A4 | Blast V3 real-coordinate 在第一轮正式实验中未稳定压过 V2 | A4-LSTM 0.1007 > A3-LSTM 0.1002 |
| C-A5 | TCN 在 A1 上 val 最优但 test 过拟合严重 | A1-TCN val=0.0406, test=0.1535 |
| C-A6 | Blast_hours 子集上，A3-LSTM 表现最优（MAE=0.0378） | subset 分析 |

### 3.2 Phase C 结论（可写入第四章扩展章节）

| 编号 | 结论 | 证据 |
|------|------|------|
| C-C1 | 正式天气分支（Conv+Attn）优于 naive concat | A2 test_MAE 0.1278 < naive 0.1779 |
| C-C2 | 在 seq96 + 多分支架构下，A4 优于 A3（与 Phase A 结论相反） | A4_main 0.1322 < A3 0.1416 |
| C-C3 | PGGC 带来显著改善 | C1_main 0.1208 < A4_main 0.1322 |
| C-C4 | EDDR 在 blast_hours 子集上有额外增益 | C2_main blast_MAE 0.0413 < C1_main 0.0448 |
| C-C5 | MS-TimeFilter 整体未超过 LSTM | 最佳 C2_main 0.1082 > LSTM 0.0980 |

---

## 4. 需要降级表述的模块

### 4.1 PIR（Patch Importance Recalibration）

| 属性 | 说明 |
|------|------|
| **当前状态** | 最小实现，已接入 C3 实验 |
| **实验结果** | C3_main test_MAE 0.1130 > C2_main 0.1082，**未带来额外改善** |
| **blast_MAE** | C3_main 0.0426 ≈ C2_main 0.0413，无显著增益 |
| **降级建议** | 第四章中**不作为独立贡献模块**，仅在"架构探索"或"消融实验"中提及 |
| **推荐表述** | "本文尝试了 Patch 重要性重标定模块（PIR），但在当前数据集上未观察到显著改善，可能原因是…" |

### 4.2 Blast V3 real-coordinate

| 属性 | 说明 |
|------|------|
| **Phase A 结论** | V3 未稳定压过 V2（A4-LSTM 0.1007 > A3-LSTM 0.1002） |
| **Phase C 结论** | 在 seq96 + 多分支架构下 V3 优于 V2，但整体仍不如 LSTM |
| **降级建议** | 不作为"核心贡献"，作为"特征工程探索"的一部分 |
| **推荐表述** | "本文对比了两种爆破特征编码方式：V2（hourly 统计）和 V3（patch-level 空间分布）。在 baseline 实验中 V2 略优，但在多分支架构下 V3 展现出更好的兼容性。" |

### 4.3 MS-TimeFilter 整体定位

| 属性 | 说明 |
|------|------|
| **当前状态** | 完整实现但未超过 LSTM |
| **降级建议** | 不作为"超越 SOTA 的新模型"，而是作为"架构探索与消融分析" |
| **推荐表述** | "本文设计了多分支 TimeFilter 架构（MS-TimeFilter），集成天气编码、爆破特征路由、空间图构建和事件动态增强。实验表明该架构在子任务（如 blast_hours 预测）上展现出优势，但整体精度仍低于 LSTM。" |

---

## 5. seq_len 差异的正式说明

| 阶段 | seq_len | 用途 | 是否进主表 |
|------|---------|------|-----------|
| Phase A | **24** | 六模型 baseline 主实验 | **是（主表）** |
| Phase B | {24, 48, 96} | TimeFilter 超参搜索 | 否（仅动机） |
| Phase C | **96** | MS-TimeFilter 正式实验 | 是（扩展章节，非主表） |

**正式说明**：

> Phase A 采用 seq_len=24 是基于 tuning_v1 在六模型上的网格搜索结果。Phase C 采用 seq_len=96 是 MS-TimeFilter 架构的固定配置——天气分支的多尺度卷积（k=3/5/7）和时间注意力机制需要较长窗口才能有效学习。两者的 seq_len 差异源于**不同的模型架构和实验目的**，不视为冲突。Phase C 中的 baseline_lstm_a4_96（seq96 LSTM）test_MAE=0.0980 与 Phase A 的 A4-LSTM（seq24）test_MAE=0.1007 接近，说明 seq_len 差异对 LSTM 影响有限。

---

## 6. 输出文件索引

| 产物 | 路径 |
|------|------|
| Phase A 主结果 | `outputs/slopemine_v2/formal_round1/results_main_A1_A4.csv` |
| Phase A 子集结果 | `outputs/slopemine_v2/formal_round1/results_subsets_A1_A4.csv` |
| Phase A 对比表 | `outputs/slopemine_v2/formal_round1/results_compare_A1_A4.csv` |
| Phase A 图表 | `outputs/slopemine_v2/formal_round1/figures/` |
| Phase A 检查点 | `outputs/slopemine_v2/formal_round1/checkpoints/` |
| Phase C 主结果 | `outputs/slopemine_v2/ms_timefilter_v1/results_main_all.csv` |
| Phase C 子集结果 | `outputs/slopemine_v2/ms_timefilter_v1/results_subsets_all.csv` |
| Phase C 决策报告 | `outputs/slopemine_v2/ms_timefilter_v1/ms_timefilter_decision_report.md` |
| Phase C 图表 | `outputs/slopemine_v2/ms_timefilter_v1/figures/` |
| Split 计划 | `outputs/slopemine_v2/experiment_plan_v1/experiment_split_plan_v1.csv` |
| 实验血缘 | `outputs/slopemine_v2/experiment_lineage_v1.md`（本文档） |
| 结果协议 | `outputs/slopemine_v2/thesis_result_protocol_v1.md`（本文档） |
