# experiment_lineage_v1

- 本说明只承认“当前仓库中已经真实落盘”的结果文件。
- 当前统一把实验分成 `A/B/C/D` 四个阶段；不同阶段的 split、窗口、天气分支和模型结构不能混表解释。
- 截至当前版本，`C 阶段` 与 `D 阶段` 都已经有正式结果，但二者研究目的不同：
  - `C 阶段`：第一轮正式 baseline 深度实验。
  - `D 阶段`：MS-TimeFilter 机制实验，当前主模型固定为 `C2 main = PGGC + EDDR`。

## 统一血缘表

| 阶段 | 阶段名称 | 数据版本 | split 版本 | seq_len / pred_len | 天气分支版本 | 爆破特征版本 | 模型版本 | 当前状态 | 对应主要结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | gate 基线阶段 | `slopemine_v2` 早期正式资产，核心入口是 `patch_tensor_base_v2_ps10.npz + window_manifest_v2_ps10.csv` | `window_manifest_v2_ps10.csv` 内部旧切分；不是 `split_cand_01` | 多组搜索：`24/1,3,6,12`、`48/1,3,6,12`、`96/1,3,6,12` | `naive concat` | `V2` 与 `real-coordinate V3` | 共享权重 `summary ridge` gate baseline | 已完成 | `A4` 相比 `A3` 在 12 组里有 10 组更优，`best A4 = seq_len=96, pred_len=1`。该结论只用于“是否进入机制阶段”，不能直接作为正式深度模型主结果。 |
| B | 正式 split / 窗口 / 公平性冻结阶段 | `dataset_manifest_v1` | `experiment_split_plan_v1.csv`，正式方案为 `split_cand_01` | 窗口搜索：`24/1,3,6,12`、`48/1,3,6,12`、`96/1,3,6,12`；TimeFilter 复核：`24/48/96 × lr × dropout` | 不适用 | 不适用 | split 搜索、窗口搜索、TimeFilter 公平性复核 | 已完成 | 正式 split 冻结为 `train=2024-06-01 00:00~2024-06-20 06:00`、`val=2024-06-20 07:00~2024-06-25 15:00`、`test=2024-06-25 16:00~2024-06-30 23:00`；正式 baseline 窗口推荐为 `24/12`；公平性复核后，原始 TimeFilter 的相对更优配置是 `96/12 + lr=1e-4 + dropout=0.1`。 |
| C | 第一轮正式 patch baseline 深度实验 | `dataset_manifest_v1`，正式训练 patch 数 `307`，`stable_background` 排除 | `split_cand_01` | 固定 `24 / 12` | `naive concat`；天气只是外生变量直接拼接 | `A3=V2`，`A4=real-coordinate V3` | 六个原始骨干：`LSTM / TCN / DLinear / PatchTST / STGCN / raw TimeFilter` | 已完成 | 当前第四章 baseline 主结果来源。最佳 `A4` 来自 `LSTM`，`test_mae=0.100700`；`raw TimeFilter` 在相同正式口径下显著弱于 `LSTM`；在这一阶段，`A4` 没有同骨干稳定压过 `A3`。 |
| D | MS-TimeFilter 机制正式实验 | 仍基于 `dataset_manifest_v1` 与 `patch_tensor_base_v2_ps10.npz` | `split_cand_01` | 固定 `96 / 12` | 第三章正式版本：`formal Conv+Attn`，即 `Linear Projection + Multi-scale Conv1D + Temporal Attention + Time-block Pooling` | `A3=V2`，`A4/C1/C2/C3=real-coordinate V3` | `baseline_lstm_a4_96`、`baseline_raw_timefilter_a4_96`、`A2/A3/A4`、`C1=PGGC`、`C2=EDDR`、`C3=PIR` | 已完成 | 当前机制阶段主模型固定为 `C2 main`。在这一阶段，`A2` 明显优于 `baseline_naive_weather_concat`，`A4` 优于 `A3`，`C1` 优于 `A4` 与 `raw TimeFilter`，`C2` 进一步优于 `C1`；但 `C2 main(test_mae=0.108187)` 仍未超过 `baseline_lstm_a4_96(test_mae=0.097959)`。`C3` 当前退化，因此 `PIR` 暂不作为主模块。 |

## 口径约束

1. `A 阶段` 与 `C/D 阶段` 不能混用。
`A 阶段` 的 `A4 > A3` 只是早期 gate 决策，不等于深度模型已经正式验证 `A4 > A3`。

2. `B 阶段` 只冻结口径和公平性，不直接提供机制增益结论。
`B 阶段` 冻结了 `dataset_manifest_v1 + split_cand_01`，并提供了正式 baseline 的窗口与 raw TimeFilter 的相对公平配置。

3. `C 阶段` 与 `D 阶段` 都是正式结果，但用途不同。
- `C 阶段` 用于“第一轮正式 baseline 对比”。
- `D 阶段` 用于“MS-TimeFilter 机制与当前主模型版本”。

4. 当前主模型版本固定为 `D 阶段` 的 `C2 main`。
凡是写“当前机制主线模型”“当前推荐主模型”“PGGC + EDDR 主版本”，都应指向 `outputs/slopemine_v2/ms_timefilter_v1/results_C2.csv` 中的 `C2 main`。

5. `weather_branch_debug.md` 的 `seq_len=96` 与早期 `formal_round1` 的 `seq_len=24` 不视为冲突。
原因是二者属于不同阶段：
- `seq_len=24` 属于 `C 阶段` baseline。
- `seq_len=96` 属于 `D 阶段` 机制实验。
它们的模型结构、天气分支定义和研究目的都不同，不能直接混在同一张主结果表里。

## 对现有文件的解释约束

- `outputs/slopemine_v2/baseline_v2/baseline_gate_summary_v2.json`
  解释为 `A 阶段` gate 决策文件，不解释为正式主结果。

- `outputs/slopemine_v2/timefilter_retune/timefilter_retune_results.csv`
  解释为 `B 阶段` 的 raw TimeFilter 公平性复核文件，不解释为机制模型结果。

- `outputs/slopemine_v2/formal_round1/results_main_A1_A4.csv`
  解释为 `C 阶段` baseline 正式主结果来源。

- `outputs/slopemine_v2/ms_timefilter_v1/results_main_all.csv`
  解释为 `D 阶段` 机制正式主结果来源。

- `outputs/slopemine_v2/ms_timefilter_v1/weather_branch_debug.md`
  解释为 `D 阶段` 正式天气分支调试文件，不与 `C 阶段` 的 `24/12` baseline 混用。
