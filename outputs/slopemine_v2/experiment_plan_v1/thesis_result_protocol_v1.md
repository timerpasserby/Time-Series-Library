# thesis_result_protocol_v1

## 总原则

- 论文里只允许引用“当前已经完成并落盘”的实验结果。
- 同一主题若在不同阶段存在多个口径，必须先判断它属于 `C 阶段 baseline` 还是 `D 阶段机制实验`，不能跨阶段混表。
- 当前仓库里，`formal_round1` 与 `ms_timefilter_v1` 都已经完成；但二者回答的问题不同。

## 第四章主结果采用哪个版本

第四章建议拆成两层：

1. baseline 主结果
- 采用 `C 阶段`：
  - 数据版本：`dataset_manifest_v1`
  - split：`split_cand_01`
  - patch 粒度：`patch_size=10`
  - 正式训练 patch 数：`307`
  - `stable_background`：排除
  - 窗口：`seq_len=24`、`pred_len=12`
  - 天气版本：`naive concat`
  - 爆破版本：`A3 = V2`，`A4 = real-coordinate V3`
  - 结果来源：
    - `outputs/slopemine_v2/formal_round1/results_main_A1_A4.csv`
    - `outputs/slopemine_v2/formal_round1/results_subsets_A1_A4.csv`

2. 机制主结果 / 当前主模型版本
- 采用 `D 阶段`：
  - 数据版本仍为 `dataset_manifest_v1`
  - split 仍为 `split_cand_01`
  - patch 粒度仍为 `patch_size=10`
  - 窗口：`seq_len=96`、`pred_len=12`
  - 天气版本：`formal Conv+Attn`
  - 爆破版本：`A4/C1/C2/C3 = real-coordinate V3`
  - 当前主模型固定为：`C2 main = PGGC + EDDR`
  - 结果来源：
    - `outputs/slopemine_v2/ms_timefilter_v1/results_main_all.csv`
    - `outputs/slopemine_v2/ms_timefilter_v1/results_subsets_all.csv`
    - `outputs/slopemine_v2/ms_timefilter_v1/compare_to_lstm_and_timefilter.md`

## 哪些实验只作为中间验证，不进主表

以下内容只作为门槛、准备或辅助复核，不进入论文主结果表：

1. `baseline_v2` 的 shared ridge gate 实验
- 文件：
  - `outputs/slopemine_v2/baseline_v2/baseline_total_metrics_v2.csv`
  - `outputs/slopemine_v2/baseline_v2/baseline_gate_summary_v2.json`
- 作用：
  - 只用于早期判断是否进入机制阶段。

2. `experiment_plan_v1` 的 split 搜索与数据清单
- 文件：
  - `experiment_split_candidates.csv`
  - `split_selection_report.md`
  - `dataset_manifest_v1.md`
- 作用：
  - 只定义实验口径，不构成结果表。

3. `tuning_v1` 的窗口搜索
- 文件：
  - `window_tuning_metrics_v1.csv`
  - `window_tuning_summary_v1.md`
- 作用：
  - 只用于冻结正式 baseline 窗口，不写成模型有效性结论。

4. `timefilter_retune`
- 文件：
  - `outputs/slopemine_v2/timefilter_retune/timefilter_retune_results.csv`
  - `outputs/slopemine_v2/timefilter_retune/timefilter_retune_report.md`
- 作用：
  - 只用于证明“原始 TimeFilter 在更公平配置下仍然弱于当前主线”，可进附录，不进主表。

5. `blast_opt` 分支
- 文件：
  - `A4_blast_opt / C1_blast_opt / C2_blast_opt / C3_blast_opt`
- 作用：
  - 只作为对照分支，不作为当前正式主模型版本。

## 哪些结果可以写成正式结论

以下内容现在可以写成正式结论：

1. 正式数据与 split 已经冻结
- `dataset_manifest_v1` 与 `split_cand_01` 是后续正式实验唯一口径。
- `stable_background` 排除出主训练。
- 正式训练 patch 数固定为 `307`。

2. 第一轮 baseline 深度实验已经完成
- `formal_round1` 是 baseline 主结果来源。
- 在这轮结果中，最佳 `A4` 来自 `LSTM`。

3. 第三章正式天气分支在机制阶段已经有正式结果
- 在 `D 阶段`，`A2(main)` 明显优于 `baseline_naive_weather_concat(control)`。
- 因此可以正式写：
  - “在 `seq_len=96, pred_len=12` 的 MS-TimeFilter 机制阶段，正式 `Conv+Attn` 天气分支优于 naive weather concat 控制实现。”

4. real-coordinate V3 在机制阶段有正向增益
- 在 `D 阶段` 同 backbone 下，`A4(main)` 优于 `A3(main)`。
- 正式写法应限定为：
  - “在 MS-TimeFilter 机制阶段，real-coordinate V3 相对 V2 带来正向提升。”
- 不应夸大成“对所有 backbone 都已稳定验证”。

5. `PGGC + EDDR` 是当前机制主线
- `C1(main)` 优于 `A4(main)`。
- `C2(main)` 进一步优于 `C1(main)` 与 `baseline_raw_timefilter_a4_96(reference)`。
- 因此当前正式主模型版本可固定为：
  - `C2 main = PGGC + EDDR`

## 哪些结果不能写成正式结论

以下内容现在不能写成强结论：

1. 不能写“完整 MS-TimeFilter 已经超过 LSTM”
- 原因：
  - 当前 `C2 main(test_mae=0.108187)` 仍高于 `baseline_lstm_a4_96(test_mae=0.097959)`。

2. 不能写“EDDR 已明显实现复杂工况动态调节”
- 原因：
  - 当前 `C2` 的 sample-level gate 证据偏弱；
  - `blast_hours` 与 `non_event_hours` 的 gate 均值差异很小。

3. 不能写“PIR 已被正式验证有效”
- 原因：
  - 当前正式结果里 `C3 main` 退化于 `C2 main`。
  - 因此 `PIR` 只能保留为辅助约束候选项。

## 哪些模块需要降级表述

1. `PGGC`
- 可以写成：
  - “PGGC 已在正式机制实验中带来增益，是当前主模型的一部分。”

2. `EDDR`
- 可以写成：
  - “EDDR 使 `C2` 相对 `C1` 获得进一步增益，因此被纳入当前主模型。”
- 但应避免写成：
  - “EDDR 已经显著建立了强事件路由分化。”

3. `PIR`
- 当前必须降级。
- 推荐写法：
  - “PIR 当前仅作为辅助物理约束候选项；在现有正式结果下，尚未证明其能够稳定提升 `C2` 主线模型。”

## 对 `weather_branch_debug.md` 的统一解释

- `outputs/slopemine_v2/ms_timefilter_v1/weather_branch_debug.md` 属于 `D 阶段` 正式机制实验。
- 其中 `seq_len=96` 与早期 `formal_round1` 的 `seq_len=24` 不构成冲突。
- 原因是：
  - `formal_round1` 回答的是 baseline 问题；
  - `weather_branch_debug.md` 回答的是正式天气分支与机制模块问题。
- 二者阶段不同、模型不同、天气分支版本不同，不能放进同一张主结果表里直接比较。

## 写作时的统一引用规则

1. 写 baseline 主表时，只引用 `formal_round1`。
2. 写机制主表或当前主模型时，只引用 `ms_timefilter_v1`，并把 `C2 main` 作为当前正式主模型版本。
3. 写 split、patch 数、背景区处理时，只引用 `dataset_manifest_v1` 与 `experiment_split_plan_v1.csv`。
4. 写早期进入机制阶段的依据时，可以引用 `baseline_gate_summary_v2.json`，但必须显式标注它是早期 gate baseline。
5. 若写到 `PIR`，默认使用降级表述，除非后续 `pir_beta_sweep` 给出新的正式反证。
