# SlopeMine V2 Architecture

## 文件职责

- `scripts/slopemine_v2/config_v2.toml`
  统一管理输入路径、五类工程区 polygon、背景区特殊处理策略、爆破区域代表坐标、patch 参数、窗口参数和基线参数。

- `scripts/slopemine_v2/build_patch_dataset_v2.py`
  负责 A-E：
  从原始监测表生成 `point_meta_v2.csv`、冻结分区规则文件、refined `patch_meta_v2_ps8.csv / patch_meta_v2_ps10.csv`、背景区替代方案、`patch_series_v2_ps10.csv`、`patch_blast_features_v3_ps10.csv`、基础张量和窗口清单，并导出空间图、爆破图和正式诊断包。

- `scripts/slopemine_v2/export_final_zone_assignment_figure.py`
  负责从最终 `point_meta_v2.csv` 和冻结分区边界导出论文版正式工程分区结果图，只用点位最终归属做主视觉，polygon 仅作为参考边界线。

- `scripts/slopemine_v2/generate_proxy_blast_ledger.py`
  负责基于爆破小时表、最终 patch centroid 和冻结分区规则生成三套代理爆破坐标台账，导出 proxy V3 特征、空间分布图、对比表和免责声明。

- `scripts/slopemine_v2/analyze_blast_v3_variants.py`
  负责对 `current / proxy` 两个来源生成 `V3a_truncated`、`V3b_max_local`、`V3c_local_contrast` 三类局部事件敏感版本，导出比较表、诊断报告和代表性热力图。

- `scripts/slopemine_v2/infer_internal_blast_events.py`
  负责完全忽略旧爆破输入，只基于 `patch_series_v2_ps10.csv` 的内部异常反演 `15-25` 个疑似爆破事件，输出 `Qe` 等效强度台账、候选小时排名、空间图、分区叠加图和内部诊断报告。

- `scripts/slopemine_v2/generate_experiment_plan_v1.py`
  负责基于冻结后的 `ps10` 数据资产自动搜索正式 train / val / test 候选方案，输出推荐 split、数据清单、实验矩阵和第四章补写项。

- `scripts/slopemine_v2/tune_window_config_v1.py`
  负责在正式 split 下扫描 `seq_len / pred_len` 组合，输出窗口搜索指标表、最佳基础配置 JSON 和简明摘要，用于替换旧整轴切分下的历史窗口推荐。

- `data_provider/slopemine_formal.py`
  负责正式 patch 深度实验的数据加载与窗口构造：把 720 小时整轴按 `drop_global_missing` 压缩为 712 个有效时间步，严格依据 `experiment_split_plan_v1.csv` 生成 `window_manifest_v1_ps10.csv`，并提供 A1-A4 特征拼接、subset 掩码和 dataset 类。

- `models/slopemine_formal_wrappers.py`
  负责把 patch-feature 输入包装成现有 `LSTM / TCN / DLinear / PatchTST / STGCN / TimeFilter` 可直接消费的统一接口，并提供 masked loss 与参数统计。

- `models/components/weather_encoder.py`
  负责正式天气分支编码：`Linear Projection + Multi-scale Conv1D + Temporal Attention + Time-block Pooling`，输出 block-level weather embedding、全局 weather embedding 和 attention 权重样本。

- `models/components/ms_timefilter_graph.py`
  负责 `PGGC / EDDR` 的稀疏图组件：构造 `A_s / A_t / A_prior`，按 chunked top-k 生成 `A_learned`，并提供 `PGGC` 一次图传播和 `EDDR` 三专家 softmax gate 路由。

- `models/ms_timefilter.py`
  负责正式 `MS-TimeFilter` 主模型：把 internal / weather / blast V2 / blast V3 组织成 patch-major、block-minor token，按实验阶段接入 `A2 / A3 / A4 / C1 / C2 / C3`，最后复用原始 `TimeFilter_Backbone` 做 temporal backbone。

- `data_provider/slopemine_patch_graph.py`
  负责从冻结的 `patch_tensor_base_v2_ps10.npz + patch_meta_v2_ps10_bg_excluded.csv` 构造 `307` 个 patch 的静态 8 邻接图，不再额外重采样成 `100` 个规则网格节点。

- `models/components/spatial_gnn.py`
  负责逐时间步共享参数的空间 GNN 编码器，当前提供 dense `GraphSAGE / GCN` 两种轻量实现，输入形状为 `[B, T, N, F]`，输出形状为 `[B, T, N, d_model]`。

- `models/spatial_gnn_timefilter.py`
  负责新的 `SpatialGNN + TimeFilter` 支线：先在 `307 patch` 静态图上做逐时间步空间编码，再把每个 patch 的时序送入原始 `TimeFilter_Backbone`。当前全长 `712` 小时输入下默认采用较大的 `patch_len=89`，避免 `307 x 712` 直接展开后产生过大的 `O(L^2)` 图学习开销。

- `scripts/slopemine_v2/run_formal_patch_experiments_v1.py`
  负责第一轮正式 patch 深度实验：在冻结的正式 split 与 `seq_len=24 / pred_len=12` 下运行六种模型的 `A1-A4`，输出 `results_main_A1_A4.csv`、`results_subsets_A1_A4.csv`、对比表、最佳模型、预测文件和图表。

- `scripts/slopemine_v2/run_ms_timefilter_experiments_v1.py`
  负责新的 `MS-TimeFilter` 正式机制实验：固定 `seq_len=96 / pred_len=12`，重跑 `A4-LSTM` 和原始 `TimeFilter` 参考，并运行 `A2 / A3 / A4 / C1 / C2 / C3` 主线与 `blast-hours` 优化分支，输出天气调试、图调试、gate 统计、PIR 报告和统一总表。

- `scripts/slopemine_v2/run_patch_baselines_v2.py`
  负责 F：
  读取基础张量和窗口清单，运行 `A1-A4` 共享权重 summary ridge 基线，输出总表、子集表、patch 误差热力图、代表性事件图和 `A4 vs A3` 门槛判断。

- `dataset/slopemine_v2/`
  保存可复用的数据资产：
  点位元数据、冻结工程分区规则、refined patch 元数据、patch 级时序、blast V3 特征、基础张量、邻接矩阵、窗口起点和窗口清单。

- `outputs/slopemine_v2/`
  保存面向汇报和检查的结果：
  工程分区图、refined patch 网格图、点密度热力图、爆破热力图、sigma 对比图、lag-correlation 图、正式诊断包、baseline 指标表和预测图。

## 调用关系

1. `config_v2.toml`
   提供原始数据路径、分区规则和实验参数。
2. `build_patch_dataset_v2.py`
   读取原始监测、天气、爆破输入；
   先冻结 polygon 规则并重建点位，再生成 refined patch 和 patch 时序，再生成 blast V3、正式诊断包，最后保存基础张量和窗口清单。
3. `run_patch_baselines_v2.py`
   读取 `patch_tensor_base_v2_ps10.npz`、`window_manifest_v2_ps10.csv` 和 `patch_meta_v2_ps10.csv`；
   自动只使用 `is_train_patch=1` 对应的 patch 顺序，按 `A1-A4` 组合特征并输出 baseline 结果。
4. `run_formal_patch_experiments_v1.py`
   读取 `patch_tensor_base_v2_ps10.npz`、`experiment_split_plan_v1.csv` 和 `patch_meta_v2_ps10_bg_excluded.csv`；
   先生成正式 `window_manifest_v1_ps10.csv`，再用统一 dataloader 和包装层跑 `LSTM / TCN / DLinear / PatchTST / STGCN / TimeFilter` 六种模型，最后导出正式结果表、子集指标和图表。
5. `run_ms_timefilter_experiments_v1.py`
   读取同一套冻结基础张量和正式 split；
   生成新的 `window_manifest_ms_timefilter_ps10_seq96_pred12.csv`，再分别构造旧 formal wrapper baseline 和多分支 `MS-TimeFilter` dataloader，最后统一输出 `A2~C3` 正式结果、参考对照、天气/图/gate/PIR 调试文件和决策报告。

6. `smoke_spatial_gnn_timefilter_v1.py`
   读取冻结后的 `307 patch` formal bundle 和 patch 图结构，直接对全长 `712` 小时序列做一次 `SpatialGNN + TimeFilter` 前向烟测，输出 `pred=[1, 12, 307]` 级别的 shape 校验报告，不触发训练。

## 关键设计决定

- 新工程区不复用旧标签：
  `zone_id_v2` 完全由 5 个冻结 polygon + priority order 决定，旧的 `area_id_old / area_id_seq_old` 只保留做对照；正式规则会落到 `engineering_zone_polygons_v2.json` 和 `zone_assignment_rule.md`。

- patch 只在 zone 内独立划分：
  每个 zone 单独计算 `patch_x_idx / patch_y_idx`，避免 patch 跨工程区。

- 稀疏 patch 先合并再建时序：
  点数小于 12 的 patch 会并到同 zone 最近的非稀疏 patch，减少后续时序和爆破特征里的空洞。

- 稳定背景区单独处理：
  正式主方案中，背景区只保留一个 merged background patch，并在 `patch_meta_v2_ps8.csv / patch_meta_v2_ps10.csv` 中标记 `is_train_patch=0`；同时额外导出完全排除背景区的 patch 版本，供主实验直接训练。

- 代理爆破台账与正式台账严格隔离：
  代理坐标不写回真实爆破输入，只额外生成 `proxy_blast_ledger_*` 和 `patch_blast_features_v3_ps10_proxy*`；三套版本只改抽样权重、最小间距和 sigma 范围，用于方法开发与敏感性分析。

- 原始连续累积型 V3 不直接进主实验：
  `current_v3 / proxy_v3` 虽保留做参照，但正式事件驱动建模应优先使用局部事件敏感版本；当前诊断结果优先推荐 `V3a_truncated_Rc60_Hc6`。

- 内部反演爆破事件与真实坐标严格区分：
  `infer_internal_blast_events.py` 输出的是内部异常驱动的疑似事件台账，`coordinate_source` 固定为 `internal_inferred`，`is_real_coordinate=0`，`Q` 明确表示 `Qe` 等效强度而非真实装药量。

- 正式实验 split 改按有效时间步冻结：
  `generate_experiment_plan_v1.py` 不再把 720 个自然小时直接等比例切开，而是先剔除 8 个 `is_global_missing=1` 的全局缺失小时，再在 712 个有效时间步上做连续切分；正式实验应以 `experiment_split_plan_v1.csv` 为准，而不是旧 `window_manifest_v2_ps10.csv` 的默认切分。

- 基础窗口参数也改按正式 split 选：
  `tune_window_config_v1.py` 在 712 个有效时间步和正式 train / val / test 切分上重新比较窗口组合；实验矩阵和第四章补写项优先引用这份正式窗口搜索结果，而不再直接沿用旧 `baseline_gate_summary_v2.json`。

- 正式输入不直接保存 12 份大窗口张量：
  先保存基础张量和 `window_manifest`，按需生成具体窗口，避免大量重复磁盘占用；基础张量只保留训练 patch，避免背景区参照 patch 混入主实验。

- baseline 先做 patch 独立，再谈机制：
  `A1-A4` 用共享权重 summary ridge，只比较 internal / weather / blast V2 / blast V3；
  只有 `A4` 稳定优于 `A3` 后，才应该继续到 `PGGC / EDDR / PIR`。

- 第一轮正式深度实验已落地，但 `A4` 尚未稳定压过 `A3`：
  当前正式六模型首轮结果里，按 `val_mae` 选最佳时 `A1=TCN`、`A2=LSTM`、`A3=TCN`、`A4=LSTM`；按 `test_mae` 看四组实验最佳都落在 `LSTM`，其中 `A3` 略优于 `A4`，因此后续接 `PGGC / EDDR / PIR` 前还需要继续做特征或训练策略迭代。

- `PatchTST` 已修成非原地归一化：
  为了兼容正式 patch 包装层的反向传播，`models/PatchTST.py` 中涉及输入标准化的 `/=` 被替换成非原地写法，避免梯度图因 in-place 修改而报错。

- `MS-TimeFilter` 与旧 formal wrapper 并存：
  旧 `formal_round1` 链路继续服务 `LSTM / TCN / DLinear / PatchTST / STGCN / TimeFilter` 参考基线；新的机制实验不再把 weather / blast 直接拼到 patch 特征后再压成单标量，而是通过多分支 token 融合接入 `TimeFilter_Backbone`。

- `SpatialGNN + TimeFilter` 是独立探索支线：
  这条线当前固定以冻结后的 `307 patch` 为图节点，并按 patch lattice 的 8 邻接构造静态空间图；不再把节点强行降成 `100` 个规则网格，以免和现有正式主实验口径混用。

- `PGGC / EDDR / PIR` 现在已经有正式机制结果：
  `run_ms_timefilter_experiments_v1.py` 会在冻结后的 `split_cand_01`、`seq_len=96`、`pred_len=12` 上统一跑 `A2 / A3 / A4 / C1 / C2 / C3`；当前主模型固定为 `C2 main = PGGC + EDDR`，`PIR` 对应的 `C3` 暂未带来稳定收益，因此默认降级为辅助约束。

- 第四章论文图件有独立导出入口：
  `generate_chapter4_figures_v1.py` 只读取冻结后的正式结果文件，不重新训练模型；它统一导出总体性能图、子集对比图、典型事件窗口、patch 级误差热力图、PGGC 三联图、EDDR 路由图和平滑性对比图到 `outputs/slopemine_v2/chapter4_figures_v1`。
