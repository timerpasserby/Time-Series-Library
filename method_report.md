# 方法与模型报告

## 数据集
- **数据集**：露天煤矿边坡多源异构模拟数据 (`gemi_sim.py` 版)
- **特征规模**：1004维 (3气象 + 1爆破 + 1000雷达节点)
- **目标列**：`node_0` (已在 `run_models.sh` 中通过 `--target` 指定)
- **时间频率**：小时级 (H)
- **任务类型**：长期多变量预测 (Long-term forecasting)

## 评估模型
计划评估以下模型：
1. **TimesNet**：通过 2D 变分建模捕捉周期性任务。
2. **PatchTST**：基于 Patch 的通道独立 Transformer 模型。
3. **TimeFilter**：具有高效时空相关性捕捉能力的 Patch-Specific 模型。

## 评估指标
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)

## 当前状态
模型启动遇到 `patoolib` 依赖缺失问题，正在等待依赖项解决。
