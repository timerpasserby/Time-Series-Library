# 当前正在做什么
整理雷达边坡训练数据，并解决多变量预测中的列名对齐问题。

# 上次停在哪个位置
已修复 `run.py` 启动时的 `ValueError: list.remove(x): x not in list` 错误（通过在 `run_models.sh` 中明确指定 `--target node_0`）。目前仍受限于环境缺失 `patoolib` 依赖，建议在环境中执行 `pip install patoolib sktime`。

# 近期关键决定和原因
- **修复目标列配置**：在 `run_models.sh` 中补充了 `--target node_0`。原因是 `Dataset_Custom` 默认寻找 `OT` 列，而我们在预处理脚本中将主预测目标命名为 `node_0` 并移至了最后一列。
- **数据源选择**：选择了 `gemi_sim.py` 生成的数据，因为它具有更真实的物理空间拓扑关系，更适合时空序列预测任务。
