# 当前正在做什么
整理雷达边坡训练数据，并尝试运行 TimesNet, TimeFilter, PatchTST 模型。

# 上次停在哪个位置
已完成数据预处理（合并气象、爆破与雷达位移数据），并在运行模型脚本时，由于环境缺失 `patoolib` 依赖导致 `run.py` 启动失败。

# 近期关键决定和原因
- **数据源选择**：选择了 `gemi_sim.py` 生成的数据，因为它具有更真实的物理空间拓扑关系，更适合时空序列预测任务。
- **数据对齐**：通过 `process_radar_data.py` 将气象、爆破离散事件与雷达小时位移对齐，并转换为单文件 `radar.csv` 以适配 `Dataset_Custom`。
- **环境切换**：根据用户要求切换至 `/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python` 环境。
