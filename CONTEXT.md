# 当前正在做什么
完善 `gemi_sim.py` 边坡仿真器，补充加速破坏阶段物理本构和运动学特征提取；同时扩充 `run_models.sh` 整理雷达边坡训练数据，仿真复杂物理特性（加速阶段、运动学特征、3D地形）。

# 上次停在哪个位置
2026-04-16：已完成以下工作：
1. `run_models.sh` 新增 DLinear、Informer、Autoformer、TimeMixer，统一配置 `--gpu_type mps`。
2. `gemi_sim.py` 注入斋藤法加速阶段逻辑及 TTF 标签。
3. `gemi_sim.py` 实现基于 S-G 滤波的运动学特征计算（速度/加速度）。
4. `gemi_sim.py` 增加基于坡向物理的 3D 地形高度（Z轴）生成。

# 近期关键决定和原因
- **修复目标列配置**：在 `run_models.sh` 中补充了 `--target node_0`。原因是 `Dataset_Custom` 默认寻找 `OT` 列，而我们在预处理脚本中将主预测目标命名为 `node_0` 并移至了最后一列。
- **数据源选择**：选择了 `gemi_sim.py` 生成的数据，因为它具有更真实的物理空间拓扑关系，更适合时空序列预测任务。
- **MPS 设备配置**：全部模型统一添加 `--gpu_type mps`，适配 Apple Silicon 本地训练，避免回退 CPU 导致速度过慢。
- **DLinear / TimeMixer 的特殊参数**：这两个模型不使用 decoder，`label_len=0`；TimeMixer 额外需要 `--down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1 --decomp_method moving_avg`。
- **斋藤法加速阶段**：采用 `v = C / (T_f - t + 0.1)` 近似本构，记录阶段标签和 TTF。
- **S-G 滤波求导**：位移、速度均经过 Savitzky-Golay 滤波降噪，求取高精度运动学特征。
- **3D 地形模拟**：引入 `grid_z`，基于西高东低（坡角 35°）物理模型生成，为时空模型提供空间三维特征。
