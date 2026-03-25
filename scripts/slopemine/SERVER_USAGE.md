# 服务器批量实验使用说明

## 1. 数据文件准备

将以下文件上传到服务器对应目录：

```
Time-Series-Library/
├── dataset/slopemine/
│   ├── slopemine_aggregate.csv      # 聚合宽表（712行×8列）
│   ├── slopemine_agg_weather.csv    # 含天气（712行×13列）
│   └── slopemine_full.csv           # 含天气+爆破（712行×19列）
└── scripts/slopemine/
    ├── run_epochs30.sh              # 天气+爆破实验
    ├── run_weather_baseline.sh      # 仅天气基线（A/B对比用）
    └── summarize_results.py         # 结果汇总
```

## 2. 运行实验

### 方式 A：仅跑天气+爆破实验（推荐先跑）

```bash
cd /path/to/Time-Series-Library
PYTHON_BIN=python bash scripts/slopemine/run_epochs30.sh
```

### 方式 B：A/B 对比（天气 vs 天气+爆破）

```bash
# 终端1：天气+爆破
PYTHON_BIN=python bash scripts/slopemine/run_epochs30.sh

# 终端2：仅天气基线（同时跑）
PYTHON_BIN=python bash scripts/slopemine/run_weather_baseline.sh
```

### 方式 C：服务器 GPU 运行

```bash
PYTHON_BIN=python USE_GPU=1 bash scripts/slopemine/run_epochs30.sh
```

### 方式 D：sbatch 提交（SLURM 集群）

```bash
sbatch --job-name=slopemine --gres=gpu:1 --cpus-per-task=4 \
  --mem=32G --time=12:00:00 \
  --wrap="PYTHON_BIN=python USE_GPU=1 bash scripts/slopemine/run_epochs30.sh"
```

## 3. 实验配置

| 参数 | 值 |
|---|---|
| 数据 | slopemine_full.csv（17 特征） |
| seq_len | 24 |
| label_len | 24 |
| pred_len | 1, 3, 6, 12 |
| epochs | 30 |
| batch_size | 32 |
| learning_rate | 0.0001 |
| models | DLinear, PatchTST, TimeFilter, LSTM, TCN, STGCN |
| 总实验数 | 6 模型 × 4 pred_len = 24 组 |

## 4. 查看结果

### 实时查看日志

```bash
tail -f scripts/slopemine/logs/epochs30_*.log
```

### 汇总结果

```bash
python scripts/slopemine/summarize_results.py ./results
```

输出示例：
```
======================================================================
Feature set: full
======================================================================
Model        PredLen  Epochs          MSE          MAE
------------ ------- ------ ------------ ------------
DLinear            1     30       8.xxxx       1.xxxx
LSTM               1     30       7.xxxx       1.xxxx
...
```

### A/B 对比

如果同时跑了两个实验，汇总脚本会自动输出对比：

```
Model    pl   MSE(w)    MSE(w+b)   ΔMSE%    MAE(w)    MAE(w+b)   ΔMAE%
DLinear   6   11.2490   11.2490    +0.0%    1.1500    1.1500     +0.0%
STGCN     6   11.3300   11.1020    -2.0%    1.4980    1.2670    -15.4%
TimeFilter 6  13.6480   13.0470    -4.4%    1.5320    1.4140     -7.7%
```

## 5. 预计时间

| 环境 | 单组实验 | 全部 24 组 |
|---|---|---|
| CPU (M1 Mac) | ~1 min | ~25 min |
| GPU (V100) | ~10 sec | ~5 min |
| GPU (A100) | ~5 sec | ~3 min |

## 6. 输出文件

- **Checkpoint**: `checkpoints/slopemine_MODEL_sl24_plYY_e30_full/`
- **Results**: `results/slopemine_MODEL_sl24_plYY_e30_full/`
  - `metrics.npy` — MSE/MAE 指标
  - `pred.npy` — 预测值
  - `true.npy` — 真实值
- **Logs**: `scripts/slopemine/logs/epochs30_*.log`
