# 边坡位移预测实验链路打通记录

## 1. 任务目标

本次工作的目标不是追求最终最优精度，而是先完成硕士论文第四章 4.3 “对比模型与主实验结果”的首轮可运行实验链路验证。优先级如下：

1. 数据能够被 Time-Series-Library (TSLib) 正确读取。
2. 训练、验证、测试流程能够完整跑通。
3. `DLinear`、`PatchTST`、`TimeFilter` 在统一框架下完成首轮实验。
4. 同时补充 `MICN`、`LSTM`、`TCN`、`STGCN` 的可运行基线。
5. 训练轮数统一设置为 `train_epochs=1`，用于链路验证。

## 2. 原始数据与问题定位

原始数据文件：

- `/Users/dc/Z研究生/eedsProject/device_10001_sorted.csv`

原始表头示例：

```csv
grid_x,grid_y,deformation,speed,acceleration,report_time,device_id,area_id,area_id_seq
162,112,0.5833333,-0.0007208333,-0.00011541667,2024-06-01 00:00:00,10001,2,2
```

原始数据不是 TSLib `custom` 数据集直接可用的“宽表”，而是“长表”：

- 同一个 `report_time` 下包含多个 `(grid_x, grid_y)` 监测点。
- TSLib `custom` 数据集要求格式为：
  - 第一列必须是时间列 `date`
  - 每一行代表一个时间步
  - 后续列全部为数值特征
  - 目标列需要存在于这些数值列中

因此，实验前必须先进行数据适配。

## 3. 数据适配方案

### 3.1 采用的方案

本次优先采用“聚合宽表”方案，以保证最小可运行：

- 按 `report_time` 聚合；
- 目标列使用全区域聚合后的 `deformation` 均值；
- 特征列使用：
  - `speed`
  - `acceleration`
  - `deformation_std`
  - `speed_std`
  - `acceleration_std`
  - `deformation_max`（原 `grid_count`，因常量无信息量，替换为最大变形量）
- 最终输出列顺序为：

```csv
date,speed,acceleration,deformation_std,speed_std,acceleration_std,deformation_max,deformation
```

这样可直接配置为：

- `data=custom`
- `features=MS`
- `target=deformation`
- `enc_in=7`
- `dec_in=7`
- `c_out=7`

其中：

- `MS` 表示多变量输入、单变量预测；
- 由于 `Dataset_Custom` 会把目标列放到最后，TSLib 在 `MS` 模式下会只对最后一个通道计算目标损失。

### 3.2 生成脚本

新增脚本：

- `/Users/dc/Z研究生/time_series/Time-Series-Library/scripts/slopemine/prepare_tslib_csv.py`

实际执行命令：

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python \
/Users/dc/Z研究生/time_series/Time-Series-Library/scripts/slopemine/prepare_tslib_csv.py \
  --input "/Users/dc/Z研究生/eedsProject/device_10001_sorted.csv" \
  --output "/Users/dc/Z研究生/time_series/Time-Series-Library/dataset/slopemine/slopemine_aggregate.csv" \
  --mode aggregate
```

### 3.3 处理结果

脚本输出：

- 处理原始行数：`13,930,992`
- 生成聚合后小时级样本数：`712`

生成文件：

- `/Users/dc/Z研究生/time_series/Time-Series-Library/dataset/slopemine/slopemine_aggregate.csv`

数据检查结果：

- 行数：`712`
- 列数：`8`
- 时间范围：`2024-06-01 00:00:00` 至 `2024-06-30 23:00:00`
- 缺失值：`0`

## 4. 数据完整性说明

理论上 2024 年 6 月整月小时级应有 `720` 个时间步，但当前原始数据实际只覆盖了 `712` 个小时，缺失 `8` 个时间点。

缺失的时间戳如下：

```text
2024-06-06 05:00:00
2024-06-06 06:00:00
2024-06-06 07:00:00
2024-06-06 08:00:00
2024-06-14 00:00:00
2024-06-14 01:00:00
2024-06-14 05:00:00
2024-06-20 08:00:00
```

这意味着：

1. 当前链路验证用数据是“真实缺测后的 712 步数据”。
2. 如果后续论文要求严格 720 步，可在正式实验前增加“缺失时间补齐”步骤。
3. 当前结果适合用于“实验框架已跑通”的阶段性说明。

## 5. 实验环境

本次实际验证使用环境：

- Python: `3.9.0`
- pandas: `1.2.4`
- torch: `2.0.0`
- sklearn: `1.2.2`

解释器路径：

- `/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python`

硬件情况：

- `cuda_available=False`
- `mps_available=False`

因此本次验证为 CPU 运行。

## 6. 为保证跑通所做的最小工程修改

为避免无关任务依赖阻塞当前实验，对仓库做了最小修改：

### 6.1 PatchTST 支持命令行传入 patch 参数

修改文件：

- `/Users/dc/Z研究生/time_series/Time-Series-Library/run.py`
- `/Users/dc/Z研究生/time_series/Time-Series-Library/models/PatchTST.py`
- `/Users/dc/Z研究生/time_series/Time-Series-Library/utils/print_args.py`

修改内容：

- 新增 `--stride`
- `PatchTST` 真实读取 `--patch_len` 和 `--stride`

### 6.2 新增可运行基线模型

新增文件：

- `/Users/dc/Z研究生/time_series/Time-Series-Library/models/LSTM.py`
- `/Users/dc/Z研究生/time_series/Time-Series-Library/models/TCN.py`
- `/Users/dc/Z研究生/time_series/Time-Series-Library/models/STGCN.py`

### 6.3 处理不必要的全局依赖阻塞

修改文件：

- `/Users/dc/Z研究生/time_series/Time-Series-Library/data_provider/data_loader.py`
- `/Users/dc/Z研究生/time_series/Time-Series-Library/layers/SelfAttention_Family.py`

处理的问题：

- `patoolib`：仅 M4 任务需要
- `sktime`：仅分类任务需要
- `datasets` / `huggingface_hub`：仅远程下载数据时需要
- `reformer_pytorch`：仅 Reformer 模型需要

修改后策略：

- 当任务确实需要这些依赖时再报错；
- 对当前 `long_term_forecast + custom` 实验不再形成阻塞。

## 7. 最小可运行实验设置

统一设置如下：

- `task_name=long_term_forecast`
- `data=custom`
- `root_path=./dataset/slopemine`
- `data_path=slopemine_aggregate.csv`
- `features=MS`
- `target=deformation`
- `freq=h`
- `seq_len=96`
- `label_len=96`
- `pred_len=1`
- `train_epochs=1`
- `batch_size=32`
- `itr=1`
- `num_workers=0`

统一输入输出维度：

- `enc_in=7`
- `dec_in=7`
- `c_out=7`

## 8. 实际执行的批量命令

批量脚本：

- `/Users/dc/Z研究生/time_series/Time-Series-Library/scripts/slopemine/run_slopemine_baselines.sh`

实际执行命令：

```bash
cd /Users/dc/Z研究生/time_series/Time-Series-Library
PYTHON_BIN=/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python \
ROOT_PATH=./dataset/slopemine \
DATA_PATH=slopemine_aggregate.csv \
PRED_LENS="1" \
EXTRA_ARGS="--no_use_gpu" \
bash scripts/slopemine/run_slopemine_baselines.sh
```

## 9. 实际跑通模型与结果

下表为 `pred_len=1`、`train_epochs=1`、CPU 环境下的首轮真实运行结果：

| 模型 | MAE | MSE | RMSE | 说明 |
|---|---:|---:|---:|---|
| DLinear | 1.019309 | 10.939543 | 3.307498 | 已跑通 |
| PatchTST | 1.632762 | 13.601808 | 3.688063 | 已跑通 |
| TimeFilter | 1.252606 | 12.012307 | 3.465878 | 已跑通 |
| MICN | 41.250988 | 2858.207275 | 53.462204 | 已跑通，但当前设置下数值很差 |
| LSTM | 1.200824 | 10.660291 | 3.265010 | 已跑通 |
| TCN | 1.485019 | 12.562633 | 3.544380 | 已跑通 |
| STGCN | 1.180480 | 10.706397 | 3.272063 | 已跑通 |

说明：

1. 当前所有结果仅用于“实验链路验证”；
2. 由于仅训练 `1` 个 epoch，不能代表模型最终性能；
3. `MICN` 在当前聚合数据和极简设置下表现明显异常，后续正式实验应单独调参或重新检查其适配性；
4. `DLinear / LSTM / STGCN` 在首轮验证中数值相对更稳定。

## 10. 输出目录

本次实验结果已保存至：

- 模型 checkpoint：`/Users/dc/Z研究生/time_series/Time-Series-Library/checkpoints`
- 预测结果与指标：`/Users/dc/Z研究生/time_series/Time-Series-Library/results`
- 汇总文本：`/Users/dc/Z研究生/time_series/Time-Series-Library/result_long_term_forecast.txt`

每个实验目录下包含：

- `metrics.npy`
- `pred.npy`
- `true.npy`

## 11. 现阶段可直接用于论文的表述建议

可以将当前阶段描述为：

> 为保证第四章主实验后续能够在服务器端稳定开展，本文首先基于 Time-Series-Library 完成了边坡位移预测任务的数据适配与实验链路验证。针对原始监测长表数据，构建了符合 TSLib `custom` 数据集要求的小时级宽表，并在统一任务定义下完成了 DLinear、PatchTST、TimeFilter、MICN、LSTM、TCN 和 STGCN 等模型的首轮运行验证。首轮实验统一设置 `seq_len=96`、`pred_len=1`、`train_epochs=1`，重点验证数据读入、模型训练、验证、测试以及结果保存流程的完整性。实验结果表明，所构建的数据适配方案和统一实验框架已能够稳定支撑后续正式论文实验。

## 12. 后续建议

下一步建议按以下顺序推进：

1. 先决定是否补齐缺失的 8 个小时数据，使样本严格达到 720 步；
2. 将 `pred_len` 从 `1` 扩展到 `1/3/6/12`；
3. 保持训练/验证/测试划分一致，统一各模型对比设置；
4. 将 `train_epochs` 从 `1` 提升到正式训练轮数，如 `20` 或 `30`；
5. 将服务器批量实验输出统一管理，方便后续填写论文第四章表格。

