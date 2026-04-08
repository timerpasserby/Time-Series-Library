# server_run_pir_beta_sweep_v1

## Purpose

- 当前本地 CPU 过慢，`PIR beta sweep` 建议切到服务器执行。
- 本说明只针对当前待完成任务：
  - `pir_beta_sweep.csv`
  - `pir_beta_report.md`

## Files Already Prepared

- 同步脚本：
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/scripts/slopemine_v2/server_sync_minimal.sh`
- 服务器启动脚本：
  - `/Users/dc/Z研究生/time_series/Time-Series-Library/scripts/slopemine_v2/server_launch_pir_beta_sweep_v1.sh`

## Recommended Steps

1. 本地同步到服务器

```bash
cd /Users/dc/Z研究生/time_series/Time-Series-Library
bash scripts/slopemine_v2/server_sync_minimal.sh
```

默认目标：

- host: `root@connect.cqa1.seetacloud.com`
- port: `43537`
- remote root: `/root/autodl-tmp/Time-Series-Library`

如果你想改远端目录，可以这样：

```bash
REMOTE_ROOT=/root/Time-Series-Library bash scripts/slopemine_v2/server_sync_minimal.sh
```

2. 登录服务器

```bash
ssh -p 43537 root@connect.cqa1.seetacloud.com
```

3. 进入项目目录并检查 Python

```bash
cd /root/autodl-tmp/Time-Series-Library
which python
python -c "import torch; print(torch.__version__)"
```

4. 后台运行 `PIR beta sweep`

```bash
cd /root/autodl-tmp/Time-Series-Library
chmod +x scripts/slopemine_v2/server_launch_pir_beta_sweep_v1.sh
nohup bash scripts/slopemine_v2/server_launch_pir_beta_sweep_v1.sh > logs/pir_beta_sweep_v1.log 2>&1 &
```

如果服务器上需要指定 Python：

```bash
cd /root/autodl-tmp/Time-Series-Library
PYTHON_BIN=/opt/conda/bin/python nohup bash scripts/slopemine_v2/server_launch_pir_beta_sweep_v1.sh > logs/pir_beta_sweep_v1.log 2>&1 &
```

如果你想把轻量扫描再压短一点：

```bash
cd /root/autodl-tmp/Time-Series-Library
MAX_EPOCHS=8 PATIENCE=2 nohup bash scripts/slopemine_v2/server_launch_pir_beta_sweep_v1.sh > logs/pir_beta_sweep_v1.log 2>&1 &
```

5. 查看日志

```bash
cd /root/autodl-tmp/Time-Series-Library
tail -f logs/pir_beta_sweep_v1.log
```

6. 结果位置

```text
outputs/slopemine_v2/ms_timefilter_v1/pir_beta_sweep.csv
outputs/slopemine_v2/ms_timefilter_v1/pir_beta_report.md
outputs/slopemine_v2/ms_timefilter_v1/pir_beta_sweep_artifacts/
```

7. 拉回结果

```bash
cd /Users/dc/Z研究生/time_series/Time-Series-Library
rsync -av --progress -e "ssh -p 43537" \
  root@connect.cqa1.seetacloud.com:/root/autodl-tmp/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/pir_beta_sweep.csv \
  root@connect.cqa1.seetacloud.com:/root/autodl-tmp/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/pir_beta_report.md \
  root@connect.cqa1.seetacloud.com:/root/autodl-tmp/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/pir_beta_sweep_artifacts \
  outputs/slopemine_v2/ms_timefilter_v1/
```

## Notes

- 当前仓库里 `C2 vs LSTM` 的分析文件已经在本地落盘，不需要再重跑。
- 这次上服务器的核心目标只是把 `PIR beta sweep` 跑完。
