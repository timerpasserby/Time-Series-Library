# 当前正在做什么
`run_models.sh` 的 TimeFilter 配置已去掉 `target`，改为直接预测全部节点的未来位移。

# 上次停在哪个位置
2026-04-16：`run_models.sh` 已增加日志重定向到 `logs/run_models_*.log`，方便保存完整训练过程。

# 近期关键决定和原因
- **解释器固定**：`run_models.sh` 已切到 `/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python`。
- **日志留存**：脚本输出通过 `tee` 写入时间戳日志文件，便于回看训练过程。
- **全节点预测**：TimeFilter 使用 `features=M`，`enc_in/c_out=1000`，不需要 `target`。
- **MPS 退回 CPU**：当前 macOS 版本不支持 MPS，TimeFilter 改走 CPU 以避免 `RuntimeError: Invalid buffer size` 和设备初始化失败。
- **TimeFilter 缩参**：将 `patch_len` 提到 `96` 并把批量、宽度、层数降下来，先确保 1000 节点场景能跑通。
