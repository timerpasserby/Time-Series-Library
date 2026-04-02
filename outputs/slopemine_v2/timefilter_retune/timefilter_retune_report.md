# timefilter_retune_report

## Run Context

- 本轮只复核 `TimeFilter`，实验对象固定为正式 `A4` 数据版本。
- split 固定为 `split_cand_01`，`patch_size=10`，`stable_background` 继续排除。
- 扫描网格：`seq_len=[24, 48, 96]`，`learning_rate=[0.0001, 0.0003, 0.0005]`，`dropout=[0.1, 0.2, 0.3]`。
- 其余设置固定：`pred_len=12`、`epochs=8`、`batch_size=8`、`weight_decay=1e-05`、`grad_clip=5.0`。
- 训练设备：`cuda`。
- 默认参考配置来自 `formal_round1`：`seq_len=24, learning_rate=1e-3, dropout=0.1`。

## Split Check

| seq_len | train_samples | val_samples | test_samples |
| ---: | ---: | ---: | ---: |
| 24 | 421 | 117 | 117 |
| 48 | 397 | 117 | 117 |
| 96 | 349 | 117 | 117 |

- 所有配置都使用同一正式 split 源和同一 `A4` 特征定义；`val/test` 样本数固定不变。
- `train_samples` 会随着 `seq_len` 变长而自然减少，这是因果窗口在不做左侧 padding 的前提下的正式行为，不属于 split 漂移。

## Reference Comparison

- 当前正式默认 `A4-TimeFilter`：`val_mae=0.046964`，`test_mae=0.161289`，`blast_test_mae=0.101394`。
- 当前正式 `A4-LSTM`：`val_mae=0.044104`，`test_mae=0.100700`，`blast_test_mae=0.038569`。
- 默认 `TimeFilter` 在“27 组 retune + 1 个正式默认参考”的扩展排序里：`val_mae` 排名 `6/28`，`test_mae` 排名 `5/28`。

## Best Configs

- 按 `val_mae` 最优：`seq_len=48`、`learning_rate=3e-04`、`dropout=0.3`，`val_mae=0.044730`，`test_mae=0.165991`。
- 按 `test_mae` 最优：`seq_len=96`、`learning_rate=1e-04`、`dropout=0.1`，`val_mae=0.057540`，`test_mae=0.159156`，`blast_test_mae=0.079194`。

- 相对默认 `TimeFilter` 的 `test_mae` 改善：`1.32%`。
- 相对默认 `TimeFilter` 的 `blast_test_mae` 改善：`21.89%`。
- 相对 `A4-LSTM` 的 `test_mae` 差距：`58.05%`。
- 相对 `A4-LSTM` 的 `blast_test_mae` 差距：`105.33%`。

## Ranking by val_mae

| rank | seq_len | learning_rate | dropout | train_samples | val_samples | test_samples | val_mae |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48 | 3e-04 | 0.3 | 397 | 117 | 117 | 4.473012e-02 |
| 2 | 48 | 5e-04 | 0.1 | 397 | 117 | 117 | 4.599115e-02 |
| 3 | 96 | 5e-04 | 0.2 | 349 | 117 | 117 | 4.609941e-02 |
| 4 | 96 | 5e-04 | 0.3 | 349 | 117 | 117 | 4.631365e-02 |
| 5 | 96 | 3e-04 | 0.2 | 349 | 117 | 117 | 4.636173e-02 |

## Ranking by test_mae

| rank | seq_len | learning_rate | dropout | train_samples | val_samples | test_samples | test_mae |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96 | 1e-04 | 0.1 | 349 | 117 | 117 | 1.591561e-01 |
| 2 | 48 | 5e-04 | 0.3 | 397 | 117 | 117 | 1.601500e-01 |
| 3 | 96 | 3e-04 | 0.1 | 349 | 117 | 117 | 1.604303e-01 |
| 4 | 96 | 5e-04 | 0.2 | 349 | 117 | 117 | 1.610176e-01 |
| 5 | 96 | 5e-04 | 0.1 | 349 | 117 | 117 | 1.648151e-01 |

## Final Answer

- 是否明显优于当前正式默认 `TimeFilter`：否。
- 是否能接近或超过当前正式 `A4-LSTM`：否。
- 综合判断：`更偏 backbone 不适配`。

## Decision Rule

- 本报告把“默认配置存在明显不公平”定义为：`test_mae` 改善至少 `5%`，`blast_test_mae` 改善至少 `5%`，且最优配置与 `A4-LSTM` 的 `test_mae` 或 `blast_test_mae` 差距不超过 `15%`。
- 若上述条件不满足，则判定更偏 `backbone` 不适配，而不是单纯默认配置不公平。
