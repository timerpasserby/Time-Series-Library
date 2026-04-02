# weather_branch_debug

- 正式天气分支已使用 `Linear Projection + Multi-scale Conv1D + Temporal Attention + Time-block Pooling`。
- 这里的注意力权重来自一个代表性样本的 encoder 96 小时窗口。

## Tensor Shapes

- `weather_seq` = `(1, 96, 4)`
- `projected` = `(1, 96, 32)`
- `conv_k3` = `(1, 96, 32)`
- `conv_k5` = `(1, 96, 32)`
- `conv_k7` = `(1, 96, 32)`
- `fused` = `(1, 96, 32)`
- `weather_block_embed` = `(1, 12, 32)`
- `weather_global_embed` = `(1, 32)`
- `attention_weights` = `(1, 96)`

## Top Attention Hours

| timestamp | attention_weight |
| --- | ---: |
| 2024-06-23T09:00:00 | 0.011736 |
| 2024-06-23T08:00:00 | 0.011672 |
| 2024-06-23T05:00:00 | 0.011636 |
| 2024-06-23T06:00:00 | 0.011538 |
| 2024-06-23T07:00:00 | 0.011512 |
| 2024-06-23T04:00:00 | 0.011478 |
| 2024-06-26T07:00:00 | 0.011469 |
| 2024-06-23T10:00:00 | 0.011422 |
