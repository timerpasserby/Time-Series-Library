# experiment_matrix_v1

- 正式 split 来源：`experiment_split_plan_v1.csv`。
- 默认 patch 粒度：`patch_size=10`；正式主训练 patch 数：307；`stable_background` 不参与主训练。
- 现有基线门槛结果：A4 相对 A3 已满足继续进入机制阶段；当前推荐基础配置为 `seq_len=24, pred_len=12`（基于正式 split 的 A4 验证集窗口搜索结果。）。

| ID | 名称 | 正式输入 | patch 方案 | 目的 | 运行备注 |
| --- | --- | --- | --- | --- | --- |
| A0 | aggregate baseline（仅参考） | 全局聚合内部统计 + 正式 split | 非主实验，不做 patch 训练 | 给出全局参考下界 | 只做参考，不参与主结论对比 |
| A1 | patch internal only | `patch_tensor_base_v2_ps10.npz` 内部特征 | zone-constrained, ps10, bg excluded | 建立 patch 主实验基线 | 第一轮先跑 |
| A2 | A1 + weather | A1 + `weather` | zone-constrained, ps10, bg excluded | 检验天气外生增益 | 第二轮跑 |
| A3 | A2 + blast V2 | A2 + `blast_v2` | zone-constrained, ps10, bg excluded | 检验真实爆破小时级输入增益 | 第三轮跑 |
| A4 | A2 + blast V3 real-coordinate | A2 + `patch_blast_features_v3_ps10.csv` | zone-constrained, ps10, bg excluded | 检验真实坐标 patch 级爆破输入增益 | 第四轮跑，作为机制阶段入口 |
| B1 | no-zone patch vs zone-constrained patch | 与 A4 同输入口径 | `no-zone patch` 对比 `zone-constrained patch` | 检验工程分区约束是否必要 | 先复用正式 split，再补 no-zone patch 资产 |
| B2 | patch_size=8 vs 10 | 与 A4 同输入口径 | `ps8` 对比 `ps10` | 检验空间粒度敏感性 | 同一正式 split 下比较 |
| C1 | A4 + PGGC | A4 + PGGC | zone-constrained, ps10, bg excluded | 引入空间图结构 | 允许进入 |
| C2 | C1 + EDDR | C1 + EDDR | zone-constrained, ps10, bg excluded | 引入事件驱动动态增强 | 允许进入 |
| C3 | C2 + PIR | C2 + PIR | zone-constrained, ps10, bg excluded | 引入 patch 重要性重标定 | 允许进入 |

## 正式运行顺序

- 第一轮：A0（可选参考） -> A1 -> A2 -> A3 -> A4。
- 第二轮：B1、B2 作为结构对照与粒度敏感性实验。
- 第三轮：若 A4 稳定优于 A3，则进入 C1 -> C2 -> C3。
