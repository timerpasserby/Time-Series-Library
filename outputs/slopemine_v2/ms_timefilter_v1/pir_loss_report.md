# pir_loss_report

- `C3` 使用 `L_total = L_pred + beta * L_phy`，其中 `alpha_v=1.0`、`alpha_a=0.5`、`beta=0.05`。
- 当前 open-source TimeFilter 只稳定暴露 `moe_loss`，`loss_dyn` / `loss_imp` 接口保留为日志列并填 `NA`。
- 动态一致性图：`/root/autodl-tmp/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/pir_dynamics_C3.png`。
- C2 vs C3 典型事件窗口图：`/root/autodl-tmp/Time-Series-Library/outputs/slopemine_v2/ms_timefilter_v1/figures/typical_event_windows_C2_vs_C3.png`。
