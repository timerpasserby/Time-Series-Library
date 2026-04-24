# 数据处理报告

## 处理过程
1. **取消合并**：`process_radar_data.py` 不再执行 weather / blast / radar 的 `pd.merge`。
2. **独立导出**：分别输出 `radar.csv`、`weather.csv`、`blast.csv` 三个文件。
3. **雷达纯化**：`radar.csv` 仅保留 `date + node_0...node_999`（1000节点）。

## 结果文件
- [radar.csv](/Users/dc/Z研究生/time_series/Time-Series-Library/dataset/radar_sim/radar.csv)
- [weather.csv](/Users/dc/Z研究生/time_series/Time-Series-Library/dataset/radar_sim/weather.csv)
- [blast.csv](/Users/dc/Z研究生/time_series/Time-Series-Library/dataset/radar_sim/blast.csv)
- `sim_stage_labels.csv`: 包含破坏阶段标签 (0:等速, 1:加速, 2:破坏) 与 TTF (剩余破坏时间)。
- `sim_radar_features_full.csv`: 包含位移、速度、加速度的多通道长表数据。
- `sim_nodes_static.csv`: 包含节点的三维坐标 (X, Y, Z) 及敏感度。

## 本轮仿真结果
- `data/radar/sim_radar_hourly_displacement.csv` 已重生成，并补回 `report_time` 时间列。
- 新单次位移统计：全局均值约 `0.489`，99% 分位约 `3.85`，最大值约 `31.41`。
- 强降雨时段平均单次位移约 `2.77`，无雨时段约 `0.42`。
- 爆破时刻的节点最大位移均值约 `10.25`，非爆破时刻约 `1.38`。
