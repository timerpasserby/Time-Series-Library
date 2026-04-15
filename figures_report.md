# 数据处理报告

## 处理过程
1. **数据合并**：使用 `process_radar_data.py` 将 `sim_weather.csv` (气象), `sim_blast_logs.csv` (爆破) 与 `sim_radar_hourly_displacement.csv` (位移) 按照时间轴对齐。
2. **特征工程**：将爆破强度聚合为小时频率。
3. **格式转换**：生成 `dataset/radar_sim/radar.csv`，其中包含：
    - `date`: 时间戳
    - `temperature`, `humidity`, `rainfall`: 气象特征
    - `blast_intensity`: 爆破干扰特征
    - `node_0 ... node_999`: 1000个监测点的位移数据。

## 结果文件
- [radar.csv](file:///Users/dc/Z研究生/time_series/Time-Series-Library/dataset/radar_sim/radar.csv)
