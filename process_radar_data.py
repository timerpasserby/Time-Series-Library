import pandas as pd
import numpy as np
import os

print("1. 加载数据...")
radar_df = pd.read_csv('sim_radar_hourly_displacement.csv')
radar_df.rename(columns={'report_time': 'date'}, inplace=True)
radar_df['date'] = pd.to_datetime(radar_df['date'])

weather_df = pd.read_csv('sim_weather.csv')
weather_df.rename(columns={'report_time': 'date'}, inplace=True)
weather_df['date'] = pd.to_datetime(weather_df['date'])

blast_df = pd.read_csv('sim_blast_logs.csv')
blast_df.rename(columns={'timestamp': 'date'}, inplace=True)
blast_df['date'] = pd.to_datetime(blast_df['date'])

print("2. 处理离散的爆破数据...")
# 将爆破日志按小时聚合（如果有同一小时多次爆破，则将强度相加）
blast_agg = blast_df.groupby('date')['intensity'].sum().reset_index()
blast_agg.rename(columns={'intensity': 'blast_intensity'}, inplace=True)

print("3. 合并特征数据...")
merged_df = pd.merge(weather_df, blast_agg, on='date', how='left')
merged_df['blast_intensity'] = merged_df['blast_intensity'].fillna(0)

# 合并雷达位移数据
merged_df = pd.merge(merged_df, radar_df, on='date', how='inner')

# 将作为 Target 的 `node_0` 移动到最后一列（Dataset_Custom 默认预测 target）
cols = list(merged_df.columns)
cols.remove('node_0')
cols.append('node_0')
merged_df = merged_df[cols]

print("4. 保存为 CustomDataset 格式...")
os.makedirs('dataset/radar_sim', exist_ok=True)
merged_df.to_csv('dataset/radar_sim/radar.csv', index=False)

print(f"数据处理完毕！文件保存在: dataset/radar_sim/radar.csv")
print(f"总时长/样本数: {len(merged_df)}, 特征维度: {len(merged_df.columns)-1}")
