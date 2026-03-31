# patch_data_diagnostic_report_v2

## 数据概况
- 点位总数：19566
- 新分区数：5 个
- 稳定背景区点位：32 个，占比 0.1635%

## 关键结论
- 新分区比旧分区更均衡：旧规则几乎只分成“稳定背景区 + 其余全部区域”两块，其中旧分区2独占绝大多数点位；新规则把活跃区拆成坡脚、坡中、平台、坡顶四个工程区，空间语义和 patch 数都更可解释。
- `patch_size=10` 适合作为主实验粒度：主方案下 ps8 有 428 个 patch，ps10 有 308 个 patch；ps10 在明显降低 patch 数的同时，仍保留足够的 zone 内细粒度结构，更适合作为正式实验主粒度。
- 稳定背景区不建议纳入主实验训练：该区只有 32 个点位，正式主方案中仅保留 1 个 background patch，并显式标记 `is_train_patch=0`，更适合作为参照区而不是预测目标。

## 新旧分区对比
- 旧分区点位分布：旧分区1=32, 旧分区2=19534
- 新分区点位分布：稳定背景区=32, 坡脚区=6430, 坡中区=3447, 平台区=5106, 坡顶区=4551

## patch_size 诊断
- ps8 主方案：active patch=428，sparse patch=0，平均点数/patch=45.71
- ps10 主方案：active patch=308，sparse patch=0，平均点数/patch=63.53

## zone 特征摘要
- 稳定背景区：patch_count=1，disp_mean_avg=0.0443，disp_max_p95=1.1817，valid_ratio_avg=1.0000
- 坡脚区：patch_count=93，disp_mean_avg=0.1174，disp_max_p95=1.1841，valid_ratio_avg=1.0000
- 坡中区：patch_count=70，disp_mean_avg=0.1077，disp_max_p95=1.3333，valid_ratio_avg=1.0000
- 平台区：patch_count=71，disp_mean_avg=0.1354，disp_max_p95=1.3333，valid_ratio_avg=1.0000
- 坡顶区：patch_count=73，disp_mean_avg=0.1537，disp_max_p95=1.3333，valid_ratio_avg=1.0000

## 代表时刻热力图
- 2024-06-02 14:00：mean_abs_disp=0.119278，mean_disp_max=0.715313，图文件=`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/diagnostic_package_v2/selected_time_patch_heatmaps/disp_mean_heatmap_20240602_1400.png`
- 2024-06-08 22:00：mean_abs_disp=0.250077，mean_disp_max=0.784444，图文件=`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/diagnostic_package_v2/selected_time_patch_heatmaps/disp_mean_heatmap_20240608_2200.png`
- 2024-06-19 03:00：mean_abs_disp=0.009137，mean_disp_max=0.175494，图文件=`/Users/dc/Z研究生/time_series/Time-Series-Library/outputs/slopemine_v2/diagnostic_package_v2/selected_time_patch_heatmaps/disp_mean_heatmap_20240619_0300.png`