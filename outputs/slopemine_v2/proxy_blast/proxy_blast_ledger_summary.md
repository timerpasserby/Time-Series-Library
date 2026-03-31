# proxy_blast_ledger_summary

- 本页所有坐标均为代理推断坐标，不是实时真实爆破坐标。
- 主代理版本为 `balanced`，并额外导出 `conservative` 与 `localized` 用于敏感性分析。

## 版本参数
- conservative：min_event_distance=12.0，sigma∈[38.0, 44.0]，weights=(toe=0.30, middle_slope=0.30, platform=0.23, crest=0.17)
- balanced：min_event_distance=15.0，sigma∈[32.0, 38.0]，weights=(toe=0.35, middle_slope=0.35, platform=0.20, crest=0.10)
- localized：min_event_distance=18.0，sigma∈[24.0, 30.0]，weights=(toe=0.42, middle_slope=0.33, platform=0.17, crest=0.08)

## 台账摘要
### conservative
- 代理事件数：70
- 爆破小时数：57
- zone 分布：{'坡顶区': 18, '坡脚区': 18, '坡中区': 18, '平台区': 16}
- Q 均值：378.397
- 同小时多事件最小距离：15.569

### balanced
- 代理事件数：70
- 爆破小时数：57
- zone 分布：{'坡中区': 25, '坡脚区': 20, '平台区': 18, '坡顶区': 7}
- Q 均值：378.397
- 同小时多事件最小距离：22.361

### localized
- 代理事件数：70
- 爆破小时数：57
- zone 分布：{'坡脚区': 25, '平台区': 21, '坡中区': 18, '坡顶区': 6}
- Q 均值：378.397
- 同小时多事件最小距离：21.674

## proxy V3 对比

| name | row_count | nonzero_patch_ratio | p50 | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_v3 | 221760 | 0.984722 | 13.410102 | 143.369091 | 407.523403 | 1130.394900 |
| proxy_conservative | 221760 | 0.986111 | 52.073383 | 256.996158 | 565.796985 | 1010.279480 |
| proxy_balanced | 221760 | 0.986111 | 32.350882 | 217.945215 | 524.774782 | 951.840759 |
| proxy_localized | 221760 | 0.986111 | 15.782247 | 152.087848 | 433.447382 | 1490.497070 |
