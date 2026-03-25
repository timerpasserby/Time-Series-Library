#!/usr/bin/env python3
"""Data quality diagnostics for slope monitoring data.
Outputs 6 statistics required by the experimental guide.
"""
import argparse
import csv
import math
from collections import Counter, defaultdict


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disp', required=True, help='Path to disp_series.csv')
    parser.add_argument('--meta', required=True, help='Path to point_meta.csv')
    parser.add_argument('--weather', required=True, help='Path to weather.csv')
    parser.add_argument('--outdir', required=True, help='Output directory')
    return parser


def diag1_point_count(meta_path):
    """Diagnostic 1: Total points, valid points, zone distribution."""
    points = []
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            points.append(row)
    total = len(points)
    valid = sum(1 for p in points if int(p['is_valid']) == 1)
    zone_counter = Counter(int(p['zone_id']) for p in points)
    print(f'\n[1] Monitoring Points')
    print(f'  Total points:  {total}')
    print(f'  Valid points:  {valid}')
    print(f'  Zone distribution: {dict(zone_counter)}')
    return points


def diag2_missing_rate(disp_path, total_points, total_times):
    """Diagnostic 2: Missing rate per point."""
    point_counts = Counter()
    with open(disp_path) as f:
        for row in csv.DictReader(f):
            point_counts[int(row['point_id'])] += 1

    missing_rates = []
    for pid in range(total_points):
        observed = point_counts.get(pid, 0)
        rate = 1.0 - observed / total_times
        missing_rates.append(rate)

    zero_missing = sum(1 for r in missing_rates if r == 0)
    under_5pct = sum(1 for r in missing_rates if r < 0.05)
    max_rate = max(missing_rates)
    mean_rate = sum(missing_rates) / len(missing_rates)

    # Distribution buckets
    buckets = Counter()
    for r in missing_rates:
        if r == 0:
            buckets['0%'] += 1
        elif r < 0.05:
            buckets['0-5%'] += 1
        elif r < 0.10:
            buckets['5-10%'] += 1
        elif r < 0.20:
            buckets['10-20%'] += 1
        else:
            buckets['20%+'] += 1

    print(f'\n[2] Missing Rate per Point (expected={total_times} timestamps)')
    print(f'  Points with 0% missing: {zero_missing}/{total_points} ({100*zero_missing/total_points:.1f}%)')
    print(f'  Points with <5% missing: {under_5pct}/{total_points} ({100*under_5pct/total_points:.1f}%)')
    print(f'  Mean missing rate: {100*mean_rate:.2f}%')
    print(f'  Max missing rate: {100*max_rate:.2f}%')
    print(f'  Distribution: {dict(buckets)}')
    return missing_rates


def diag3_outlier_rate(disp_path, total_points):
    """Diagnostic 3: Outlier ratio using simple z-score on disp values."""
    # First pass: compute global stats
    sums = defaultdict(float)
    counts = defaultdict(int)
    with open(disp_path) as f:
        for row in csv.DictReader(f):
            pid = int(row['point_id'])
            sums[pid] += float(row['disp'])
            counts[pid] += 1

    means = {pid: sums[pid] / counts[pid] for pid in sums if counts[pid] > 0}

    # Second pass: compute std
    sumsq = defaultdict(float)
    with open(disp_path) as f:
        for row in csv.DictReader(f):
            pid = int(row['point_id'])
            diff = float(row['disp']) - means[pid]
            sumsq[pid] += diff * diff

    stds = {pid: math.sqrt(sumsq[pid] / counts[pid]) for pid in sumsq if counts[pid] > 0}

    # Third pass: count outliers (|z| > 3)
    outlier_counts = defaultdict(int)
    total_counts = defaultdict(int)
    with open(disp_path) as f:
        for row in csv.DictReader(f):
            pid = int(row['point_id'])
            total_counts[pid] += 1
            if stds.get(pid, 0) > 0:
                z = (float(row['disp']) - means[pid]) / stds[pid]
                if abs(z) > 3:
                    outlier_counts[pid] += 1

    outlier_rates = []
    for pid in range(total_points):
        if total_counts.get(pid, 0) > 0:
            rate = outlier_counts[pid] / total_counts[pid]
            outlier_rates.append(rate)
        else:
            outlier_rates.append(0)

    mean_rate = sum(outlier_rates) / len(outlier_rates) if outlier_rates else 0
    max_rate = max(outlier_rates) if outlier_rates else 0
    has_outlier = sum(1 for r in outlier_rates if r > 0)

    print(f'\n[3] Outlier Ratio (|z| > 3 on disp)')
    print(f'  Points with outliers: {has_outlier}/{total_points} ({100*has_outlier/total_points:.1f}%)')
    print(f'  Mean outlier rate: {100*mean_rate:.4f}%')
    print(f'  Max outlier rate: {100*max_rate:.4f}%')


def diag4_distributions(disp_path):
    """Diagnostic 4: Distribution of disp/vel/acc."""
    stats = {}
    for col in ['disp', 'vel', 'acc']:
        vals = []
        with open(disp_path) as f:
            for row in csv.DictReader(f):
                vals.append(float(row[col]))

        vals.sort()
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        p50 = vals[n // 2]
        p01 = vals[int(n * 0.01)]
        p99 = vals[int(n * 0.99)]
        zero_count = sum(1 for v in vals if v == 0)

        stats[col] = {
            'min': vals[0], 'max': vals[-1],
            'mean': mean, 'std': std,
            'p01': p01, 'p50': p50, 'p99': p99,
            'zero_pct': 100 * zero_count / n,
        }
        print(f'\n[4] Distribution: {col}')
        print(f'  Range: [{vals[0]:.6f}, {vals[-1]:.6f}]')
        print(f'  Mean: {mean:.6f}, Std: {std:.6f}')
        print(f'  P1: {p01:.6f}, P50: {p50:.6f}, P99: {p99:.6f}')
        print(f'  Zero values: {100*zero_count/n:.2f}%')

    return stats


def diag5_rainfall(weather_path):
    """Diagnostic 5: Rainfall distribution."""
    rainfall_vals = []
    rain_hours = 0
    with open(weather_path) as f:
        for row in csv.DictReader(f):
            r = float(row['rainfall'])
            rainfall_vals.append(r)
            if r > 0:
                rain_hours += 1

    non_zero = [r for r in rainfall_vals if r > 0]
    print(f'\n[5] Rainfall Distribution')
    print(f'  Total hours: {len(rainfall_vals)}')
    print(f'  Rain hours: {rain_hours} ({100*rain_hours/len(rainfall_vals):.1f}%)')
    if non_zero:
        print(f'  Non-zero range: [{min(non_zero):.4f}, {max(non_zero):.4f}] mm')
        print(f'  Non-zero mean: {sum(non_zero)/len(non_zero):.4f} mm')
    return rainfall_vals


def diag6_disp_rain_overlap(disp_path, weather_path):
    """Diagnostic 6: Check if displacement spikes overlap with rainfall."""
    # Get rainfall by timestamp
    rain_by_time = {}
    with open(weather_path) as f:
        for row in csv.DictReader(f):
            rain_by_time[row['timestamp']] = float(row['rainfall'])

    # Compute global disp stats
    all_disp = []
    with open(disp_path) as f:
        for row in csv.DictReader(f):
            all_disp.append(float(row['disp']))

    mean_disp = sum(all_disp) / len(all_disp)
    std_disp = math.sqrt(sum((v - mean_disp) ** 2 for v in all_disp) / len(all_disp))
    threshold = mean_disp + 2 * std_disp

    # Count: rain hours, spike hours, overlap
    rain_times = set(ts for ts, r in rain_by_time.items() if r > 0)

    # Aggregate disp by timestamp
    disp_by_time = defaultdict(list)
    with open(disp_path) as f:
        for row in csv.DictReader(f):
            disp_by_time[row['timestamp']].append(float(row['disp']))

    spike_times = set()
    for ts, disps in disp_by_time.items():
        max_disp = max(disps)
        if max_disp > threshold:
            spike_times.add(ts)

    overlap = rain_times & spike_times
    print(f'\n[6] Displacement-Rainfall Overlap')
    print(f'  Disp threshold (mean+2std): {threshold:.4f}')
    print(f'  Rain hours: {len(rain_times)}')
    print(f'  Spike hours (max_disp > threshold): {len(spike_times)}')
    print(f'  Overlap hours: {len(overlap)}')
    if len(rain_times) > 0:
        print(f'  Overlap rate (spike|rain): {100*len(overlap)/len(rain_times):.1f}%')
    if len(spike_times) > 0:
        print(f'  Overlap rate (rain|spike): {100*len(overlap)/len(spike_times):.1f}%')


def main():
    args = build_parser().parse_args()

    # Read point count from meta
    with open(args.meta) as f:
        total_points = sum(1 for _ in csv.DictReader(f))

    # Count timestamps from weather
    with open(args.weather) as f:
        total_times = sum(1 for _ in csv.DictReader(f))

    print('=' * 60)
    print('SlopeMine Data Quality Report')
    print('=' * 60)

    diag1_point_count(args.meta)
    diag2_missing_rate(args.disp, total_points, total_times)
    diag3_outlier_rate(args.disp, total_points)
    diag4_distributions(args.disp)
    diag5_rainfall(args.weather)
    diag6_disp_rain_overlap(args.disp, args.weather)

    print('\n' + '=' * 60)
    print('Diagnostics complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
