#!/usr/bin/env python3
"""Generate the three main tables for slope experiment:
  1. point_meta.csv  — monitoring point metadata
  2. disp_series.csv — displacement time series (long format)
  3. weather.csv     — weather data aligned to monitoring timestamps
"""
import argparse
import csv
import os
from collections import Counter, defaultdict


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, help='Path to device_10001_sorted.csv')
    parser.add_argument('--weather', required=True, help='Path to processed_BJLweather_data.csv')
    parser.add_argument('--outdir', required=True, help='Output directory for three tables')
    return parser


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_raw_data(raw_path):
    """Read raw CSV in chunks, build point metadata and collect series."""
    point_info = {}  # (grid_x, grid_y) -> {zone_ids, area_seqs, count}
    series_rows = []  # list of (timestamp, grid_x, grid_y, disp, vel, acc)
    time_set = set()

    row_count = 0
    with open(raw_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            gx = int(row['grid_x'])
            gy = int(row['grid_y'])
            key = (gx, gy)

            if key not in point_info:
                point_info[key] = {'zone_ids': Counter(), 'area_seqs': Counter(), 'count': 0}
            point_info[key]['zone_ids'][int(row['area_id'])] += 1
            point_info[key]['area_seqs'][int(row['area_id_seq'])] += 1
            point_info[key]['count'] += 1

            series_rows.append((
                row['report_time'],
                gx, gy,
                float(row['deformation']),
                float(row['speed']),
                float(row['acceleration']),
            ))
            time_set.add(row['report_time'])

            if row_count % 1_000_000 == 0:
                print(f'  Read {row_count:,} rows...')

    print(f'Total raw rows: {row_count:,}')
    print(f'Unique points: {len(point_info):,}')
    print(f'Unique timestamps: {len(time_set)}')

    return point_info, series_rows, sorted(time_set)


def write_point_meta(point_info, outpath):
    """Assign point_id and write point_meta.csv."""
    # Sort points by (grid_x, grid_y) for deterministic ordering
    sorted_keys = sorted(point_info.keys())
    point_id_map = {}  # (grid_x, grid_y) -> point_id
    for pid, key in enumerate(sorted_keys):
        point_id_map[key] = pid

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['point_id', 'grid_x', 'grid_y', 'x', 'y', 'zone_id', 'area_id_seq', 'is_valid'])
        for key in sorted_keys:
            pid = point_id_map[key]
            gx, gy = key
            info = point_info[key]
            zone_id = info['zone_ids'].most_common(1)[0][0]
            area_seq = info['area_seqs'].most_common(1)[0][0]
            writer.writerow([pid, gx, gy, gx, gy, zone_id, area_seq, 1])

    print(f'Wrote {len(point_id_map)} points to {outpath}')
    return point_id_map


def write_disp_series(series_rows, point_id_map, outpath):
    """Write disp_series.csv with point_id mapping."""
    # Sort by (timestamp, point_id) for consistent ordering
    mapped = [(ts, point_id_map[(gx, gy)], disp, vel, acc)
              for ts, gx, gy, disp, vel, acc in series_rows]
    mapped.sort(key=lambda r: (r[0], r[1]))

    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'point_id', 'disp', 'vel', 'acc'])
        for ts, pid, disp, vel, acc in mapped:
            writer.writerow([ts, pid, disp, vel, acc])

    print(f'Wrote {len(mapped):,} rows to {outpath}')


def write_weather(weather_path, monitoring_times, outpath):
    """Read weather data, filter to monitoring timestamps, write weather.csv."""
    weather_data = {}
    with open(weather_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row['report_time']
            weather_data[ts] = {
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'wind_speed': row['wind_speed'],
                'rainfall': row['rainfall'],
            }

    monitor_set = set(monitoring_times)
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'temperature', 'humidity', 'wind_speed', 'rainfall'])
        matched = 0
        for ts in sorted(monitoring_times):
            if ts in weather_data:
                w = weather_data[ts]
                writer.writerow([ts, w['temperature'], w['humidity'], w['wind_speed'], w['rainfall']])
                matched += 1
            else:
                print(f'  WARNING: no weather data for {ts}')

    print(f'Wrote {matched}/{len(monitoring_times)} weather rows to {outpath}')


def main():
    args = build_parser().parse_args()
    ensure_dir(args.outdir)

    print('=== Step 1: Reading raw data ===')
    point_info, series_rows, time_list = process_raw_data(args.raw)

    print('\n=== Step 2: Writing point_meta.csv ===')
    meta_path = os.path.join(args.outdir, 'point_meta.csv')
    point_id_map = write_point_meta(point_info, meta_path)

    print('\n=== Step 3: Writing disp_series.csv ===')
    disp_path = os.path.join(args.outdir, 'disp_series.csv')
    write_disp_series(series_rows, point_id_map, disp_path)

    print('\n=== Step 4: Writing weather.csv ===')
    weather_path = os.path.join(args.outdir, 'weather.csv')
    write_weather(args.weather, time_list, weather_path)

    print('\n=== Done ===')
    print(f'Output directory: {args.outdir}')
    print(f'Files: point_meta.csv, disp_series.csv, weather.csv')


if __name__ == '__main__':
    main()
