#!/usr/bin/env python3
import argparse
import csv
import math
import os


REQUIRED_COLUMNS = ['grid_x', 'grid_y', 'report_time', 'deformation', 'speed', 'acceleration']


def build_parser():
    parser = argparse.ArgumentParser(description='Convert long-format slope monitoring CSV into a TSLib custom CSV.')
    parser.add_argument('--input', required=True, help='Path to the raw long-format CSV file.')
    parser.add_argument('--output', required=True, help='Path to the generated TSLib-compatible CSV file.')
    parser.add_argument(
        '--mode',
        choices=['aggregate', 'single_grid_context'],
        default='aggregate',
        help='aggregate: fastest minimal runnable dataset; single_grid_context: target one grid point with global context.',
    )
    parser.add_argument('--grid_x', help='Required when mode=single_grid_context.')
    parser.add_argument('--grid_y', help='Required when mode=single_grid_context.')
    return parser


def make_bucket():
    return {
        'count': 0,
        'deformation_sum': 0.0,
        'deformation_sumsq': 0.0,
        'deformation_max': float('-inf'),
        'speed_sum': 0.0,
        'speed_sumsq': 0.0,
        'acceleration_sum': 0.0,
        'acceleration_sumsq': 0.0,
    }


def update_bucket(bucket, deformation, speed, acceleration):
    bucket['count'] += 1
    bucket['deformation_sum'] += deformation
    bucket['deformation_sumsq'] += deformation * deformation
    if deformation > bucket['deformation_max']:
        bucket['deformation_max'] = deformation
    bucket['speed_sum'] += speed
    bucket['speed_sumsq'] += speed * speed
    bucket['acceleration_sum'] += acceleration
    bucket['acceleration_sumsq'] += acceleration * acceleration


def mean_and_std(total, total_sq, count):
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return mean, math.sqrt(variance)


def finalize_bucket(bucket):
    deformation_mean, deformation_std = mean_and_std(
        bucket['deformation_sum'], bucket['deformation_sumsq'], bucket['count']
    )
    speed_mean, speed_std = mean_and_std(bucket['speed_sum'], bucket['speed_sumsq'], bucket['count'])
    acceleration_mean, acceleration_std = mean_and_std(
        bucket['acceleration_sum'], bucket['acceleration_sumsq'], bucket['count']
    )
    return {
        'deformation': deformation_mean,
        'speed': speed_mean,
        'acceleration': acceleration_mean,
        'deformation_std': deformation_std,
        'speed_std': speed_std,
        'acceleration_std': acceleration_std,
        'deformation_max': bucket['deformation_max'],
    }


def validate_columns(fieldnames):
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def ensure_parent_dir(output_path):
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def aggregate_mode(reader, output_path):
    by_time = {}
    row_count = 0

    for row in reader:
        row_count += 1
        report_time = row['report_time']
        bucket = by_time.setdefault(report_time, make_bucket())
        update_bucket(
            bucket,
            float(row['deformation']),
            float(row['speed']),
            float(row['acceleration']),
        )

    ensure_parent_dir(output_path)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date',
            'speed',
            'acceleration',
            'deformation_std',
            'speed_std',
            'acceleration_std',
            'deformation_max',
            'deformation',
        ])
        for report_time in sorted(by_time):
            values = finalize_bucket(by_time[report_time])
            writer.writerow([
                report_time,
                values['speed'],
                values['acceleration'],
                values['deformation_std'],
                values['speed_std'],
                values['acceleration_std'],
                values['deformation_max'],
                values['deformation'],
            ])

    print(f'Processed {row_count} raw rows into {len(by_time)} hourly rows.')
    print(f'Wrote aggregate dataset to: {output_path}')
    print('Recommended TSLib settings: data=custom, features=MS, target=deformation, enc_in=dec_in=c_out=7')


def single_grid_context_mode(reader, output_path, grid_x, grid_y):
    if grid_x is None or grid_y is None:
        raise ValueError('--grid_x and --grid_y are required when mode=single_grid_context')

    by_time = {}
    selected_grid = {}
    row_count = 0

    for row in reader:
        row_count += 1
        report_time = row['report_time']
        bucket = by_time.setdefault(report_time, make_bucket())
        deformation = float(row['deformation'])
        speed = float(row['speed'])
        acceleration = float(row['acceleration'])
        update_bucket(bucket, deformation, speed, acceleration)

        if row['grid_x'] == grid_x and row['grid_y'] == grid_y:
            selected_grid[report_time] = {
                'speed': speed,
                'acceleration': acceleration,
                'deformation': deformation,
            }

    missing_times = [report_time for report_time in sorted(by_time) if report_time not in selected_grid]
    if missing_times:
        raise ValueError(
            f'Selected grid ({grid_x}, {grid_y}) is missing {len(missing_times)} timestamps. '
            'Choose a grid with full hourly coverage.'
        )

    ensure_parent_dir(output_path)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date',
            'speed',
            'acceleration',
            'global_deformation_mean',
            'global_speed_mean',
            'global_acceleration_mean',
            'global_deformation_std',
            'global_speed_std',
            'global_acceleration_std',
            'deformation',
        ])
        for report_time in sorted(by_time):
            context = finalize_bucket(by_time[report_time])
            target = selected_grid[report_time]
            writer.writerow([
                report_time,
                target['speed'],
                target['acceleration'],
                context['deformation'],
                context['speed'],
                context['acceleration'],
                context['deformation_std'],
                context['speed_std'],
                context['acceleration_std'],
                target['deformation'],
            ])

    print(f'Processed {row_count} raw rows into {len(by_time)} hourly rows.')
    print(f'Wrote single-grid context dataset to: {output_path}')
    print('Recommended TSLib settings: data=custom, features=MS, target=deformation, enc_in=dec_in=c_out=9')


def main():
    args = build_parser().parse_args()

    with open(args.input, 'r', newline='') as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames or [])
        if args.mode == 'aggregate':
            aggregate_mode(reader, args.output)
        else:
            single_grid_context_mode(reader, args.output, args.grid_x, args.grid_y)


if __name__ == '__main__':
    main()
