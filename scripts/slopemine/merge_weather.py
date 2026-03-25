#!/usr/bin/env python3
"""Build enhanced aggregate CSV with weather features merged.
Input: slopemine_aggregate.csv + weather.csv
Output: slopemine_agg_weather.csv (with weather features)
"""
import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', required=True, help='Path to slopemine_aggregate.csv')
    parser.add_argument('--weather', required=True, help='Path to weather.csv')
    parser.add_argument('--output', required=True, help='Output path for merged CSV')
    args = parser.parse_args()

    # Read weather data
    weather = {}
    with open(args.weather) as f:
        for row in csv.DictReader(f):
            weather[row['timestamp']] = {
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'wind_speed': row['wind_speed'],
                'rainfall': row['rainfall'],
            }

    # Read aggregate data and merge
    with open(args.agg) as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as fout:
        writer = csv.writer(fout)
        # Target=deformation, put it last for TSLib MS mode
        writer.writerow([
            'date',
            'speed', 'acceleration',
            'deformation_std', 'speed_std', 'acceleration_std', 'deformation_max',
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'deformation',
        ])

        merged = 0
        for row in rows:
            ts = row['date']
            w = weather.get(ts)
            if w is None:
                print(f'WARNING: no weather for {ts}')
                continue
            writer.writerow([
                ts,
                row['speed'], row['acceleration'],
                row['deformation_std'], row['speed_std'],
                row['acceleration_std'], row['deformation_max'],
                w['temperature'], w['humidity'], w['wind_speed'], w['rainfall'],
                row['deformation'],
            ])
            merged += 1

    print(f'Merged {merged} rows -> {args.output}')
    print(f'Columns: date + 11 features + deformation(target) = 13 columns')
    print(f'Recommended: enc_in=dec_in=c_out=11, features=MS, target=deformation')


if __name__ == '__main__':
    main()
