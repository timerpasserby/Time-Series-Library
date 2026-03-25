#!/usr/bin/env python3
"""Merge blast features into aggregate CSV.
Target (deformation) must be the LAST column for TSLib MS mode.
"""
import argparse
import csv
import os


BLAST_COLS = [
    'blast_count', 'total_charge_kg', 'disturbance_index',
    'blast_count_6h', 'charge_24h_kg', 'max_ppv_est_mm_s', 'mean_charge_kg',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', required=True)
    parser.add_argument('--blast', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Read blast data (handle BOM)
    blast = {}
    with open(args.blast, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            blast[row['timestamp']] = {col: row[col] for col in BLAST_COLS}

    # Read aggregate data
    with open(args.agg) as fin:
        rows = list(csv.DictReader(fin))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as fout:
        writer = csv.writer(fout)
        # Order: date, internal(6), weather(4), blast(7), deformation(target=last)
        writer.writerow([
            'date',
            'speed', 'acceleration',
            'deformation_std', 'speed_std', 'acceleration_std', 'deformation_max',
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'blast_count', 'total_charge_kg', 'disturbance_index',
            'blast_count_6h', 'charge_24h_kg', 'max_ppv_est_mm_s', 'mean_charge_kg',
            'deformation',
        ])

        for row in rows:
            ts = row['date']
            b = blast.get(ts, {col: '0.0' for col in BLAST_COLS})
            writer.writerow([
                ts,
                row['speed'], row['acceleration'],
                row['deformation_std'], row['speed_std'],
                row['acceleration_std'], row['deformation_max'],
                row['temperature'], row['humidity'],
                row['wind_speed'], row['rainfall'],
                b['blast_count'], b['total_charge_kg'], b['disturbance_index'],
                b['blast_count_6h'], b['charge_24h_kg'],
                b['max_ppv_est_mm_s'], b['mean_charge_kg'],
                row['deformation'],
            ])

    print(f'Merged {len(rows)} rows -> {args.output}')
    print(f'enc_in=dec_in=c_out=17 (6 internal + 4 weather + 7 blast)')


if __name__ == '__main__':
    main()
