#!/usr/bin/env python3
"""Extract and summarize experiment results from TSLib output.
Scans result_long_term_forecast.txt and organizes into a table.
"""
import os
import re
import sys


def parse_results(results_dir='./results'):
    """Parse result directories to extract model metrics."""
    results = []
    if not os.path.exists(results_dir):
        print(f'Results dir not found: {results_dir}')
        return results

    for entry in os.listdir(results_dir):
        path = os.path.join(results_dir, entry)
        if not os.path.isdir(path):
            continue

        # Parse model_id: slopemine_MODEL_slXX_plYY_eZZ_FEATURES
        m = re.match(
            r'slopemine_(\w+)_sl(\d+)_pl(\d+)_e(\d+)_(\w+)',
            entry
        )
        if not m:
            continue

        model, seq_len, pred_len, epochs, features = m.groups()

        # Read metrics
        metric_file = os.path.join(path, 'metrics.npy')
        if not os.path.exists(metric_file):
            continue

        try:
            import numpy as np
            metrics = np.load(metric_file, allow_pickle=True).item()
            mse = metrics.get('mse', None)
            mae = metrics.get('mae', None)
        except Exception:
            mse, mae = None, None

        results.append({
            'model': model,
            'seq_len': int(seq_len),
            'pred_len': int(pred_len),
            'epochs': int(epochs),
            'features': features,
            'mse': mse,
            'mae': mae,
        })

    return sorted(results, key=lambda r: (r['features'], r['pred_len'], r['mse'] or 999))


def print_table(results):
    """Print results as a formatted table."""
    if not results:
        print('No results found.')
        return

    # Group by features
    groups = {}
    for r in results:
        key = r['features']
        groups.setdefault(key, []).append(r)

    for feature_set, rows in sorted(groups.items()):
        print(f'\n{"="*70}')
        print(f'Feature set: {feature_set}')
        print(f'{"="*70}')
        print(f'{"Model":<12} {"PredLen":>7} {"Epochs":>6} {"MSE":>12} {"MAE":>12}')
        print(f'{"-"*12} {"-"*7} {"-"*6} {"-"*12} {"-"*12}')

        for r in sorted(rows, key=lambda x: (x['pred_len'], x['mse'] or 999)):
            mse_str = f'{r["mse"]:.4f}' if r['mse'] is not None else 'N/A'
            mae_str = f'{r["mae"]:.4f}' if r['mae'] is not None else 'N/A'
            print(f'{r["model"]:<12} {r["pred_len"]:>7} {r["epochs"]:>6} {mse_str:>12} {mae_str:>12}')

    # A/B comparison if both feature sets exist
    if 'full' in groups and 'weather' in groups:
        print(f'\n{"="*70}')
        print('A/B Comparison: weather vs weather+blast')
        print(f'{"="*70}')
        print(f'{"Model":<12} {"pl":>3} {"MSE(w)":>10} {"MSE(w+b)":>10} {"ΔMSE%":>8} {"MAE(w)":>10} {"MAE(w+b)":>10} {"ΔMAE%":>8}')
        print(f'{"-"*12} {"-"*3} {"-"*10} {"-"*10} {"-"*8} {"-"*10} {"-"*10} {"-"*8}')

        weather_map = {}
        for r in groups['weather']:
            weather_map[(r['model'], r['pred_len'])] = r

        for r in sorted(groups['full'], key=lambda x: (x['pred_len'], x['model'])):
            key = (r['model'], r['pred_len'])
            w = weather_map.get(key)
            if w and w['mse'] and r['mse']:
                dm = (r['mse'] - w['mse']) / w['mse'] * 100
                da = (r['mae'] - w['mae']) / w['mae'] * 100
                print(f'{r["model"]:<12} {r["pred_len"]:>3} {w["mse"]:>10.4f} {r["mse"]:>10.4f} {dm:>+7.1f}% {w["mae"]:>10.4f} {r["mae"]:>10.4f} {da:>+7.1f}%')


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else './results'
    results = parse_results(results_dir)
    print_table(results)


if __name__ == '__main__':
    main()
