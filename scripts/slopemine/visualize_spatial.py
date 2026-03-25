#!/usr/bin/env python3
"""Spatial visualization for slope monitoring points.
Generates plots for point distribution, zone partition, and patch grid.
"""
import argparse
import csv
import os


def load_points(meta_path):
    points = []
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            points.append({
                'pid': int(row['point_id']),
                'x': int(row['x']),
                'y': int(row['y']),
                'zone': int(row['zone_id']),
            })
    return points


def plot_point_distribution(points, outdir):
    """Plot 1: Monitoring points colored by zone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # All points
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    axes[0].scatter(xs, ys, s=1, alpha=0.5, c='steelblue')
    axes[0].set_title(f'All Monitoring Points (n={len(points)})', fontsize=14)
    axes[0].set_xlabel('Grid X')
    axes[0].set_ylabel('Grid Y')
    axes[0].set_aspect('equal')

    # Colored by zone
    colors = {1: 'red', 2: 'blue'}
    for zone, color in colors.items():
        zone_pts = [p for p in points if p['zone'] == zone]
        if zone_pts:
            axes[1].scatter(
                [p['x'] for p in zone_pts],
                [p['y'] for p in zone_pts],
                s=1, alpha=0.6, c=color, label=f'Zone {zone} (n={len(zone_pts)})'
            )
    axes[1].set_title('Points by Zone', fontsize=14)
    axes[1].set_xlabel('Grid X')
    axes[1].set_ylabel('Grid Y')
    axes[1].legend(markerscale=10)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(outdir, 'point_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_patch_grid(points, patch_size, outdir):
    """Plot 2: Points with patch grid overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot points
    ax.scatter(xs, ys, s=2, alpha=0.4, c='steelblue')

    # Draw grid
    for x in range(x_min, x_max + 1, patch_size):
        ax.axvline(x=x, color='gray', linewidth=0.3, alpha=0.5)
    for y in range(y_min, y_max + 1, patch_size):
        ax.axhline(y=y, color='gray', linewidth=0.3, alpha=0.5)

    # Count points per patch
    patch_counts = {}
    for p in points:
        px = p['x'] // patch_size
        py = p['y'] // patch_size
        key = (px, py)
        patch_counts[key] = patch_counts.get(key, 0) + 1

    n_patches = len(patch_counts)
    counts = list(patch_counts.values())
    empty_patches = (x_max - x_min) // patch_size * (y_max - y_min) // patch_size - n_patches

    ax.set_title(
        f'Patch Grid (size={patch_size}) | {n_patches} occupied patches\n'
        f'Points per patch: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.1f}',
        fontsize=13
    )
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(outdir, f'patch_grid_{patch_size}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')

    # Print patch stats
    print(f'  Patch size: {patch_size}')
    print(f'  Occupied patches: {n_patches}')
    print(f'  Points per patch: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.1f}')
    print(f'  Median points/patch: {sorted(counts)[len(counts)//2]}')

    return patch_counts


def plot_patch_heatmap(points, patch_size, outdir):
    """Plot 3: Heatmap of points per patch."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    nx = (x_max - x_min) // patch_size + 1
    ny = (y_max - y_min) // patch_size + 1

    grid = np.zeros((ny, nx), dtype=int)
    for p in points:
        px = (p['x'] - x_min) // patch_size
        py = (p['y'] - y_min) // patch_size
        grid[py, px] += 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(grid, origin='lower', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Point count')
    ax.set_title(f'Point Density per Patch (patch_size={patch_size})', fontsize=14)
    ax.set_xlabel('Patch X index')
    ax.set_ylabel('Patch Y index')

    plt.tight_layout()
    path = os.path.join(outdir, f'patch_heatmap_{patch_size}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def suggest_patch_size(points):
    """Suggest reasonable patch sizes based on point spread."""
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    print(f'\nPoint coordinate range:')
    print(f'  X: [{min(xs)}, {max(xs)}], span={x_range}')
    print(f'  Y: [{min(ys)}, {max(ys)}], span={y_range}')

    suggestions = []
    for ps in [5, 8, 10, 16, 20]:
        nx = (x_range // ps) + 1
        ny = (y_range // ps) + 1
        patch_counts = {}
        for p in points:
            key = (p['x'] // ps, p['y'] // ps)
            patch_counts[key] = patch_counts.get(key, 0) + 1
        n_occupied = len(patch_counts)
        avg_pts = len(points) / n_occupied if n_occupied else 0
        suggestions.append((ps, nx, ny, n_occupied, avg_pts))
        print(f'  patch_size={ps}: grid={nx}x{ny}, occupied={n_occupied}, avg_pts/patch={avg_pts:.1f}')

    return suggestions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--patch_sizes', nargs='+', type=int, default=[5, 10, 16])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    points = load_points(args.meta)
    print(f'Loaded {len(points)} points')

    print('\n=== Plot 1: Point Distribution ===')
    plot_point_distribution(points, args.outdir)

    print('\n=== Patch Size Suggestions ===')
    suggest_patch_size(points)

    for ps in args.patch_sizes:
        print(f'\n=== Plot: Patch Grid (size={ps}) ===')
        plot_patch_grid(points, ps, args.outdir)
        plot_patch_heatmap(points, ps, args.outdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
