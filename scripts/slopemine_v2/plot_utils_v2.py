"""这个文件统一管理 slopemine v2 的绘图字体和 patch 坐标轴设置。
相关模块：build_patch_dataset_v2.py、run_patch_baselines_v2.py
"""

from __future__ import annotations

import copy

from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np


CJK_FONT_CANDIDATES = [
    "Songti SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "STHeiti",
    "Arial Unicode MS",
]

GRAPH_ZERO_COLOR = "#F3F4F6"
GRAPH_POSITIVE_QUANTILE = 0.995
GRAPH_POWER_GAMMA = 0.4
GRAPH_CMAP = "magma"


def setup_plot_style() -> None:
    """设置统一的中文友好绘图风格。"""

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    cjk_font = next((name for name in CJK_FONT_CANDIDATES if name in available_fonts), "DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": [cjk_font, "Times New Roman", "DejaVu Serif"],
            "font.serif": [cjk_font, "Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#D0D0D0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def set_patch_axes(ax, patch_meta, x_min_col: str = "cell_x_min", x_max_col: str = "cell_x_max", y_min_col: str = "cell_y_min", y_max_col: str = "cell_y_max", pad: float = 4.0) -> None:
    """按 patch 边界设置真实坐标轴范围。"""

    x_min = float(patch_meta[x_min_col].min()) - pad
    x_max = float(patch_meta[x_max_col].max()) + pad
    y_min = float(patch_meta[y_min_col].min()) - pad
    y_max = float(patch_meta[y_max_col].max()) + pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def _to_numpy_matrix(matrix) -> np.ndarray:
    if hasattr(matrix, "detach"):
        matrix = matrix.detach()
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu()
    return np.asarray(matrix, dtype=np.float32)


def _build_even_ticks(length: int, tick_count: int = 6) -> np.ndarray:
    if length <= 1:
        return np.array([0], dtype=int)
    raw_ticks = np.linspace(0, length - 1, num=min(int(tick_count), int(length)))
    return np.unique(np.rint(raw_ticks).astype(int))


def summarize_graph_matrix(matrix) -> dict[str, float | int | tuple[int, int]]:
    """汇总稀疏图矩阵的非零占比和正值分位数，供报告与图注复用。"""

    values = _to_numpy_matrix(matrix)
    positive = values[values > 0]
    nonzero_count = int(np.count_nonzero(values))
    total_count = int(values.size)
    stats: dict[str, float | int | tuple[int, int]] = {
        "shape": tuple(int(dim) for dim in values.shape),
        "nonzero_count": nonzero_count,
        "nonzero_ratio": float(nonzero_count / total_count) if total_count > 0 else 0.0,
        "positive_count": int(positive.size),
        "positive_p50": float("nan"),
        "positive_p90": float("nan"),
        "positive_p99": float("nan"),
        "positive_p995": float("nan"),
        "positive_max": float("nan"),
    }
    if positive.size > 0:
        stats.update(
            {
                "positive_p50": float(np.quantile(positive, 0.50)),
                "positive_p90": float(np.quantile(positive, 0.90)),
                "positive_p99": float(np.quantile(positive, 0.99)),
                "positive_p995": float(np.quantile(positive, GRAPH_POSITIVE_QUANTILE)),
                "positive_max": float(positive.max()),
            }
        )
    return stats


def render_sparse_graph_heatmap(
    ax,
    matrix,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    tick_count: int = 6,
    add_colorbar: bool = True,
):
    """按论文展示风格渲染 PGGC 稀疏 dense 调试矩阵。"""

    values = _to_numpy_matrix(matrix)
    masked_values = np.ma.masked_less_equal(values, 0.0)
    stats = summarize_graph_matrix(values)

    cmap = copy.copy(plt.get_cmap(GRAPH_CMAP))
    cmap.set_bad(GRAPH_ZERO_COLOR)
    ax.set_facecolor(GRAPH_ZERO_COLOR)

    vmax = float(stats["positive_p995"])
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(stats["positive_max"])
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = colors.PowerNorm(gamma=GRAPH_POWER_GAMMA, vmin=0.0, vmax=max(vmax, 1e-8))

    image = ax.imshow(
        masked_values,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_ticks = _build_even_ticks(values.shape[1], tick_count=tick_count)
    y_ticks = _build_even_ticks(values.shape[0], tick_count=tick_count)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(int(tick)) for tick in x_ticks])
    ax.set_yticklabels([str(int(tick)) for tick in y_ticks])

    colorbar = None
    if add_colorbar:
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=9)
    return image, colorbar, stats
