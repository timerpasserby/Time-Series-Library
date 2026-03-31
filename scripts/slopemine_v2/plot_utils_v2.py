"""这个文件统一管理 slopemine v2 的绘图字体和 patch 坐标轴设置。
相关模块：build_patch_dataset_v2.py、run_patch_baselines_v2.py
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib import font_manager


CJK_FONT_CANDIDATES = [
    "Songti SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "STHeiti",
    "Arial Unicode MS",
]


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
