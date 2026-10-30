import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#84837c"
GRID = "#e3e2dd"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#e87ba4", "#008300"]

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "viz_sequential", ["#f2f6fc", "#bcd4f2", "#7fabe4", "#4a8ad9", "#2a78d6", "#1b4f8c"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK_PRIMARY,
    "axes.labelsize": 9,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def strip_spines(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def label_bars(ax, bars, fmt="{:.3f}", offset=0.006, rotation=0, size=7.4):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + offset, fmt.format(height),
            ha="center", va="bottom", fontsize=size, color=INK_SECONDARY, rotation=rotation,
        )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path
