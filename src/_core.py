from __future__ import annotations

__all__ = ('plot_presets', 'compute_hermite', 'PRESETS')

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

if TYPE_CHECKING:
    from _types import F64NDArray, HermitePresets


# fmt: off
# Default Hermite curve presets
PRESETS: HermitePresets = {
    'smooth':           {'p0': 0, 'v0': 0, 'p1': 1, 'v1': 0},
    'accelerate':       {'p0': 0, 'v0': 0, 'p1': 1, 'v1': 1},
    'decelerate':       {'p0': 0, 'v0': 1, 'p1': 1, 'v1': 0},
    'bump':             {'p0': 0, 'v0': 4, 'p1': 0, 'v1': -4},
    'acceleratebump':   {'p0': 0, 'v0': 0, 'p1': 0, 'v1': -6.75},
    'deceleratebump':   {'p0': 0, 'v0': 6.75, 'p1': 0, 'v1': 0},

    # New presets
    'sharp':            {'p0': 0, 'v0': 3, 'p1': 1, 'v1': 0},
    'smooth_overshoot': {'p0': 0, 'v0': 0, 'p1': 1, 'v1': -0.6},
    'sharp_overshoot':  {'p0': 0, 'v0': 3, 'p1': 1, 'v1': -0.25},
    'inout_overshoot':  {'p0': 0, 'v0': -0.8, 'p1': 1, 'v1': -0.8},
    'codbump':          {'p0': 0, 'v0': 6, 'p1': 1, 'v1': 3},
}
# fmt: on


# Generate time values
X_CORDS = np.linspace(0, 1, 200, dtype=np.float64)

# Set font
FONT_PATH = Path(__file__).parent / 'fonts' / 'Renogare-Regular.otf'
font_manager.fontManager.addfont(FONT_PATH)
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams['font.family'] = FONT_PROP.get_name()

# Set dark theme
plt.style.use('dark_background')


def compute_hermite(t: F64NDArray, *, p0: float, v0: float, p1: float, v1: float) -> F64NDArray:
    """Compute cubic Hermite spline value."""
    t2: F64NDArray = t * t
    t3: F64NDArray = t2 * t
    return (
        (2 * t3 - 3 * t2 + 1) * p0  # h00
        + (t3 - 2 * t2 + t) * v0  # h10
        + (-2 * t3 + 3 * t2) * p1  # h01
        + (t3 - t2) * v1  # h11
    )


def plot_presets(presets: HermitePresets = PRESETS) -> None:
    """Plot preset curves."""

    # Create plot with square aspect ratio
    _fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # Set labels
    ax.set_title('Cubic Hermite Spline Easing Curves', color='white', pad=15, fontsize=16)
    ax.set_xlabel('Normalized Time (t)', color='white')
    ax.set_ylabel('Interpolation Progress (v)', color='white')
    ax.plot(
        [0, 1],
        [0, 1],
        label='Linear Reference',
        color='0.66',
        linestyle='--',
        alpha=0.4,
    )

    # Plot each preset curve
    for preset_name, params in presets.items():
        y = compute_hermite(X_CORDS, **params)
        ax.plot(
            X_CORDS,
            y,
            label=preset_name,
            linewidth=2.5,
            solid_capstyle='round',
            alpha=0.9,
        )

    # Create legend
    ax.legend(
        facecolor='0.125',
        edgecolor='0.25',
        bbox_to_anchor=(0.5, -0.07),
        loc='upper center',
        ncol=3,
    )

    # Fix top padding
    plt.subplots_adjust(top=1)

    # Show plot
    plt.show()


def main() -> None:
    plot_presets()


if __name__ == "__main__":
    main()
