"""
visuals.py

Modular visualization tools useful for aiding reaserch & development.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def mean_rev_assess_plot(rolling_window, start_date, end_date, pairs, axes, metric_label, stat_fn):
    """
    Plot spread and rolling mean/std for pairs with custom metric labeling.

    Parameters:
    - rolling_window: window (in days) for rolling calculations
    - start_date, end_date: zoom-in window
    - pairs: dict like {'High-Performing': row, 'Low-Performing': row}
    - axes: matplotlib axes to plot on
    - metric_label: e.g., 'ADF', 'Hurst'
    - stat_fn: function to extract and format stats from a row
    """

    for ax, (title, row) in zip(axes, pairs.items()):
        pair_name = f"{row['ticker1']} - {row['ticker2']}"
        spread = row['spread_series']

        # Rolling mean and std (using minutes: 5.5 hrs * 60 min * days)
        rolling_mean = spread.rolling(window=int(rolling_window * 5.5 * 60)).mean()
        rolling_std = spread.rolling(window=int(rolling_window * 5.5 * 60)).std()

        # Zoom window
        spread_zoom = spread[start_date:end_date]
        rolling_mean_zoom = rolling_mean[start_date:end_date]
        rolling_std_zoom = rolling_std[start_date:end_date]

        # Absolute mean over zoom window
        zoom_mean = spread_zoom.mean()

        # Plot spread and rolling metrics
        ax.plot(spread_zoom.index, spread_zoom, label='Spread', linewidth=1.2)
        ax.plot(rolling_mean_zoom.index, rolling_mean_zoom, label=f'Rolling Mean ({rolling_window} bars)', color='orange')
        ax.plot(rolling_mean_zoom.index, rolling_mean_zoom + rolling_std_zoom, label='+1 STD', color='green', linestyle='--')
        ax.plot(rolling_mean_zoom.index, rolling_mean_zoom - rolling_std_zoom, label='-1 STD', color='green', linestyle='--')
        ax.axhline(zoom_mean, color='dimgray', linestyle='--', linewidth=1.2,
                   label=f'True Mean ({zoom_mean:.4f})')

        # Titles and formatting
        ax.set_title(f"{title} {metric_label} Pair: {pair_name}\n{stat_fn(row)}")
        ax.set_xlabel("Time")
        ax.grid(True)

    axes[0].set_ylabel("Spread Value")
    axes[-1].legend(loc='upper right')
    plt.tight_layout()
    plt.show()
