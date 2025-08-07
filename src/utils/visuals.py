"""
visuals.py

Modular visualization tools useful for aiding reaserch & development.
"""

import matplotlib.pyplot as plt


def mean_rev_assess_plot(rolling_window, start_date, end_date, pairs, axes, metric_label, stat_fn, legend_locs=None):
    """
    Plot spread and rolling mean/std for pairs with custom metric labeling.
    Used repeatedly in pair_research.ipynb

    Parameters:
    - rolling_window: window (in days) for rolling calculations
    - start_date, end_date: zoom-in window
    - pairs: dict like {'High-Performing': row, 'Low-Performing': row}
    - axes: matplotlib axes to plot on
    - metric_label: e.g., 'ADF', 'Hurst'
    - stat_fn: function to extract and format stats from a row
    - legend_locs: list of legend locations corresponding to each subplot (optional)
    """

    mean_colors = ['tab:gray'] * len(pairs)

    for i, (ax, (title, row), mean_color) in enumerate(zip(axes, pairs.items(), mean_colors)):
        pair_name = f"{row['ticker1']} - {row['ticker2']}"
        spread = row['spread_series']

        # Rolling mean and std (using minutes: 5.5 hrs * 60 min * days)
        rolling_window_bars = int(rolling_window * 5.5 * 60)
        rolling_mean = spread.rolling(window=rolling_window_bars).mean()
        rolling_std = spread.rolling(window=rolling_window_bars).std()

        # Zoom window
        spread_zoom = spread[start_date:end_date]
        rolling_mean_zoom = rolling_mean[start_date:end_date]
        rolling_std_zoom = rolling_std[start_date:end_date]

        zoom_mean = spread_zoom.mean()

        # Plotting
        ax.plot(spread_zoom.index, spread_zoom, label='Spread', linewidth=1.2)
        ax.plot(rolling_mean_zoom.index, rolling_mean_zoom, label=f'Rolling Mean ({rolling_window} bars)', color='orange')
        ax.plot(rolling_mean_zoom.index, rolling_mean_zoom + rolling_std_zoom, label='+1 STD', color='green', linestyle='--')
        ax.plot(rolling_mean_zoom.index, rolling_mean_zoom - rolling_std_zoom, label='-1 STD', color='green', linestyle='--')
        ax.axhline(zoom_mean, color=mean_color, linestyle='--', linewidth=1.2,
                   label=f'True Mean: {pair_name} ({zoom_mean:.4f})')

        # Titles and formatting
        ax.set_title(f"{title} {metric_label} Pair: {pair_name}\n{stat_fn(row)}")
        ax.set_xlabel("Time")
        ax.grid(True)

        # Legend
        legend_loc = legend_locs[i] if legend_locs and i < len(legend_locs) else 'upper right'
        ax.legend(loc=legend_loc)

    axes[0].set_ylabel("Spread Value")
    plt.tight_layout()
    plt.show()


def _zoom_comp_helper(ax, pair_row, title, rolling_window, start_date, end_date, metric_label, stat_fn):
    """
    Plot a single pair's spread on a given axis.
    Used internally by mean_rev_zoom_comp_plot()
    """
    pair_name = f"{pair_row['ticker1']} - {pair_row['ticker2']}"
    spread = pair_row['spread_series']

    # Rolling mean and std (using minutes: 5.5 hrs * 60 min * days)
    rolling_mean = spread.rolling(window=int(rolling_window * 5.5 * 60)).mean()
    rolling_std = spread.rolling(window=int(rolling_window * 5.5 * 60)).std()

    # Zoom window
    spread_zoom = spread[start_date:end_date]
    rolling_mean_zoom = rolling_mean[start_date:end_date]
    rolling_std_zoom = rolling_std[start_date:end_date]

    zoom_mean = spread_zoom.mean()

    # Plot spread and bands
    ax.plot(spread_zoom.index, spread_zoom, label='Spread', linewidth=1.2)
    ax.plot(rolling_mean_zoom.index, rolling_mean_zoom, label=f'Rolling Mean ({rolling_window} bars)', color='orange')
    ax.plot(rolling_mean_zoom.index, rolling_mean_zoom + rolling_std_zoom, label='+1 STD', color='green', linestyle='--')
    ax.plot(rolling_mean_zoom.index, rolling_mean_zoom - rolling_std_zoom, label='-1 STD', color='green', linestyle='--')
    ax.axhline(zoom_mean, color='dimgray', linestyle='--', linewidth=1.2, label=f'True Mean ({zoom_mean:.4f})')

    ax.set_title(f"{title} {metric_label} Pair: {pair_name}\n{stat_fn(pair_row)}")
    ax.set_xlabel("Time")
    ax.grid(True)


def mean_rev_zoom_comp_plot(pair_row, zoom1, zoom2, rolling_window, metric_label, stat_fn):
    """
    Compare a single pair's mean-reversion across two timeframes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (title, start_date, end_date) in zip(axes, [zoom1, zoom2]):
        _zoom_comp_helper(
            ax=ax,
            pair_row=pair_row,
            title=title,
            rolling_window=rolling_window,
            start_date=start_date,
            end_date=end_date,
            metric_label=metric_label,
            stat_fn=stat_fn
        )

    axes[0].set_ylabel("Spread Value")
    axes[-1].legend(loc='upper right')
    plt.tight_layout()
    plt.show()



