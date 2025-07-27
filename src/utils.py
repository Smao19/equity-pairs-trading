"""
utils.py

Storage of modular tools useful for repeated tasks aiding reaserch & development.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


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


def prepare_backtrader_dataframes(tickers_csv="final_pairs.csv", historical_parquet="test.parquet", data_dir=os.path.abspath(os.path.join('..', 'data'))):
    """
    Given a data directory, and the names of a csv file containing tickers 
    and a parquet file containing historical OHLCV data
    return a dictionary of cleaned Backtrader-ready DataFrames.
    """
    
    # Read tickers and historical data down from specified files
    ticker_df = pd.read_csv(os.path.join(data_dir, tickers_csv))
    ticker_list = ticker_df.values.ravel().tolist()
    historical_df = pd.read_parquet(os.path.join(data_dir, historical_parquet))

    # Ensure tickers are unique
    unique_tickers = set(ticker_list)
    data_dict = {}

    for ticker in unique_tickers:
        try:
            # Extract the sub-columns for this ticker
            df = historical_df[ticker].copy()

            # Ensure correct column order for Backtrader
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Sort index and drop missing
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().dropna()

            data_dict[ticker] = df

        except KeyError:
            print(f"[Warning] Ticker '{ticker}' not found in columns.")

    return data_dict


def compute_rolling_hedge_ratio(df_A, df_B, window=60):
    """
    Rolling regression to compute hedge ratio series for specified pair.
    """
    
    hedge_ratios = []

    for i in range(window, len(df_A)):
        y = df_A['close'].iloc[i - window:i].values
        x = df_B['close'].iloc[i - window:i].values
        x = add_constant(x)

        model = OLS(y, x).fit()
        hedge_ratios.append(model.params[1])  # beta coefficient

    # Pad beginning with NaNs
    hedge_ratios = [np.nan]*window + hedge_ratios
    return pd.Series(hedge_ratios, index=df_A.index)
