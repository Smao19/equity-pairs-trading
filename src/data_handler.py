"""
data_handler.py

Responsible for loading, cleaning, and aligning raw OHLCV data for the asset universe.
- Handles missing values and timestamp alignment.
- Outputs cleaned DataFrames for downstream modules.
- Offers data handling methods for facilitating backtest simulation
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from polygon import RESTClient
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


# Create a logger for the data handler module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if re-imported
if not logger.handlers:
    # Create log directory if it doesn't exist
    log_dir = os.path.join("..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Log file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "data_handler.log"))
    file_handler.setLevel(logging.DEBUG)

    # Log formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)


def get_pairs_list(tickers_csv="final_pairs.csv", data_dir=os.path.abspath(os.path.join('.', 'data'))):
    # Read tickers down from specified files
    ticker_df = pd.read_csv(os.path.join(data_dir, tickers_csv))
    ticker_list = ticker_df.values.tolist()
    return ticker_list


def compute_zscore_series(spread: pd.Series, window: int) -> pd.Series:
    """
    Compute the rolling z-score of a spread series.
    z_t = (spread_t - mean_t) / std_t
    where mean_t and std_t are the rolling mean and std at time t over the past window periods.
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    return (spread - rolling_mean) / rolling_std


def compute_rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling beta of y regressed on x using OLS over a moving window.
    beta_t = Cov(y_t, x_t) / Var(x_t)
    """
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    return beta


def prepare_backtest_data(rolling_window: int, tickers_csv="final_pairs.csv", historical_parquet="test.parquet", data_dir=os.path.abspath("data")):
    """
    Prepares cleaned and aligned data for backtest. Computes spread z-score and rolling hedge ratio (beta).
    """
    pair_list = get_pairs_list(tickers_csv=tickers_csv, data_dir=data_dir)
    historical_df = pd.read_parquet(os.path.join(data_dir, historical_parquet))

    pair_data_dict = {}

    for t1, t2 in pair_list:
        try:
            df1 = historical_df[t1][['open', 'high', 'low', 'close', 'volume']].copy()
            df2 = historical_df[t2][['open', 'high', 'low', 'close', 'volume']].copy()

            df1.index = pd.to_datetime(df1.index)
            df2.index = pd.to_datetime(df2.index)

            df1 = df1.sort_index().dropna()
            df2 = df2.sort_index().dropna()

            # Align on datetime index
            df1_aligned, df2_aligned = df1.align(df2, join='inner', axis=0)

            # Add price column
            df1_aligned['price'] = df1_aligned['close']
            df2_aligned['price'] = df2_aligned['close']

            # Compute rolling hedge ratio (beta) of df1 on df2
            beta_series = compute_rolling_beta(df1_aligned['price'], df2_aligned['price'], rolling_window)
            df1_aligned['hedge_ratio'] = beta_series

            # Compute spread using rolling beta
            spread = df1_aligned['price'] - beta_series * df2_aligned['price']
            df1_aligned['zscore'] = compute_zscore_series(spread, window=rolling_window)

            pair_data_dict[(t1, t2)] = (df1_aligned, df2_aligned)

        except KeyError as e:
            print(f"[Warning] Ticker '{e.args[0]}' not found in columns.")

    return pair_data_dict


def clean_data(df):
    """
    Cleans OHLCV financial time series data:
    - Ensures datetime index is sorted and unique
    - Removes or aggregates duplicate timestamps
    - Drops rows with NaN values
    - Removes price anomalies (e.g. negative or zero prices)
    - Drops first and last 30 minutes of each trading day to reduce noise
    """
    import pandas as pd
    import numpy as np

    # Ensure datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df.set_index('datetime', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index or 'datetime' column.")

    # Sort index
    df.sort_index(inplace=True)

    # Remove duplicate timestamps by aggregating
    if not df.index.is_unique:
        df = df.groupby(df.index).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Filter out price anomalies
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    # Keep only times between 10:00 and 15:30 (i.e., remove first & last 30 mins of market)
    df = df.between_time("10:00", "15:30")

    return df


def align_data(df, method='inner'):
    """
    Aligns a multi-indexed DataFrame of OHLCV data across tickers using the specified method.

    Parameters:
    - df: A DataFrame with MultiIndex columns (ticker, attribute).
    - method: Join method — 'inner' (default) or 'outer'.

    Returns:
    - Aligned DataFrame with a common datetime index.
    """

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns with (ticker, attribute) format.")

    # Split and realign each ticker-specific subframe
    tickers = df.columns.get_level_values(0).unique()
    aligned_dfs = []

    for ticker in tickers:
        sub_df = df[ticker].copy()
        sub_df.columns = pd.MultiIndex.from_product([[ticker], sub_df.columns])
        aligned_dfs.append(sub_df)

    return pd.concat(aligned_dfs, axis=1, join=method)


class DataHandler:
    def __init__(self, tickers, api_key=None):
        self.tickers = tickers
        self.api_key = api_key
        self.client = RESTClient(self.api_key)

    def fetch_single_df(self, ticker, start_date="2022-01-01", end_date="2023-12-31"):
        """Fetch and clean minute-bar OHLCV data from Polygon for a single ticker and return as a DataFrame."""
        all_data = []
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        step = timedelta(days=5)

        total_days = (end - start).days + 1
        windows = [
            (start + i * step, min(start + (i + 1) * step, end))
            for i in range((total_days + step.days - 1) // step.days)
        ]

        logger.info(f"Fetching {ticker} from {start_date} to {end_date}...")

        for window_start, window_end in windows:
            try:
                resp = self.client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="minute",
                    from_=window_start.strftime("%Y-%m-%d"),
                    to=window_end.strftime("%Y-%m-%d"),
                    limit=50000
                )
                for bar in resp:
                    all_data.append({
                        "datetime": datetime.fromtimestamp(bar.timestamp / 1000.0),
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume
                    })
            except Exception as e:
                logger.error(f"[ERROR] {ticker} {window_start.date()} - {window_end.date()} failed: {e}")
                return None

        df = pd.DataFrame(all_data)
        if df.empty:
            logger.warning(f"[WARN] No data for {ticker}")
            return None

        df = clean_data(df)
        return df

    def fetch_all_to_df(self, start_date="2022-01-01", end_date="2023-12-31", show_progress=True, max_workers=12):
        """
        Fetch all tickers (multi-threaded) into a single multi-indexed DataFrame
        `max_workers` controls the number of parallel threads
        """

        data = {}

        def task(ticker):
            return ticker, self.fetch_single_df(ticker, start_date, end_date)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(task, t): t for t in self.tickers}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Fetching all tickers", disable=not show_progress):
                ticker, df = f.result()
                if df is not None:
                    data[ticker] = df
                else:
                    logger.warning(f"[WARN] No data fetched for {ticker}")

        if not data:
            raise RuntimeError("No data fetched for any tickers.")

        # Combine into a single multi-indexed DataFrame
        combined_df = pd.concat(data.values(), axis=1, keys=data.keys())
        return combined_df
