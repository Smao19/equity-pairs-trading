"""
data_handler.py

Responsible for fetching, cleaning, and aligning raw OHLCV data for the asset universe.
- Handles missing values and timestamp alignment.
- Outputs cleaned DataFrames for downstream modules.
- Offers data handling methods for facilitating backtest simulation
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
import aiohttp
import asyncio
import sys


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


def compute_rolling_spread_vol(df1_px, df2_px, beta_series, window):
    """
    Compute rolling volatility of pair spread. 
    Useful for volatility weighted position sizing in execution.
    """
    spread = df1_px - beta_series * df2_px
    return spread.rolling(window=window).std()


def compute_rolling_slope(series, window):
    """
    Compute rolling spread slope. Simple diff over window lag.
    Useful for analyzing spread momentum and determining if mean reversion has started.
    """
    # Simple diff over window lag: slope = (series - series.shift(window)) / window
    return (series - series.shift(window)) / window


def prepare_backtest_data(rolling_window=60, slope_window=5, tickers_csv="final_pairs.csv", historical_parquet="test.parquet", data_dir=os.path.abspath("data"), static_beta=False):
    """
    Prepares cleaned and aligned data for backtest. 
    Computes:
    - spread 
    - z-score 
    - hedge ratio (beta)
    - beta volatility
    - spread volatility
    - spread slope
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
            df1_aligned['price'] = np.log(df1_aligned['close'])
            df2_aligned['price'] = np.log(df2_aligned['close'])

            if static_beta:
                # Compute static hedge ratio (beta) of df1 on df2 using OLS
                X = df2_aligned['price'].values.reshape(-1, 1)
                y = df1_aligned['price'].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
                df1_aligned['hedge_ratio'] = beta
            else:
                # Compute rolling hedge ratio (beta) of df1 on df2
                beta_series = compute_rolling_beta(df1_aligned['price'], df2_aligned['price'], rolling_window)
                df1_aligned['hedge_ratio'] = beta_series

            # Compute rolling beta volatility
            if not static_beta:
                beta_vol = beta_series.rolling(window=rolling_window).std()
                df1_aligned['beta_vol'] = beta_vol

            # Compute spread using appropriate beta
            if static_beta:
                spread = df1_aligned['price'] - beta * df2_aligned['price']
                df1_aligned['spread'] = spread
            else: 
                spread = df1_aligned['price'] - beta_series * df2_aligned['price']
                df1_aligned['spread'] = spread
           
            # Compute zscore
            df1_aligned['zscore'] = compute_zscore_series(spread, window=rolling_window)

            # Compute rolling spread volatility
            if static_beta:
                spread_vol = compute_rolling_spread_vol(df1_aligned['price'], df2_aligned['price'], beta, rolling_window)
            else:
                spread_vol = compute_rolling_spread_vol(df1_aligned['price'], df2_aligned['price'], beta_series, rolling_window)
            df1_aligned['spread_vol'] = spread_vol

            # Compute spread momentum metric
            df1_aligned['spread_slope'] = compute_rolling_slope(spread, slope_window)

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
    Aligns OHLCV data across tickers using the specified method.

    Parameters:
    - df: A DataFrame with multiIndex columns: (ticker, attribute)
    - method: Join method — 'inner' (default), 'outer', 'ffill', etc.

    Returns:
    - Aligned DataFrame with a common datetime index (same column format as input).
    """

    # Multi-ticker DataFrame with MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique()
        aligned_dfs = []

        for ticker in tickers:
            sub_df = df[ticker].copy()
            sub_df.columns = pd.MultiIndex.from_product([[ticker], sub_df.columns])
            aligned_dfs.append(sub_df)

        return pd.concat(aligned_dfs, axis=1, join=method)
    else:
        raise ValueError("Expected DataFrame with MultiIndex columns.")


class DataHandler:
    def __init__(self, tickers, api_key, chunk_size="monthly", max_concurrent_requests=15):
        self.tickers = tickers
        self.api_key = api_key
        self.chunk_size = chunk_size  # 'monthly' or number of days as int
        self.max_concurrent_requests = max_concurrent_requests


    def _generate_chunks(self, start_date, end_date):
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        if self.chunk_size == "monthly":
            chunks = []
            current = start
            while current < end:
                next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
                chunks.append((current, min(next_month, end)))
                current = next_month
            return chunks
        elif isinstance(self.chunk_size, int):
            step = timedelta(days=self.chunk_size)
            return [(start + i * step, min(start + (i + 1) * step, end))
                    for i in range((end - start).days // self.chunk_size + 1)]
        else:
            raise ValueError("chunk_size must be 'monthly' or an integer.")


    async def _fetch_range(self, session, ticker, start, end, semaphore):
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start.date()}/{end.date()}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }

        async with semaphore:
            for attempt in range(2):  # Try once, then retry once
                try:
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")

                        data = await resp.json()
                        results = data.get("results", None)
                        
                        if not isinstance(results, list) or not results:
                            return ticker, None, None

                        df = pd.DataFrame([{
                            "datetime": datetime.fromtimestamp(b["t"] / 1000.0),
                            "open": b["o"],
                            "high": b["h"],
                            "low": b["l"],
                            "close": b["c"],
                            "volume": b["v"]
                        } for b in results])

                        return ticker, df, None

                except Exception as e:
                    logger.warning(f"[RETRY] {ticker} | Chunk {start.date()} to {end.date()} | Error: {e}")

                    if attempt < 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return ticker, None, "final failure"

        return ticker, None, "unexpected"


    async def _fetch_all_chunks_for_ticker(self, ticker, start_date, end_date, session, semaphore):
        # Generate date chunks (e.g., monthly) for this ticker
        chunks = self._generate_chunks(start_date, end_date)
        
        # Launch fetch tasks for each chunk
        tasks = [
            self._fetch_range(session, ticker, start, end, semaphore)
            for start, end in chunks
        ]

        # Await all chunk fetches for this ticker
        results = await asyncio.gather(*tasks)

        dfs = []
        for r in results:
            if isinstance(r, tuple) and len(r) == 3:
                _, df, _ = r
                if df is not None:
                    dfs.append(df)

        # If no chunks succeeded, log warning and return None
        if not dfs:
            logger.warning(f"[FAIL] {ticker} - all chunks failed or returned no data")
            return ticker, None

        # Combine all valid chunks for this ticker
        full_df = pd.concat(dfs)

        # Perform cleaning and alignment per ticker BEFORE combining all dataframes
        full_df.set_index("datetime", inplace=True)
        full_df = clean_data(full_df)

        # Add ticker level columns to produce MultiIndex
        full_df.columns = pd.MultiIndex.from_product([[ticker], full_df.columns])

        logger.info(f"[DONE] {ticker}")
        return ticker, full_df


    async def _fetch_all_tickers(self, start_date, end_date):
        # Limit concurrency with a semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        data = {}

        async with aiohttp.ClientSession() as session:
            # Launch one coroutine per ticker
            tasks = [
                self._fetch_all_chunks_for_ticker(t, start_date, end_date, session, semaphore)
                for t in self.tickers
            ]

            # Display progress with tqdm
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching all tickers", file=sys.stdout):
                ticker, df = await coro
                if df is not None:
                    data[ticker] = df

        logger.info(f"[SUMMARY] Fetched data for {len(data)}/{len(self.tickers)} tickers")
        return data


    def fetch_all_to_df(self, start_date: str, end_date: str,
                        align: bool = True, align_method: str = 'inner') -> pd.DataFrame:
        """
        Main method to fetch historical data for all tickers and return a combined DataFrame.
        Data is fetched, cleaned, aligned (if enabled), and combined.
        """
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(
            self._fetch_all_tickers(start_date, end_date)
        )

        # Remove tickers that returned None
        data = {ticker: df for ticker, df in data.items() if df is not None}

        # Combine using keys for proper MultiIndex
        combined_df = pd.concat(data.values(), axis=1, sort=True)

        return combined_df
