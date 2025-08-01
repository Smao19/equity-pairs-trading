from tqdm import tqdm
import os
from .data_handler import prepare_backtest_data
from .strategy import Strategy
from .execution import ExecutionModel
from .portfolio import Portfolio
from .risk import RiskModel
from .reporting import ReportGenerator
from .logger import logger


def force_liquidation(timestamps, pair_data, portfolio: Portfolio):
    # Force liquidation of all positions
    final_timestamp = timestamps[-1]
    logger.info("Performing final liquidation of all open positions...")
    liquidation_trades = {}
    liquidation_prices = {}

    for pair, (df1, df2) in pair_data.items():
        if final_timestamp not in df1.index or final_timestamp not in df2.index:
            continue

        pos = portfolio.leg_positions.get(pair)
        if not pos:
            continue

        if abs(pos["leg1"]) > 1e-6 or abs(pos["leg2"]) > 1e-6:
            liquidation_trades[pair] = {
                "leg1": -pos["leg1"],
                "leg2": -pos["leg2"]
            }
            px1 = df1.loc[final_timestamp]['price']
            px2 = df2.loc[final_timestamp]['price']
            beta = df1.loc[final_timestamp]['hedge_ratio']
            liquidation_prices[pair] = {
                "px1": px1,
                "px2": px2,
                "beta": beta
            }

        if liquidation_trades:
            portfolio.update(liquidation_trades, liquidation_prices, final_timestamp)


def load_data(config):
    # Helper for loading data into backtest engine via config specs
    tickers_csv = config.get('pairs_csv', "final_pairs.csv")
    historical_parquet = config.get('data_parquet', "test.parquet")
    data_dir = config.get('data_dir', os.path.abspath(os.path.join('.', 'data')))
    rolling_window = config.get('rwindow', 60)
    spread_momentum_window = config.get('spread_momentum_window', 5)
    return prepare_backtest_data(rolling_window, spread_momentum_window, tickers_csv, historical_parquet, data_dir)  # {('ticker1', 'ticker2'): (df1, df2)}


class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.pair_data_dict = load_data(self.config)
        self.strategy = Strategy(config)
        self.execution = ExecutionModel(config)
        self.execution.pair_data = self.pair_data_dict
        self.portfolio = Portfolio(config)
        self.risk = RiskModel(config)
        self.risk.pair_data = self.pair_data_dict
        self.reports = ReportGenerator()

        # Precompute common timestamps
        self.timestamps = self._get_common_timestamps()


    def _get_common_timestamps(self):
        timestamps_sets = []
        for df1, df2 in self.pair_data_dict.values():
            timestamps_sets.append(set(df1.index) & set(df2.index))
        return sorted(set.intersection(*timestamps_sets))


    def run_backtest(self):
        for timestamp in tqdm(self.timestamps, desc="Running Backtest", unit="bar"):
            current_prices = {}
            logger.debug(f"[{timestamp}] Processing timestamp")
            
            for pair, (df1, df2) in self.pair_data_dict.items():
            
                if timestamp in df1.index and timestamp in df2.index:
                    px1 = df1.loc[timestamp]['price']
                    px2 = df2.loc[timestamp]['price']
                    z = df1.loc[timestamp]['zscore']
                    beta = df1.loc[timestamp]['hedge_ratio']
                    current_prices[pair] = {
                        "px1": px1,
                        "px2": px2,
                        "zscore": z,
                        "beta": beta
                    }

            logger.debug(f"[{timestamp}] Prices: {current_prices}")
            signals = self.strategy.generate_signals(current_prices, self.portfolio.leg_positions)
            logger.debug(f"[{timestamp}] Signals: {signals}")
            signals = self.risk.apply_constraints(signals, self.portfolio, timestamp)
            trades = self.execution.generate_trades(signals, self.portfolio.leg_positions, self.portfolio, current_prices, timestamp)
            self.portfolio.update(trades, current_prices, timestamp)

        # Liquidate all positions for accurate final metrics
        force_liquidation(self.timestamps, self.pair_data_dict, self.portfolio)
        
        self.reports.generate(self.portfolio)
