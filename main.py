from src.backtest_engine import BacktestEngine
from src.logger import logger

"""
All config options (for documentation)

config = {  # Defaults Values
    pairs_csv,  # final_pairs.csv
    data_parquet,  # test.parquet
    data_dir,  # data
    beta_rwindow,  # 60 minutes
    initial_cash,  # 1_000_000 dollars
    entry_threshold,  # 1 z-score
    exit_threshold,  # 0.2 z-score
    slippage_bps,  # 1 basis points
    commission_bps,  # 5 basis points
    max_gross_exposure,  # 20 
    capital_usage_limit,  # 0.5
    capital_per_trade,  # 0.02
}
"""

def main():
    config = {
        "data_parquet": "val.parquet",
        "beta_rwindow": 60,
        "initial_cash": 1_000_000,
        "entry_threshold": 3,
        "exit_threshold": 1,
        "slippage_bps": 1,
        "commission_bps": 5,
        "max_gross_exposure": 20,
        "capital_usage_limit": 0.5,
        "capital_per_trade": 0.02,
    }

    engine = BacktestEngine(config)
    logger.info("Starting backtest...")
    engine.run_backtest()

if __name__ == "__main__":
    main()
