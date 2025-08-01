from .logger import logger


class RiskModel:

    SPREAD_VOL_THRESHOLD = 1.5
    BETA_VOL_THRESHOLD = 0.5  # Same threshold from pair_research

    def __init__(self, config):
        self.max_gross_exposure = config.get("max_gross_exposure", 20)
        
        # Stored for access to spread volatility
        self.pair_data = {}  # filled externally


    def pull_from_pair_data(self, pair, timestamp, col_title: str):
        df1, _ = self.pair_data.get(pair, (None, None))
        if df1 is None or timestamp not in df1.index:  # Data unavailable, skip
            return None

        desired_data = df1.loc[timestamp, col_title]
        if desired_data is None:
            return None
        
        return desired_data


    def spread_vol_filter(self, pair, timestamp) -> bool:
        # Pull spread volatility from pair data
        spread_vol = self.pull_from_pair_data(pair, timestamp, 'spread_vol')
        if spread_vol is None:
            return False
    
        # Apply threshold
        if spread_vol >= RiskModel.SPREAD_VOL_THRESHOLD:
            logger.info(f"[RISK] {pair} skipped due to high spread volatility ({spread_vol:.2f})")
            return False

        return True

    def beta_vol_filter(self, pair, timestamp) -> bool:
        # Pull beta volatility from pair data
        beta_vol = self.pull_from_pair_data(pair, timestamp, 'beta_vol')
        if beta_vol is None:
            return False
       
        # Apply threshold
        if beta_vol > RiskModel.BETA_VOL_THRESHOLD:
            logger.info(f"[RISK] {pair} skipped due to high beta volatility ({beta_vol:.3f})")
            return False
        
        return True


    def spread_momentum_filter(self, pair, timestamp, signal) -> bool:
        # Pull spread momentum metric from pair data
        slope = self.pull_from_pair_data(pair, timestamp, 'spread_slope')
        if slope is None:
            return False

        # For a long entry (signal > 0), we want slope trending downward (negative)
        if signal > 0:
            if not (slope < 0):  # Mean reversion not starting yet
                logger.info(f"[MOMENTUM] {pair} skip long: slope={slope:.4f}")
                return False

        # For a short entry (signal < 0), we want slope trending upward (positive)
        if signal < 0:
            if not (slope > 0):  # Mean reversion not starting yet
                logger.info(f"[MOMENTUM] {pair} skip short: slope={slope:.4f}")
                return False
            
        return True


    def apply_constraints(self, signals, portfolio, timestamp):
        filtered_signals = {}

        for pair, signal in signals.items():

            # Filter by spread volatility
            pass_filter = self.spread_vol_filter(pair, timestamp)
            if not pass_filter:
                continue
            
            # Filter by beta volatility
            pass_filter = self.beta_vol_filter(pair, timestamp)
            if not pass_filter:
                continue

            # Filter by spread momentum
            pass_filter = self.spread_momentum_filter(pair, timestamp, signal)
            if not pass_filter:
                continue

            filtered_signals[pair] = signal

        return filtered_signals
