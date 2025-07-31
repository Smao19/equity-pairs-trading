from .logger import logger


class Strategy:
    def __init__(self, config):
        self.entry_threshold = config.get("entry_threshold", 1.0)
        self.exit_threshold = config.get("exit_threshold", 0.2)

    def generate_signals(self, current_prices, positions):
        signals = {}
        for pair, row in current_prices.items():
            z = row["zscore"]
            pos = positions.get(pair, {}).get("leg1", 0)

            logger.debug(f"Signal logic: {pair} | z={z:.2f}, pos={pos}")

            # No position - check entry condition
            if pos == 0:
                if z > self.entry_threshold:
                    signals[pair] = -1  # Short spread
                elif z < -self.entry_threshold:
                    signals[pair] = 1   # Long spread

            # Long position - check for exit
            elif pos > 0:
                if z > -self.exit_threshold:
                    signals[pair] = 0  # Exit

            # Short position - check for exit
            elif pos < 0:
                if z < self.exit_threshold:
                    signals[pair] = 0  # Exit

            # Else, do not issue redundant signal (implicitly holding)

        return signals
