from .logger import logger


def determine_sizing(target_allocation, px1, px2, beta, pair):
    """
    Given target capital allocation, prices, and beta:
    calculates appropriate quantities of each pair ticker for trade.
    """

    denom = px1 + abs(beta * px2)
    if denom < 1e-6:
        logger.warning(f"[SKIP] {pair}: unstable sizing denominator (denom={denom})")
        return (None, None)  # Signifies skip the pair

    qty1 = target_allocation / denom
    qty2 = -qty1 * beta

    # Round to nearest 1/100th of a share
    qty1 = round(qty1, 2)
    qty2 = round(qty2, 2)

    return qty1, qty2


class ExecutionModel:

    MAX_CONFIDENCE_MULIPLIER = 3
    MAX_RISK_MULTIPLIER = 5

    def __init__(self, config):
        self.slippage_bps = config.get("slippage_bps", 1) / 10_000
        self.capital_usage_limit = config.get("capital_usage_limit", 0.5)
        self.base_allocation = config.get("base_allocation", 0.005)
        self.entry_threshold = config.get("entry_threshold", 1)

        # Stored for access to spread volatility values
        self.pair_data = {}  # filled externally


    def calculate_weighted_allocation(self, portfolio, signal, pair, timestamp):
        """
        Calculates safely capped risk weighted factor (via spread volatility) 
        and confidence weighted factor (via signal excess).
        Returns base_allocation increased by total factor of the weights
        """
        
        base_allocation = self.base_allocation * portfolio.get_equity()

        # Calculate confidence weight
        confidence = max(0, abs(signal) - self.entry_threshold)
        confidence_multiplier = min(max(1, confidence / 1.2), self.MAX_CONFIDENCE_MULIPLIER)  # caps at 3x

        # Pull precomputed rolling spread volatility from df1
        df1, _ = self.pair_data.get(pair, (None, None))
        if df1 is None or timestamp not in df1.index:
            risk_multiplier = 1.0
        else:
            vol_value = df1.loc[timestamp, 'spread_vol']
            if vol_value is None:
                vol_value = 1
            vol_value = max(vol_value, 1e-4)
            # Vol range for multipliers: 0.15 - 0.03 *caps at 0.03 
            risk_multiplier = min(max(0.15 / vol_value, 1), self.MAX_RISK_MULTIPLIER)  # caps at 5x

        return base_allocation * confidence_multiplier * risk_multiplier


    def generate_trades(self, signals, current_positions, portfolio, prices, timestamp):
        trades = {}

        equity = portfolio.get_equity()
        used_capital = portfolio.get_current_gross_exposure(prices)
        capital_limit = self.capital_usage_limit * equity
        available_capital = max(0, capital_limit - used_capital)

        logger.debug(f"Equity: {equity:.2f}, Used capital: {used_capital:.2f}, Available: {available_capital:.2f}")

        for pair, signal in signals.items():
            if pair not in prices:
                continue

            px1 = prices[pair]["px1"]
            px2 = prices[pair]["px2"]
            beta = prices[pair]["beta"]

            # Skip if any data is undesirable to work with
            if px1 <= 0 or px2 <= 0 or beta is None:
                logger.warning(f"[SKIP] {pair}: Invalid price or missing beta")
                continue

            if abs(beta) < 1e-3:
                logger.warning(f"[SKIP] {pair}: beta too small ({beta:.6f})")
                continue

            # Determine position sizing and apply direction
            target_allocation = self.calculate_weighted_allocation(portfolio, signal, pair, timestamp)
            qty1, qty2 = determine_sizing(target_allocation, px1, px2, beta, pair)
            
            if qty1 is None or qty1 == 0:
                continue

            signal = max(min(signal, 1), -1)
            qty1 *= signal
            qty2 *= signal

            # Compute leg specific delta's
            curr_leg1 = current_positions.get(pair, {}).get("leg1", 0)
            curr_leg2 = current_positions.get(pair, {}).get("leg2", 0)

            new_leg1 = qty1
            new_leg2 = qty2

            delta_leg1 = new_leg1 - curr_leg1
            delta_leg2 = new_leg2 - curr_leg2

            # Determine if trade is reducing or closing
            reducing_leg1 = abs(new_leg1) < abs(curr_leg1) or (curr_leg1 * new_leg1 < 0)
            reducing_leg2 = abs(new_leg2) < abs(curr_leg2) or (curr_leg2 * new_leg2 < 0)
            is_reducing = reducing_leg1 or reducing_leg2

            capital_required = abs(delta_leg1 * px1) + abs(delta_leg2 * px2)

            if capital_required > available_capital and not is_reducing:
                logger.info(f"[SKIP] {pair}: not enough capital for opening/increasing trade")
                continue

            if abs(delta_leg1) > 1e6 or abs(delta_leg2) > 1e6:
                logger.error(f"[BIG TRADE] {pair}: delta_qty1={delta_leg1}, delta_qty2={delta_leg2}, equity={equity:.2f}")

            if abs(delta_leg1) > 1e-6 or abs(delta_leg2) > 1e-6:  # Generate the trade
                trades[pair] = {
                    "leg1": delta_leg1,
                    "leg2": delta_leg2
                }

                available_capital -= capital_required
                if available_capital <= 0:
                    logger.info("Capital exhausted — stopping further trade generation.")
                    break

        return trades
