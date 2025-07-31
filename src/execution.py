from .logger import logger


class ExecutionModel:
    def __init__(self, config):
        self.slippage_bps = config.get("slippage_bps", 1) / 10_000
        self.capital_usage_limit = config.get("capital_usage_limit", 0.5)
        self.capital_per_trade = config.get("capital_per_trade", 0.02)

    def generate_trades(self, signals, current_positions, portfolio, prices):
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

            if px1 <= 0 or px2 <= 0 or beta is None:
                logger.warning(f"[SKIP] {pair}: Invalid price or missing beta")
                continue

            if abs(beta) < 1e-3:
                logger.warning(f"[SKIP] {pair}: beta too small ({beta:.6f})")
                continue

            target_allocation = self.capital_per_trade * equity

            # Determine sizing
            denom = px1 + abs(beta * px2)
            if denom < 1e-6:
                logger.warning(f"[SKIP] {pair}: unstable sizing denominator (denom={denom})")
                continue

            qty1 = target_allocation / denom
            qty2 = -qty1 * beta

            # Round to nearest 1/100th of a share
            qty1 = round(qty1, 2)
            qty2 = round(qty2, 2)

            signal = max(min(signal, 1), -1)
            qty1 *= signal
            qty2 *= signal

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
                logger.error(f"[BIG TRADE] {pair}: Δqty1={delta_leg1}, Δqty2={delta_leg2}, equity={equity:.2f}")

            if abs(delta_leg1) > 1e-6 or abs(delta_leg2) > 1e-6:
                trades[pair] = {
                    "leg1": delta_leg1,
                    "leg2": delta_leg2
                }

                available_capital -= capital_required
                if available_capital <= 0:
                    logger.info("Capital exhausted — stopping further trade generation.")
                    break

        return trades
