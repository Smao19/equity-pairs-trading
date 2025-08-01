from .logger import logger


class Portfolio:
    def __init__(self, config):
        self.initial_cash = config.get("initial_cash", 1_000_000)
        self.cash = self.initial_cash
        self.leg_positions = {}  # {pair: {"leg1": qty1, "leg2": qty2, "entry_px1": px1, "entry_px2": px2, "beta": beta}}

        self.realized_pnl = 0
        self.realized_pnl_history = []
        self.unrealized_pnl_history = []
        self.equity_history = []
        self.timestamps = []

        self.commission_rate = config.get("commission_bps", 5) / 10_000
        self.total_commission_paid = 0

        # Stored for visualizations
        self.pair_realized_pnl_curves = {}  # {pair: [realized_pnl_t]}
        self.pair_timestamps = {}  # {pair: [timestamp_t]}

        # Stored for performance metrics
        self.holding_periods = []  # List of durations (in minutes) across all pairs
        self.pair_holding_periods = {}  # {pair: [durations]}
        self.current_holding_start = {}  # {pair: timestamp when position was opened}
        

    def update(self, trades, prices, timestamp):
        logger.debug(f"[{timestamp}] Applying trades: {trades}")
        pnl = 0  # This will be unrealized only

        # Execute trades
        for pair, trade in trades.items():
            px1 = prices[pair]["px1"]
            px2 = prices[pair]["px2"]
            beta = prices[pair]["beta"]
            qty1 = trade["leg1"]
            qty2 = trade["leg2"]

            notional1 = abs(qty1 * px1)
            notional2 = abs(qty2 * px2)
            commission = (notional1 + notional2) * self.commission_rate

            self.cash -= commission
            self.total_commission_paid += commission

            if pair not in self.leg_positions:
                self.leg_positions[pair] = {
                    "leg1": 0,
                    "leg2": 0,
                    "entry_px1": px1,
                    "entry_px2": px2,
                    "beta": beta
                }

            pos = self.leg_positions[pair]
            
            # Check if position is opening for holding time tracking
            if pair not in self.current_holding_start:
                if qty1 != 0 or qty2 != 0:
                    self.current_holding_start[pair] = timestamp

            # LEG 1
            curr_qty1 = pos["leg1"]
            new_qty1 = curr_qty1 + qty1

            if curr_qty1 != 0 and (curr_qty1 * new_qty1 < 0 or abs(new_qty1) < abs(curr_qty1)):
                # Closing or reducing
                closing_qty1 = -qty1 if curr_qty1 * qty1 > 0 else curr_qty1
                realized1 = closing_qty1 * (px1 - pos["entry_px1"])
                self.realized_pnl += realized1
                logger.debug(f"[{pair}] Realized PnL leg1: {realized1:.2f}")
            
            pos["leg1"] = new_qty1
            if new_qty1 != 0:
                pos["entry_px1"] = px1
            pos["beta"] = beta

            # LEG 2
            curr_qty2 = pos["leg2"]
            new_qty2 = curr_qty2 + qty2

            if curr_qty2 != 0 and (curr_qty2 * new_qty2 < 0 or abs(new_qty2) < abs(curr_qty2)):
                closing_qty2 = -qty2 if curr_qty2 * qty2 > 0 else curr_qty2
                realized2 = closing_qty2 * (px2 - pos["entry_px2"])
                self.realized_pnl += realized2
                logger.debug(f"[{pair}] Realized PnL leg2: {realized2:.2f}")

            pos["leg2"] = new_qty2
            if new_qty2 != 0:
                pos["entry_px2"] = px2

            # Check if position is now fully closed
            pos_leg1 = pos["leg1"]
            pos_leg2 = pos["leg2"]

            if abs(pos_leg1) < 1e-6 and abs(pos_leg2) < 1e-6:
                start = self.current_holding_start.pop(pair, None)
                if start is not None:
                    duration = (timestamp - start).total_seconds() / 60  # duration in minutes

                    self.holding_periods.append(duration)
                    self.pair_holding_periods.setdefault(pair, []).append(duration)

                    logger.debug(f"[{pair}] Closed position — held for {duration:.2f} minutes")

            # Update pair-specific curves
            self.pair_realized_pnl_curves.setdefault(pair, []).append(self.realized_pnl)
            self.pair_timestamps.setdefault(pair, []).append(timestamp)

        # UNREALIZED PNL
        unrealized_pnl = self.get_unrealized_pnl(prices)
        total_equity = self.cash + self.realized_pnl + unrealized_pnl

        self.realized_pnl_history.append(self.realized_pnl)
        self.unrealized_pnl_history.append(unrealized_pnl)
        self.equity_history.append(total_equity)
        self.timestamps.append(timestamp)

        logger.info(f"[{timestamp}] Equity: {total_equity:.2f}, Realized PnL: {self.realized_pnl:.2f}, "
                    f"Unrealized PnL: {unrealized_pnl:.2f}, Cash: {self.cash:.2f}, Commission: {self.total_commission_paid:.2f}")

    def get_equity(self):
        return self.cash + self.realized_pnl + self.get_unrealized_pnl_latest()

    def get_unrealized_pnl(self, prices):
        unrealized = 0
        for pair, pos in self.leg_positions.items():
            if pair in prices:
                px1 = prices[pair]["px1"]
                px2 = prices[pair]["px2"]
                unrealized += pos["leg1"] * (px1 - pos["entry_px1"])
                unrealized += pos["leg2"] * (px2 - pos["entry_px2"])
        return unrealized

    def get_unrealized_pnl_latest(self):
        return self.unrealized_pnl_history[-1] if self.unrealized_pnl_history else 0

    def get_current_gross_exposure(self, prices):
        total = 0
        for pair, pos in self.leg_positions.items():
            if pair in prices:
                px1 = prices[pair]["px1"]
                px2 = prices[pair]["px2"]
                total += abs(pos["leg1"] * px1) + abs(pos["leg2"] * px2)
        return total
