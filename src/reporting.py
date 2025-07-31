import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReportGenerator:

    def visualize_simulation(self, timestamps, equity, realized, unrealized):
        _, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Equity Curve
        axs[0].plot(timestamps, equity / 1e6, label="Equity", color="blue")  # scaled down as ylabel cleany conveys scale
        axs[0].set_title("Equity Curve")
        axs[0].set_ylabel("Equity ($M)")
        axs[0].grid(True)
        axs[0].legend()

        # Realized & Unrealized PnL
        axs[1].plot(timestamps, realized, label="Realized PnL", color="orange")
        axs[1].plot(timestamps, unrealized, label="Unrealized PnL", color="green")
        axs[1].set_title("PnL Breakdown")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("PnL ($)")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()


    def compute_metrics(self, timestamps, equity, realized, portfolio, debug=False):
        # Return Analysis
        equity_series = pd.Series(equity, index=timestamps)
        daily_equity = equity_series.resample("1D").last().dropna()
        returns = daily_equity.pct_change().dropna()
        
        if debug:
            print(f"[DEBUG] Number of daily return points: {len(returns)}")
            print(f"[DEBUG] Sample daily returns: {returns.head()}")

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 else 0

        # Drawdowns
        rolling_max = daily_equity.cummax()
        drawdowns = 1 - daily_equity / rolling_max
        max_drawdown = drawdowns.max()
        drawdown_durations = (drawdowns != 0).astype(int).groupby((drawdowns == 0).astype(int).cumsum()).sum()
        longest_drawdown = drawdown_durations.max()

        if debug:
            print(f"[DEBUG] Max Drawdown: {max_drawdown}")
            print(f"[DEBUG] Drawdown Durations: {drawdown_durations}")

        # Win Rate Calculation
        # Approximate by counting upswings in realized PnL history
        realized_changes = np.diff(realized)
        if debug:
           print(f"[DEBUG] Realized PnL Deltas (sample): {realized_changes[:5]}")
        wins = (realized_changes > 0).sum()
        losses = (realized_changes < 0).sum()
        win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Average holding time (in minutes)
        avg_holding = np.mean(portfolio.holding_periods) if portfolio.holding_periods else 0
        if debug:
            print(f"[DEBUG] System-wide avg holding time: {avg_holding:.2f} minutes")

        return (
            f"{sharpe:.3f}",
            f"{sortino:.3f}", 
            f"{max_drawdown:.4%}", 
            f"{longest_drawdown}", 
            f"{win_pct:.2%}", 
            f"{avg_holding:.2f}",
        )


    def plot_pair_pnls(self, pair_realized_pnls, portfolio, title, ylabel):
        _, ax = plt.subplots(figsize=(12, 5), sharex=True)

        for label, series in pair_realized_pnls.items():
            ts = pd.to_datetime(portfolio.pair_timestamps[label])  # Actual timestamps per pair
            ax.plot(ts, series, label=str(label))

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        ax.legend(fontsize="small", loc="upper left")
        ax.grid(True)

        plt.tight_layout()
        plt.show()


    def evaluate_pair_performance(self, portfolio, debug=False):
        final_pnl = {}

        for pair, pnl_series in portfolio.pair_realized_pnl_curves.items():
            if pnl_series:
                final_pnl[pair] = pnl_series[-1]

        if not final_pnl:
            print("No pair-level equity data to evaluate.")
            return

        best_pair = max(final_pnl, key=final_pnl.get)
        worst_pair = min(final_pnl, key=final_pnl.get)

        print("\nBest Performing Pair:", best_pair)
        print("Worst Performing Pair:", worst_pair)

        pair_summary = {}

        for pair in [best_pair, worst_pair]:
            pnl = portfolio.pair_realized_pnl_curves[pair]
            ts = portfolio.pair_timestamps[pair]

            sharpe, sortino, _, ldd, win, _ = self.compute_metrics(ts, pnl, pnl, portfolio, debug=debug)
            avg_hold = np.mean(portfolio.pair_holding_periods.get(pair, [])) if portfolio.pair_holding_periods.get(pair) else 0

            print(f"\n{pair} Metrics:")
            print(f"  Sharpe Ratio: {sharpe}")
            print(f"  Sortino Ratio: {sortino}")
            print(f"  Longest Drawdown Duration (days): {ldd}")
            print(f"  Win Percentage: {win}")
            print(f"  Avg Holding Time: {avg_hold:.2f} minutes")

            pair_summary[pair] = pnl

        # Plot using helper
        self.plot_pair_pnls(
            pair_realized_pnls=pair_summary,
            portfolio=portfolio,
            title="Best/Worst Pair Realized PnL Curves",
            ylabel="Realized PnL ($)"
        )


    def generate(self, portfolio):
        timestamps = pd.to_datetime(portfolio.timestamps)
        equity = np.array(portfolio.equity_history)
        realized = np.array(portfolio.realized_pnl_history)
        unrealized = np.array(portfolio.unrealized_pnl_history)

        final_equity = equity[-1]
        total_return = (final_equity - portfolio.initial_cash) / portfolio.initial_cash
        final_cash = portfolio.cash
        realized_pnl = realized[-1]
        unrealized_pnl = unrealized[-1]

        print(f"\nPortfolio Metrics:")
        print(f"  Final Equity: ${round(final_equity, 2)}")
        print(f"  Total Return: {round(total_return*100, 3)}%")
        print(f"  Realized PnL: ${round(realized_pnl, 2)}")
        print(f"  Unrealized PnL: ${round(unrealized_pnl, 2)}")
        print(f"  Final Cash: ${round(final_cash, 2)}")
        print(f"  Total Commission Paid: ${round(portfolio.total_commission_paid, 2)}")

        # Report performance metrics
        sharpe, sortino, _, longest_drawdown, win_pct, avg_holding = self.compute_metrics(timestamps, equity, realized, portfolio)
        print(f"  Sharpe Ratio: {sharpe}")
        print(f"  Sortino Ratio: {sortino}")
        print(f"  Longest Drawdown Duration (days): {longest_drawdown}")
        print(f"  Win Percentage: {win_pct}")
        print(f"  Avg Holding Time: {avg_holding} minutes")

        # Visualization
        self.visualize_simulation(timestamps, equity, realized, unrealized)

        # Report best/worst pair performance metrics & visualization
        self.evaluate_pair_performance(portfolio)
