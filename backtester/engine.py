"""Backtesting engine — runs a strategy against historical data."""

import pandas as pd
from backtester.strategy import Strategy, Context


class Backtest:
    """Event-driven backtesting engine.

    Usage:
        bt = Backtest(data=df, strategy=MyStrategy(), cash=10000)
        results = bt.run()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        cash: float = 10_000,
    ):
        if data.empty:
            raise ValueError("Data is empty")
        if cash <= 0:
            raise ValueError("Starting cash must be positive")

        self.data = data
        self.strategy = strategy
        self.cash = cash

    def run(self) -> dict:
        """Execute the backtest.

        Returns:
            dict with keys:
                - trades: list of Trade objects
                - equity_curve: list of portfolio values per bar
                - ctx: the final Context (for further inspection)
        """
        ctx = Context(df=self.data, cash=self.cash)

        # Let the strategy set up indicators / state
        self.strategy.initialize(ctx)

        # Step through each bar
        for i in range(len(self.data)):
            ctx.i = i
            ctx.portfolio.update_value(ctx.close)

            self.strategy.on_bar(ctx)

            ctx.portfolio.update_value(ctx.close)
            ctx.equity_curve.append(ctx.portfolio.total_value)

        return {
            "trades": ctx.trades,
            "equity_curve": ctx.equity_curve,
            "ctx": ctx,
        }
