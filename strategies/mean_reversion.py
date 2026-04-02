"""Mean Reversion strategy using RSI."""

from backtester.strategy import Strategy, Context


class MeanReversion(Strategy):
    """Buy when RSI indicates oversold, sell when overbought.

    Classic mean-reversion strategy assuming prices revert to the mean.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def initialize(self, ctx: Context):
        pass

    def on_bar(self, ctx: Context):
        if ctx.i < self.rsi_period + 1:
            return

        current_rsi = ctx.indicator("rsi", period=self.rsi_period)

        if current_rsi < self.oversold and ctx.portfolio.shares == 0:
            ctx.buy()
        elif current_rsi > self.overbought and ctx.portfolio.shares > 0:
            ctx.sell()
