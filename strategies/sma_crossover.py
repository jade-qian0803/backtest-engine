"""SMA Crossover (Trend Following) strategy."""

from backtester.strategy import Strategy, Context


class SmaCrossover(Strategy):
    """Buy when short SMA crosses above long SMA, sell when it crosses below.

    This is a classic trend-following strategy.
    """

    def __init__(self, short_period: int = 20, long_period: int = 50):
        self.short_period = short_period
        self.long_period = long_period

    def initialize(self, ctx: Context):
        ctx.state["prev_short"] = None
        ctx.state["prev_long"] = None

    def on_bar(self, ctx: Context):
        # Need enough data for the long SMA
        if ctx.i < self.long_period:
            return

        short_sma = ctx.indicator("sma", period=self.short_period)
        long_sma = ctx.indicator("sma", period=self.long_period)

        prev_short = ctx.state["prev_short"]
        prev_long = ctx.state["prev_long"]

        if prev_short is not None and prev_long is not None:
            # Golden cross: short crosses above long
            if prev_short <= prev_long and short_sma > long_sma:
                if ctx.portfolio.shares == 0:
                    ctx.buy()

            # Death cross: short crosses below long
            elif prev_short >= prev_long and short_sma < long_sma:
                if ctx.portfolio.shares > 0:
                    ctx.sell()

        ctx.state["prev_short"] = short_sma
        ctx.state["prev_long"] = long_sma
