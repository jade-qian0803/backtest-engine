"""Strategy base class and execution context."""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from backtester import indicators as ind


@dataclass
class Trade:
    """Record of a single trade."""

    date: pd.Timestamp
    action: str  # "BUY" or "SELL"
    price: float
    shares: float
    value: float  # total value of the trade


@dataclass
class Portfolio:
    """Tracks cash, holdings, and portfolio value."""

    cash: float
    shares: float = 0.0
    total_value: float = 0.0

    def update_value(self, current_price: float):
        self.total_value = self.cash + self.shares * current_price


class Context:
    """Execution context passed to strategy functions on each bar.

    Provides:
        - Current bar data (ctx.open, ctx.close, etc.)
        - Full DataFrame access (ctx.df)
        - Current bar index (ctx.i)
        - Portfolio state (ctx.portfolio)
        - Indicator helper (ctx.indicator)
        - Order methods (ctx.buy, ctx.sell)
    """

    def __init__(self, df: pd.DataFrame, cash: float):
        self.df = df.copy()
        self.portfolio = Portfolio(cash=cash)
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []

        # Current bar index — set by engine
        self.i: int = 0

        # User-defined storage
        self.state: dict = {}

    # -- Current bar properties --

    @property
    def open(self) -> float:
        return self.df["open"].iloc[self.i]

    @property
    def high(self) -> float:
        return self.df["high"].iloc[self.i]

    @property
    def low(self) -> float:
        return self.df["low"].iloc[self.i]

    @property
    def close(self) -> float:
        return self.df["close"].iloc[self.i]

    @property
    def volume(self) -> float:
        return self.df["volume"].iloc[self.i]

    @property
    def date(self) -> pd.Timestamp:
        return self.df.index[self.i]

    # -- Data access --

    def history(self, n: int) -> pd.DataFrame:
        """Get the last n bars up to and including the current bar."""
        start = max(0, self.i - n + 1)
        return self.df.iloc[start : self.i + 1]

    # -- Indicators --

    def indicator(self, name: str, **kwargs) -> float | tuple:
        """Compute an indicator and return its current value.

        Args:
            name: Indicator name ("sma", "ema", "rsi", "macd", "bollinger_bands").
            **kwargs: Indicator parameters (e.g. period=50).

        Returns:
            Current value of the indicator at bar self.i.
        """
        func = ind.INDICATORS.get(name)
        if func is None:
            raise ValueError(
                f"Unknown indicator '{name}'. "
                f"Available: {list(ind.INDICATORS.keys())}"
            )

        # Default to close prices if 'series' not specified
        series = self.df["close"].iloc[: self.i + 1]
        result = func(series, **kwargs)

        if isinstance(result, tuple):
            return tuple(r.iloc[-1] if len(r) > 0 else np.nan for r in result)
        return result.iloc[-1] if len(result) > 0 else np.nan

    # -- Orders --

    def buy(self, percent: float = 1.0):
        """Buy shares using a percentage of available cash.

        Args:
            percent: Fraction of cash to use (0.0 to 1.0).
        """
        if percent <= 0 or percent > 1:
            raise ValueError("percent must be between 0 (exclusive) and 1 (inclusive)")

        price = self.close
        if price <= 0:
            return

        available = self.portfolio.cash * percent
        shares = available / price

        if shares <= 0:
            return

        self.portfolio.cash -= shares * price
        self.portfolio.shares += shares
        self.trades.append(
            Trade(
                date=self.date,
                action="BUY",
                price=price,
                shares=shares,
                value=shares * price,
            )
        )

    def sell(self, percent: float = 1.0):
        """Sell a percentage of current holdings.

        Args:
            percent: Fraction of shares to sell (0.0 to 1.0).
        """
        if percent <= 0 or percent > 1:
            raise ValueError("percent must be between 0 (exclusive) and 1 (inclusive)")

        if self.portfolio.shares <= 0:
            return

        price = self.close
        shares_to_sell = self.portfolio.shares * percent

        self.portfolio.cash += shares_to_sell * price
        self.portfolio.shares -= shares_to_sell
        self.trades.append(
            Trade(
                date=self.date,
                action="SELL",
                price=price,
                shares=shares_to_sell,
                value=shares_to_sell * price,
            )
        )


class Strategy:
    """Base class for user-defined strategies.

    Users subclass this and implement:
        - initialize(ctx): Called once before backtesting starts.
        - on_bar(ctx): Called on each time step.
    """

    def initialize(self, ctx: Context):
        """Called once before the backtest begins. Override this."""
        pass

    def on_bar(self, ctx: Context):
        """Called on each bar. Override this to define trading logic."""
        raise NotImplementedError("You must implement on_bar()")
