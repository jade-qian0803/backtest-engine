"""Tests for the backtesting engine."""

import pandas as pd
import numpy as np
import pytest
from backtester.engine import Backtest
from backtester.strategy import Strategy, Context


def make_data(prices: list[float]) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from a list of close prices."""
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        },
        index=dates,
    )


class BuyAndHold(Strategy):
    """Buy on first bar, hold forever."""

    def on_bar(self, ctx: Context):
        if ctx.i == 0:
            ctx.buy()


class SellImmediately(Strategy):
    """Buy on bar 0, sell on bar 1."""

    def on_bar(self, ctx: Context):
        if ctx.i == 0:
            ctx.buy()
        elif ctx.i == 1:
            ctx.sell()


class NeverTrade(Strategy):
    """Does nothing."""

    def on_bar(self, ctx: Context):
        pass


class TestBacktest:
    def test_buy_and_hold_profit(self):
        # Price goes from 100 to 200
        prices = [100 + i * 10 for i in range(11)]
        bt = Backtest(data=make_data(prices), strategy=BuyAndHold(), cash=10_000)
        result = bt.run()

        assert len(result["trades"]) == 1
        assert result["trades"][0].action == "BUY"
        # Final value should be ~2x starting cash
        assert result["equity_curve"][-1] == pytest.approx(20_000, rel=0.01)

    def test_buy_and_hold_loss(self):
        # Price drops from 100 to 50
        prices = [100 - i * 5 for i in range(11)]
        bt = Backtest(data=make_data(prices), strategy=BuyAndHold(), cash=10_000)
        result = bt.run()

        assert result["equity_curve"][-1] == pytest.approx(5_000, rel=0.01)

    def test_no_trades(self):
        prices = [100] * 10
        bt = Backtest(data=make_data(prices), strategy=NeverTrade(), cash=10_000)
        result = bt.run()

        assert len(result["trades"]) == 0
        assert all(v == pytest.approx(10_000) for v in result["equity_curve"])

    def test_round_trip(self):
        prices = [100, 110, 100, 100, 100]
        bt = Backtest(data=make_data(prices), strategy=SellImmediately(), cash=10_000)
        result = bt.run()

        assert len(result["trades"]) == 2
        assert result["trades"][0].action == "BUY"
        assert result["trades"][1].action == "SELL"
        # Bought at 100, sold at 110 => 10% profit
        assert result["equity_curve"][-1] == pytest.approx(11_000, rel=0.01)

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Backtest(data=pd.DataFrame(), strategy=BuyAndHold(), cash=10_000)

    def test_zero_cash_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Backtest(data=make_data([100]), strategy=BuyAndHold(), cash=0)

    def test_equity_curve_length(self):
        prices = [100] * 50
        bt = Backtest(data=make_data(prices), strategy=NeverTrade(), cash=10_000)
        result = bt.run()
        assert len(result["equity_curve"]) == 50
