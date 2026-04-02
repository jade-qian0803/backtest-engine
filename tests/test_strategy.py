"""Tests for strategy context and order execution."""

import pandas as pd
import numpy as np
import pytest
from backtester.strategy import Context


def make_ctx(prices: list[float], cash: float = 10_000) -> Context:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000] * len(prices),
        },
        index=dates,
    )
    return Context(df=df, cash=cash)


class TestContext:
    def test_current_bar_properties(self):
        ctx = make_ctx([100, 200, 300])
        ctx.i = 1
        assert ctx.close == 200
        assert ctx.open == 200

    def test_history(self):
        ctx = make_ctx([10, 20, 30, 40, 50])
        ctx.i = 3
        hist = ctx.history(2)
        assert len(hist) == 2
        assert hist["close"].iloc[-1] == 40

    def test_buy_full(self):
        ctx = make_ctx([100])
        ctx.i = 0
        ctx.buy()
        assert ctx.portfolio.cash == pytest.approx(0)
        assert ctx.portfolio.shares == pytest.approx(100)
        assert len(ctx.trades) == 1

    def test_buy_partial(self):
        ctx = make_ctx([100])
        ctx.i = 0
        ctx.buy(percent=0.5)
        assert ctx.portfolio.cash == pytest.approx(5_000)
        assert ctx.portfolio.shares == pytest.approx(50)

    def test_sell_full(self):
        ctx = make_ctx([100, 200])
        ctx.i = 0
        ctx.buy()
        ctx.i = 1
        ctx.sell()
        assert ctx.portfolio.shares == pytest.approx(0)
        assert ctx.portfolio.cash == pytest.approx(20_000)

    def test_sell_when_empty(self):
        ctx = make_ctx([100])
        ctx.i = 0
        ctx.sell()  # should do nothing
        assert len(ctx.trades) == 0
        assert ctx.portfolio.cash == pytest.approx(10_000)

    def test_buy_invalid_percent(self):
        ctx = make_ctx([100])
        ctx.i = 0
        with pytest.raises(ValueError):
            ctx.buy(percent=0)
        with pytest.raises(ValueError):
            ctx.buy(percent=1.5)

    def test_indicator_sma(self):
        ctx = make_ctx([10, 20, 30, 40, 50])
        ctx.i = 4
        val = ctx.indicator("sma", period=3)
        # SMA of [30, 40, 50] = 40
        assert val == pytest.approx(40.0)

    def test_indicator_unknown(self):
        ctx = make_ctx([10, 20, 30])
        ctx.i = 2
        with pytest.raises(ValueError, match="Unknown indicator"):
            ctx.indicator("foobar", period=3)
