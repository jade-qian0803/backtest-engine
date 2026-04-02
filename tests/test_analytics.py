"""Tests for analytics/metrics computation."""

import pytest
import numpy as np
from backtester.analytics import compute_metrics
from backtester.strategy import Trade
import pandas as pd


def make_trade(action, price, shares=100):
    return Trade(
        date=pd.Timestamp("2020-01-01"),
        action=action,
        price=price,
        shares=shares,
        value=price * shares,
    )


class TestMetrics:
    def test_total_return_positive(self):
        equity = [10_000, 10_500, 11_000, 12_000]
        metrics = compute_metrics(equity, [])
        assert metrics["total_return"] == pytest.approx(20.0, rel=0.01)

    def test_total_return_negative(self):
        equity = [10_000, 9_000, 8_000]
        metrics = compute_metrics(equity, [])
        assert metrics["total_return"] == pytest.approx(-20.0, rel=0.01)

    def test_max_drawdown(self):
        # Goes up to 12000 then drops to 9000: drawdown = -25%
        equity = [10_000, 12_000, 10_000, 9_000, 11_000]
        metrics = compute_metrics(equity, [])
        assert metrics["max_drawdown"] == pytest.approx(-25.0, rel=0.01)

    def test_no_drawdown(self):
        equity = [10_000, 11_000, 12_000, 13_000]
        metrics = compute_metrics(equity, [])
        assert metrics["max_drawdown"] == pytest.approx(0.0)

    def test_win_rate(self):
        trades = [
            make_trade("BUY", 100),
            make_trade("SELL", 120),  # win
            make_trade("BUY", 110),
            make_trade("SELL", 90),  # loss
        ]
        metrics = compute_metrics([10_000, 10_000], trades)
        assert metrics["win_rate"] == pytest.approx(50.0)
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1

    def test_sharpe_flat_returns(self):
        # Flat equity = zero std = zero sharpe
        equity = [10_000] * 100
        metrics = compute_metrics(equity, [])
        assert metrics["sharpe_ratio"] == 0.0

    def test_profit_factor(self):
        trades = [
            make_trade("BUY", 100),
            make_trade("SELL", 150),  # profit = 5000
            make_trade("BUY", 140),
            make_trade("SELL", 130),  # loss = 1000
        ]
        metrics = compute_metrics([10_000, 10_000], trades)
        assert metrics["profit_factor"] == pytest.approx(5.0)
