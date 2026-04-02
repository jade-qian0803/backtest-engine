"""Tests for technical indicators using hand-calculated values."""

import pandas as pd
import numpy as np
import pytest
from backtester.indicators import sma, ema, rsi, macd, bollinger_bands


@pytest.fixture
def sample_series():
    """Simple price series for hand-verifiable calculations."""
    return pd.Series([10, 11, 12, 13, 14, 15, 14, 13, 12, 11])


class TestSMA:
    def test_sma_basic(self, sample_series):
        result = sma(sample_series, period=3)
        # First two values should be NaN
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        # SMA of [10, 11, 12] = 11.0
        assert result.iloc[2] == pytest.approx(11.0)
        # SMA of [11, 12, 13] = 12.0
        assert result.iloc[3] == pytest.approx(12.0)

    def test_sma_full_window(self, sample_series):
        result = sma(sample_series, period=10)
        # Only last value should be non-NaN
        assert np.isnan(result.iloc[8])
        assert result.iloc[9] == pytest.approx(12.5)

    def test_sma_period_1(self, sample_series):
        result = sma(sample_series, period=1)
        pd.testing.assert_series_equal(result, sample_series.astype(float))


class TestEMA:
    def test_ema_basic(self, sample_series):
        result = ema(sample_series, period=3)
        # EMA should not have NaN values (ewm handles warmup)
        assert not result.isna().any()
        # First value equals the first data point
        assert result.iloc[0] == pytest.approx(10.0)

    def test_ema_responds_to_trend(self, sample_series):
        result = ema(sample_series, period=3)
        # During uptrend (index 0-5), EMA should be rising
        assert result.iloc[4] > result.iloc[2]


class TestRSI:
    def test_rsi_range(self, sample_series):
        result = rsi(sample_series, period=5)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_uptrend(self):
        # Pure uptrend should have RSI near 100
        up = pd.Series(range(1, 30))
        result = rsi(up, period=14)
        assert result.iloc[-1] > 90

    def test_rsi_downtrend(self):
        # Pure downtrend should have RSI near 0
        down = pd.Series(range(30, 1, -1))
        result = rsi(down, period=14)
        assert result.iloc[-1] < 10


class TestMACD:
    def test_macd_returns_three_series(self, sample_series):
        macd_line, signal_line, histogram = macd(sample_series)
        assert len(macd_line) == len(sample_series)
        assert len(signal_line) == len(sample_series)
        assert len(histogram) == len(sample_series)

    def test_histogram_equals_diff(self, sample_series):
        macd_line, signal_line, histogram = macd(sample_series)
        expected = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected)


class TestBollingerBands:
    def test_bands_relationship(self, sample_series):
        upper, middle, lower = bollinger_bands(sample_series, period=5)
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_middle_equals_sma(self, sample_series):
        upper, middle, lower = bollinger_bands(sample_series, period=5)
        expected_sma = sma(sample_series, period=5)
        pd.testing.assert_series_equal(middle, expected_sma)
