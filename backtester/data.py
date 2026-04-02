"""Data fetching and cleaning module."""

import time
import pandas as pd
import yfinance as yf


def fetch_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch OHLCV data for a given ticker and date range.

    Args:
        ticker: Stock/ETF/crypto symbol (e.g. "AAPL", "BTC-USD").
        start: Start date string (e.g. "2020-01-01").
        end: End date string (e.g. "2023-12-31").
        interval: Data interval ("1d", "1h", etc.).
        max_retries: Number of retries on rate limit.

    Returns:
        DataFrame with columns: open, high, low, close, volume.
        Index is DatetimeIndex.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            raw = t.history(start=start, end=end, interval=interval)
            break
        except Exception as e:
            last_error = e
            if "Rate" in str(e) and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise ValueError(
                f"Failed to fetch {ticker}: {e}. "
                "Yahoo Finance may be rate-limiting. Try again in a minute."
            ) from e
    else:
        raise ValueError(f"Failed after {max_retries} retries: {last_error}")

    if raw.empty:
        raise ValueError(
            f"No data returned for {ticker} from {start} to {end}. "
            "Check the ticker symbol and date range."
        )

    # Normalize column names to lowercase
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    col_map = {}
    for needed in ["open", "high", "low", "close", "volume"]:
        matches = [c for c in raw.columns if needed in c]
        if matches:
            col_map[needed] = matches[0]
        else:
            raise ValueError(
                f"Column '{needed}' not found in data: {list(raw.columns)}"
            )

    df = pd.DataFrame(
        {
            "open": raw[col_map["open"]],
            "high": raw[col_map["high"]],
            "low": raw[col_map["low"]],
            "close": raw[col_map["close"]],
            "volume": raw[col_map["volume"]],
        }
    )

    df.index.name = "date"
    # Remove timezone info for cleaner display
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna()
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file as a fallback."""
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df.index.name = "date"
    return df[required].dropna()


def validate_data(df: pd.DataFrame) -> list[str]:
    """Run basic quality checks on OHLCV data.

    Returns a list of warning messages. Empty list means data is clean.
    """
    warnings = []

    if df.isnull().any().any():
        warnings.append("Data contains NaN values")

    if (df["close"] <= 0).any():
        warnings.append("Data contains non-positive close prices")

    # Check for large gaps (>50% daily change)
    pct_change = df["close"].pct_change().abs()
    big_gaps = pct_change[pct_change > 0.5]
    if not big_gaps.empty:
        warnings.append(f"Data contains {len(big_gaps)} large price gaps (>50%)")

    return warnings
