# Backtest Engine

A quantitative backtesting platform for testing trading strategies against historical market data. Write your strategy in Python, run it in the browser, and see results instantly.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/jade-qian0803/backtest-engine.git
cd backtest-engine
```

### 2. Install dependencies

Requires Python 3.10+.

```bash
pip install -e .
```

For development (testing + formatting):

```bash
pip install -e ".[dev]"
```

### 3. Run the app

```bash
streamlit run app/main.py
```

This opens a web UI where you can:
- Pick a stock ticker and date range
- Write or edit a strategy in the code editor
- Hit **Run Backtest** to see performance metrics, charts, and a trade log

## Writing a Strategy

Subclass `Strategy` and implement two methods:

```python
class MyStrategy(Strategy):
    def initialize(self, ctx):
        # Called once before the backtest starts.
        # Use ctx.state to store any variables you need.
        ctx.state["period"] = 20

    def on_bar(self, ctx):
        # Called on every bar (each day of data).
        # Use ctx to access data, indicators, and place orders.
        if ctx.i < ctx.state["period"]:
            return

        sma = ctx.indicator("sma", period=ctx.state["period"])

        if ctx.close > sma and ctx.portfolio.shares == 0:
            ctx.buy()
        elif ctx.close < sma and ctx.portfolio.shares > 0:
            ctx.sell()
```

### Context (`ctx`) Reference

| Property / Method | Description |
|---|---|
| `ctx.open`, `ctx.high`, `ctx.low`, `ctx.close`, `ctx.volume` | Current bar OHLCV data |
| `ctx.date` | Current bar timestamp |
| `ctx.i` | Current bar index |
| `ctx.df` | Full price DataFrame |
| `ctx.history(n)` | Last `n` bars up to current bar |
| `ctx.indicator(name, **kwargs)` | Compute a technical indicator (see below) |
| `ctx.buy(percent=1.0)` | Buy using a fraction of available cash |
| `ctx.sell(percent=1.0)` | Sell a fraction of current holdings |
| `ctx.portfolio.cash` | Current cash balance |
| `ctx.portfolio.shares` | Current share count |
| `ctx.state` | Dict for storing custom variables |

### Available Indicators

Use via `ctx.indicator("name", ...)`:

| Indicator | Parameters | Returns |
|---|---|---|
| `sma` | `period` | Simple Moving Average |
| `ema` | `period` | Exponential Moving Average |
| `rsi` | `period` (default 14) | Relative Strength Index (0-100) |
| `macd` | `fast`, `slow`, `signal` | Tuple: (macd_line, signal_line, histogram) |
| `bollinger_bands` | `period`, `std_dev` | Tuple: (upper, middle, lower) |

### Example Strategies

Two built-in strategies are included in `strategies/`:

- **SMA Crossover** (`strategies/sma_crossover.py`) - Trend following using golden/death cross signals
- **Mean Reversion** (`strategies/mean_reversion.py`) - RSI-based mean reversion (buy oversold, sell overbought)

## Performance Metrics

After running a backtest, you get:

- **Total Return** - Overall percentage gain/loss
- **Sharpe Ratio** - Risk-adjusted return (annualized)
- **Max Drawdown** - Largest peak-to-trough decline
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / gross loss
- **Volatility** - Annualized standard deviation of returns

## Running Tests

```bash
pytest
```

## Project Structure

```
backtest-engine/
  app/main.py              # Streamlit web UI
  backtester/
    engine.py              # Core backtest loop
    strategy.py            # Strategy base class, Context, Portfolio
    indicators.py          # Technical indicators (SMA, EMA, RSI, MACD, Bollinger)
    analytics.py           # Performance metrics computation
    data.py                # Data fetching (yfinance)
  strategies/              # Example strategy implementations
  tests/                   # Unit tests
```
