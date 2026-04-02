"""Performance analytics and metrics."""

import numpy as np
import pandas as pd
from backtester.strategy import Trade


def compute_metrics(
    equity_curve: list[float],
    trades: list[Trade],
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute performance metrics from backtest results.

    Args:
        equity_curve: List of portfolio values at each bar.
        trades: List of Trade objects.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dict of metric name -> value.
    """
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0 else 0

    # Sharpe Ratio (annualized, assuming daily bars)
    if len(returns) > 1 and np.std(returns) > 0:
        excess = returns - risk_free_rate / 252
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Maximum Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(np.min(drawdown))

    # Volatility (annualized)
    volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0

    # Win Rate
    buy_prices = {}
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in trades:
        if trade.action == "BUY":
            buy_prices[trade.date] = trade.price
        elif trade.action == "SELL" and buy_prices:
            # Compare against the most recent buy
            last_buy_price = list(buy_prices.values())[-1]
            pnl = (trade.price - last_buy_price) * trade.shares
            if pnl >= 0:
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Profit Factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown * 100, 2),
        "volatility": round(volatility * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "total_trades": len(trades),
        "winning_trades": wins,
        "losing_trades": losses,
    }
