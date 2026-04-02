"""Streamlit web app for the backtesting platform."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import traceback

from backtester.data import fetch_data, validate_data
from backtester.engine import Backtest
from backtester.strategy import Strategy, Context
from backtester.analytics import compute_metrics

EXAMPLE_STRATEGY = '''class MyStrategy(Strategy):
    """SMA Crossover — edit this code to try your own ideas."""

    def initialize(self, ctx):
        ctx.state["short_period"] = 20
        ctx.state["long_period"] = 50

    def on_bar(self, ctx):
        if ctx.i < ctx.state["long_period"]:
            return

        short_sma = ctx.indicator("sma", period=ctx.state["short_period"])
        long_sma = ctx.indicator("sma", period=ctx.state["long_period"])

        if short_sma > long_sma and ctx.portfolio.shares == 0:
            ctx.buy()
        elif short_sma < long_sma and ctx.portfolio.shares > 0:
            ctx.sell()
'''


def run_user_strategy(code: str, data: pd.DataFrame, cash: float) -> dict:
    """Execute user-provided strategy code safely."""
    # Restricted namespace — only expose what's needed
    namespace = {
        "Strategy": Strategy,
        "Context": Context,
        "pd": pd,
        "np": __import__("numpy"),
    }

    exec(code, namespace)

    # Find the user's strategy class
    strategy_cls = None
    for obj in namespace.values():
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
            strategy_cls = obj
            break

    if strategy_cls is None:
        raise ValueError(
            "No Strategy subclass found in your code. "
            "Make sure to define a class that inherits from Strategy."
        )

    strategy = strategy_cls()
    bt = Backtest(data=data, strategy=strategy, cash=cash)
    return bt.run()


def build_chart(data: pd.DataFrame, trades, equity_curve):
    """Build a two-panel chart: price + signals on top, equity on bottom."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Price & Trade Signals", "Equity Curve"),
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["close"],
            name="Close",
            line=dict(color="#636EFA"),
        ),
        row=1,
        col=1,
    )

    # Buy signals
    buys = [t for t in trades if t.action == "BUY"]
    if buys:
        fig.add_trace(
            go.Scatter(
                x=[t.date for t in buys],
                y=[t.price for t in buys],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="#00CC96"),
            ),
            row=1,
            col=1,
        )

    # Sell signals
    sells = [t for t in trades if t.action == "SELL"]
    if sells:
        fig.add_trace(
            go.Scatter(
                x=[t.date for t in sells],
                y=[t.price for t in sells],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="#EF553B"),
            ),
            row=1,
            col=1,
        )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=equity_curve,
            name="Portfolio Value",
            line=dict(color="#AB63FA"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)

    return fig


def main():
    st.set_page_config(page_title="Backtest Platform", layout="wide")
    st.title("Quantitative Backtesting Platform")

    # -- Sidebar: inputs --
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="AAPL")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=pd.Timestamp("2020-01-01"))
        with col2:
            end_date = st.date_input("End", value=pd.Timestamp("2023-12-31"))
        cash = st.number_input(
            "Starting Cash ($)", value=10_000, step=1000, min_value=1
        )

    # -- Strategy code editor --
    st.subheader("Strategy Code")
    st.caption(
        "Define a class that inherits from `Strategy`. "
        "Implement `initialize(self, ctx)` and `on_bar(self, ctx)`."
    )

    try:
        from streamlit_ace import st_ace

        code = st_ace(
            value=EXAMPLE_STRATEGY,
            language="python",
            theme="monokai",
            min_lines=20,
            key="strategy_editor",
        )
    except ImportError:
        code = st.text_area(
            "Strategy Code",
            value=EXAMPLE_STRATEGY,
            height=400,
            key="strategy_editor_fallback",
        )

    # -- Run button --
    if st.button("Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Fetching data..."):
            try:
                data = fetch_data(
                    ticker,
                    start=str(start_date),
                    end=str(end_date),
                )
            except Exception as e:
                st.error(f"Data fetch failed: {e}")
                return

        warnings = validate_data(data)
        for w in warnings:
            st.warning(w)

        with st.spinner("Running backtest..."):
            try:
                result = run_user_strategy(code, data, cash)
            except Exception as e:
                st.error(f"Strategy error:\n```\n{traceback.format_exc()}\n```")
                return

        # -- Results --
        trades = result["trades"]
        equity_curve = result["equity_curve"]
        metrics = compute_metrics(equity_curve, trades)

        # Metrics row
        st.subheader("Performance Metrics")
        cols = st.columns(5)
        cols[0].metric("Total Return", f"{metrics['total_return']}%")
        cols[1].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
        cols[2].metric("Max Drawdown", f"{metrics['max_drawdown']}%")
        cols[3].metric("Win Rate", f"{metrics['win_rate']}%")
        cols[4].metric("Profit Factor", f"{metrics['profit_factor']}")

        col_a, col_b = st.columns(2)
        col_a.metric("Volatility", f"{metrics['volatility']}%")
        col_b.metric("Total Trades", metrics["total_trades"])

        # Chart
        st.subheader("Charts")
        fig = build_chart(data, trades, equity_curve)
        st.plotly_chart(fig, use_container_width=True)

        # Trade log
        if trades:
            st.subheader("Trade Log")
            trade_data = [
                {
                    "Date": str(t.date.date()),
                    "Action": t.action,
                    "Price": f"${t.price:.2f}",
                    "Shares": f"{t.shares:.4f}",
                    "Value": f"${t.value:.2f}",
                }
                for t in trades
            ]
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True)

        # JSON export
        st.subheader("Export")
        export_data = {
            "ticker": ticker,
            "start": str(start_date),
            "end": str(end_date),
            "starting_cash": cash,
            "metrics": metrics,
            "trades": [
                {
                    "date": str(t.date.date()),
                    "action": t.action,
                    "price": t.price,
                    "shares": t.shares,
                    "value": t.value,
                }
                for t in trades
            ],
        }
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"backtest_{ticker}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
