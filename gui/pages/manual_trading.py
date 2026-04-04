"""Simple manual trading page for playful order entry and tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from gui.theme import style_altair_chart
from src.app_service import ResearchAppService


TIMEFRAME_OPTIONS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_CONTRACT_SIZE = 100.0


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Manual Trader",
        "A simple chart, quick long or short buttons, and a compact position tracker for manual trades.",
    )
    render_section_divider()

    config = service.get_configuration_summary().get("data") or {}
    default_symbol = config.get("trading_symbol", "XAUUSDm")
    default_timeframe = config.get("timeframe", "5m")
    default_quantity = float(service.config["trading"].get("position_size", 0.01))

    controls_col, action_col = st.columns([4, 1])
    with controls_col:
        symbol = st.text_input("Symbol", value=default_symbol, key="toy_trader_symbol")
        timeframe_index = TIMEFRAME_OPTIONS.index(default_timeframe) if default_timeframe in TIMEFRAME_OPTIONS else 1
        timeframe = st.selectbox("Chart Timeframe", TIMEFRAME_OPTIONS, index=timeframe_index, key="toy_trader_timeframe")
    with action_col:
        st.write("")
        st.write("")
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    snapshot = service.get_trading_snapshot(symbol=symbol, timeframe=timeframe, bars=200)
    data = snapshot.get("data") or {}
    quote = data.get("quote") or {}
    account = data.get("account") or {}
    positions = data.get("positions") or []
    chart_data = data.get("chart") or []
    broker_connected = bool(data.get("broker_connected"))
    chart_source = data.get("chart_source", "unavailable")
    chart_reason = data.get("chart_reason", "")

    result = st.session_state.get("manual_trade_result")
    if result:
        show_response(result)
        st.session_state.manual_trade_result = None

    _render_quote_metrics(quote, account, chart_source)
    _render_chart_source_notice(chart_source, chart_reason)
    render_subtle_divider()
    _render_chart(chart_data)

    render_subtle_divider()
    st.subheader("Quick Order Pad")
    if broker_connected:
        st.caption("Connected broker detected. Buttons submit real market orders through the active Exness profile.")
    else:
        st.warning("No active broker connection. You can still browse the chart, but order buttons are disabled.")

    quantity = st.number_input(
        "Lot Size",
        min_value=0.01,
        value=max(default_quantity, 0.01),
        step=0.01,
        format="%.2f",
        key="toy_trader_quantity",
    )
    _render_order_estimate(symbol, quantity, quote, account)

    long_col, short_col = st.columns(2)
    if long_col.button("Open Long", use_container_width=True, disabled=not broker_connected):
        _submit_trade(service, symbol, "buy", quantity)
    if short_col.button("Open Short", use_container_width=True, disabled=not broker_connected):
        _submit_trade(service, symbol, "sell", quantity)

    render_subtle_divider()
    st.subheader("Open Positions")
    if positions:
        positions_df = _positions_to_frame(positions)
        st.dataframe(positions_df, use_container_width=True, hide_index=True)

        close_options = [str(ticket) for ticket in positions_df["Ticket"].tolist()]
        selected_ticket = st.selectbox("Position to close", close_options, key="toy_trader_position_ticket")
        if st.button("Close Selected Position", use_container_width=True, disabled=not broker_connected):
            selected_row = positions_df.loc[positions_df["Ticket"] == selected_ticket].iloc[0]
            close_result = service.close_manual_position(selected_ticket)
            st.session_state.manual_trade_result = close_result
            _append_trade_log(close_result, action="close", symbol=str(selected_row["Symbol"]), quantity=None)
            st.rerun()
    else:
        st.info("No open broker positions right now.")

    render_subtle_divider()
    st.subheader("Action Log")
    trade_log = st.session_state.get("manual_trade_log") or []
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Your manual order actions will show up here during this session.")


def _render_quote_metrics(quote: dict[str, Any], account: dict[str, Any], chart_source: str) -> None:
    bid = _safe_float(quote.get("bid"))
    ask = _safe_float(quote.get("ask"))
    spread = ask - bid if bid is not None and ask is not None else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bid", _format_number(bid))
    col2.metric("Ask", _format_number(ask))
    col3.metric("Spread", _format_number(spread, digits=3))
    col4.metric("Chart Source", chart_source.replace("_", " ").title())

    if account:
        acc1, acc2, acc3 = st.columns(3)
        acc1.metric("Balance", _format_number(_safe_float(account.get("balance"))))
        acc2.metric("Equity", _format_number(_safe_float(account.get("equity"))))
        acc3.metric("Margin Free", _format_number(_safe_float(account.get("margin_free"))))


def _render_chart_source_notice(chart_source: str, chart_reason: str) -> None:
    if chart_source == "broker":
        st.caption(chart_reason)
        return

    if chart_source == "local_dataset":
        st.info(f"Chart is currently using the local dataset. {chart_reason}")
        return

    if chart_reason:
        st.warning(chart_reason)


def _render_chart(chart_data: list[dict[str, Any]]) -> None:
    st.subheader("Price Chart")
    if not chart_data:
        st.info("No chart data available yet. Connect the broker or import a dataset to populate the chart.")
        return

    chart_df = pd.DataFrame(chart_data)
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], unit="s", errors="coerce")
    chart_df = chart_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    close_min = float(chart_df["close"].min())
    close_max = float(chart_df["close"].max())
    price_span = max(close_max - close_min, 0.01)
    padding = price_span * 0.12
    y_domain = [close_min - padding, close_max + padding]

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=False, strokeWidth=2.5, color="#d4a017")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y(
                "close:Q",
                title="Price",
                scale=alt.Scale(domain=y_domain, zero=False, nice=False),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("open:Q", title="Open", format=",.2f"),
                alt.Tooltip("high:Q", title="High", format=",.2f"),
                alt.Tooltip("low:Q", title="Low", format=",.2f"),
                alt.Tooltip("close:Q", title="Close", format=",.2f"),
                alt.Tooltip("volume:Q", title="Volume", format=",.0f"),
            ],
        )
        .properties(height=360)
        .interactive()
    )
    chart = style_altair_chart(chart)
    st.altair_chart(chart, use_container_width=True)
    with st.expander("Show recent candles"):
        preview = chart_df.tail(15).copy()
        preview["timestamp"] = preview["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(preview, use_container_width=True, hide_index=True)


def _render_order_estimate(symbol: str, lots: float, quote: dict[str, Any], account: dict[str, Any]) -> None:
    reference_price = _safe_float(quote.get("ask")) or _safe_float(quote.get("bid")) or _safe_float(quote.get("last"))
    units = lots * DEFAULT_CONTRACT_SIZE
    notional_value = units * reference_price if reference_price is not None else None
    pnl_per_dollar = units
    leverage = _safe_float(account.get("leverage"))
    estimated_margin = (
        notional_value / leverage
        if notional_value is not None and leverage is not None and leverage > 0
        else None
    )
    balance = _safe_float(account.get("balance"))
    free_margin = _safe_float(account.get("margin_free"))
    exposure_pct = (notional_value / free_margin * 100.0) if notional_value is not None and free_margin and free_margin > 0 else None
    balance_usage_pct = (
        estimated_margin / balance * 100.0
        if estimated_margin is not None and balance and balance > 0
        else None
    )

    st.caption("Estimated order size")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Metal Units", f"{units:,.2f} oz")
    col2.metric("Notional Value", _format_currency(notional_value))
    col3.metric("PnL per $1 Move", _format_currency(pnl_per_dollar))
    col4.metric("Exposure vs Free Margin", _format_percent(exposure_pct))

    margin_col1, margin_col2, margin_col3 = st.columns(3)
    margin_col1.metric("Est. Margin Used", _format_currency(estimated_margin))
    margin_col2.metric("Account Leverage", _format_leverage(leverage))
    margin_col3.metric("Est. Balance Usage", _format_percent(balance_usage_pct))

    price_text = _format_currency(reference_price)
    if reference_price is not None and estimated_margin is not None:
        st.caption(
            f"Estimated margin hold is about `{_format_currency(estimated_margin)} at {price_text}`. "
            "That amount is typically reserved from free margin, not permanently deducted from balance unless the trade loses money."
        )
    elif reference_price is not None:
        st.caption(
            f"Estimate based on {symbol.upper()} around {price_text} with 1 lot ~= {DEFAULT_CONTRACT_SIZE:.0f} oz. "
            "Actual margin used depends on your broker leverage and account settings."
        )
    else:
        st.caption(
            f"Using the standard estimate of 1 lot ~= {DEFAULT_CONTRACT_SIZE:.0f} oz for {symbol.upper()}. "
            "Connect the broker and refresh to calculate notional value from the live quote."
        )


def _positions_to_frame(positions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for position in positions:
        timestamp = position.get("time")
        opened_at = ""
        if timestamp:
            opened_at = datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
        position_type = position.get("type", 0)
        try:
            position_type_value = int(position_type)
        except (TypeError, ValueError):
            position_type_value = 0

        rows.append(
            {
                "Ticket": str(position.get("ticket", "")),
                "Symbol": position.get("symbol", ""),
                "Side": "Buy" if position_type_value == 0 else "Sell",
                "Volume": _safe_float(position.get("volume")),
                "Open Price": _safe_float(position.get("price_open")),
                "Current Price": _safe_float(position.get("price_current")),
                "Profit": _safe_float(position.get("profit")),
                "Opened At": opened_at,
            }
        )

    return pd.DataFrame(rows)


def _submit_trade(service: ResearchAppService, symbol: str, side: str, quantity: float) -> None:
    result = service.place_manual_order(symbol=symbol, side=side, quantity=quantity)
    st.session_state.manual_trade_result = result
    _append_trade_log(result, action=side, symbol=symbol, quantity=quantity)
    st.rerun()


def _append_trade_log(result: dict[str, Any], *, action: str, symbol: str, quantity: float | None) -> None:
    log = st.session_state.setdefault("manual_trade_log", [])
    log.insert(
        0,
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Action": action.upper(),
            "Symbol": symbol,
            "Quantity": quantity if quantity is not None else "-",
            "Status": "Success" if result.get("success") else "Failed",
            "Message": result.get("message"),
        },
    )
    st.session_state.manual_trade_log = log[:20]


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}"


def _format_currency(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.1f}%"


def _format_leverage(value: float | None) -> str:
    if value is None or value <= 0:
        return "N/A"
    return f"1:{value:,.0f}"
