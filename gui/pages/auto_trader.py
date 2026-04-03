"""Automated Exness demo trading page."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Auto Trader",
        "Start or stop the Exness demo auto-trader, monitor signals, and watch managed positions.",
    )
    render_section_divider()

    status_result = service.get_auto_trader_status()
    status = status_result.get("data") or {}
    snapshot_result = service.get_trading_snapshot()
    snapshot = snapshot_result.get("data") or {}
    events = service.get_auto_trader_events(limit=25).get("data") or []

    page_result = st.session_state.get("auto_trader_result")
    if page_result:
        show_response(page_result)
        st.session_state.auto_trader_result = None

    _render_runtime_summary(service, status)

    controls_col, status_col = st.columns([1, 2])
    with controls_col:
        if st.button("Start Auto Trader", use_container_width=True, disabled=bool(status.get("running"))):
            st.session_state.auto_trader_result = service.start_auto_trader()
            st.rerun()
        if st.button("Stop Auto Trader", use_container_width=True, disabled=not bool(status.get("running"))):
            st.session_state.auto_trader_result = service.stop_auto_trader()
            st.rerun()
    with status_col:
        latest_signal = status.get("latest_signal") or {}
        st.caption(
            f"Latest action: `{status.get('latest_action', 'idle')}` | "
            f"Last processed candle: `{status.get('last_processed_candle') or 'None'}`"
        )
        if latest_signal:
            st.caption(
                f"Latest signal: `{latest_signal.get('side')}` @ "
                f"`{float(latest_signal.get('price', 0.0)):.2f}` "
                f"with confidence `{float(latest_signal.get('confidence', 0.0)):.3f}`"
            )

    market_state = str(status.get("market_state") or "idle")
    if market_state == "market_closed_or_stale":
        st.info(
            f"Market appears closed or stale. Last closed candle age: "
            f"`{_format_age(status.get('last_candle_age_seconds'))}`. "
            f"Polling has backed off to every `{status.get('inactive_poll_interval_seconds', 0)}` seconds."
        )
    elif market_state == "waiting_for_new_candle":
        st.caption(
            f"Waiting for the next closed candle. Current candle age: "
            f"`{_format_age(status.get('last_candle_age_seconds'))}`."
        )

    if status.get("latest_error"):
        st.warning(status["latest_error"])

    render_subtle_divider()
    st.subheader("Managed Positions")
    _render_positions(snapshot.get("positions") or [], service.config["trading"]["symbol"])

    render_subtle_divider()
    st.subheader("Recent Events")
    if events:
        events_df = pd.DataFrame(events)
        st.dataframe(events_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No auto-trader events yet.")

    render_subtle_divider()
    st.subheader("Chart Snapshot")
    _render_chart(snapshot.get("chart") or [])


def _render_runtime_summary(service: ResearchAppService, status: dict[str, Any]) -> None:
    system_status = service.get_system_status().get("data") or {}
    config = service.get_configuration_summary().get("data") or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runtime", "Running" if status.get("running") else "Stopped")
    col2.metric("Broker", system_status.get("active_broker") or "None")
    col3.metric("Loaded Model", system_status.get("loaded_model_file") or "None")
    col4.metric("Managed Positions", status.get("managed_positions", 0))

    rows = [
        {"Field": "Symbol", "Value": status.get("symbol") or config.get("trading_symbol")},
        {"Field": "Timeframe", "Value": status.get("timeframe") or config.get("timeframe")},
        {"Field": "Market State", "Value": status.get("market_state") or "idle"},
        {"Field": "Startup Ready", "Value": "Yes" if status.get("startup_ready") else "No"},
        {"Field": "Last Candle Age", "Value": _format_age(status.get("last_candle_age_seconds"))},
        {"Field": "History Rows", "Value": status.get("history_rows", 0)},
        {"Field": "Confidence Threshold", "Value": status.get("confidence_threshold")},
        {"Field": "Active Poll Interval", "Value": f"{status.get('active_poll_interval_seconds', 0)}s"},
        {"Field": "Inactive Poll Interval", "Value": f"{status.get('inactive_poll_interval_seconds', 0)}s"},
        {"Field": "Loaded Feature Count", "Value": status.get("loaded_feature_count", 0)},
    ]
    rows = [{"Field": row["Field"], "Value": _stringify_value(row["Value"])} for row in rows]
    st.table(pd.DataFrame(rows))


def _render_positions(positions: list[dict[str, Any]], managed_symbol: str) -> None:
    filtered_positions = [position for position in positions if str(position.get("symbol", "")).strip() == managed_symbol]
    if not filtered_positions:
        st.info(f"No open positions found for {managed_symbol}.")
        return

    rows = []
    for position in filtered_positions:
        rows.append(
            {
                "Ticket": str(position.get("ticket", "")),
                "Side": "Buy" if _safe_int(position.get("type", 0)) == 0 else "Sell",
                "Volume": float(position.get("volume", 0.0)),
                "Open Price": float(position.get("price_open", 0.0)),
                "Current Price": float(position.get("price_current", 0.0)),
                "Profit": float(position.get("profit", 0.0)),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_chart(chart_data: list[dict[str, Any]]) -> None:
    if not chart_data:
        st.info("No chart data available yet.")
        return

    chart_df = pd.DataFrame(chart_data)
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], unit="s", errors="coerce")
    chart_df = chart_df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    if chart_df.empty:
        st.info("No chart data available yet.")
        return

    y_min = float(chart_df["close"].min())
    y_max = float(chart_df["close"].max())
    padding = max((y_max - y_min) * 0.12, 0.01)
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=False, strokeWidth=2.5, color="#0f766e")
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("close:Q", title="Price", scale=alt.Scale(domain=[y_min - padding, y_max + padding], zero=False)),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("open:Q", title="Open", format=",.2f"),
                alt.Tooltip("high:Q", title="High", format=",.2f"),
                alt.Tooltip("low:Q", title="Low", format=",.2f"),
                alt.Tooltip("close:Q", title="Close", format=",.2f"),
                alt.Tooltip("volume:Q", title="Volume", format=",.0f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_age(value: Any) -> str:
    try:
        total_seconds = int(value)
    except (TypeError, ValueError):
        return "Unknown"

    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes, seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _stringify_value(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)
