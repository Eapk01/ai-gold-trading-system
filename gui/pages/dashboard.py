"""Dashboard page."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider, render_vertical_divider
from gui.components.summaries import render_artifact_summary
from src.app_service import ResearchAppService


AUTO_REFRESH_OPTIONS = [5, 10, 30]


def render(service: ResearchAppService) -> None:
    _initialize_dashboard_state()

    snapshot = service.get_dashboard_snapshot().get("data") or {}
    broker = snapshot.get("broker") or {}
    live_trading = snapshot.get("live_trading") or {}
    positions = snapshot.get("positions") or {}
    research = snapshot.get("research") or {}

    _render_header(live_trading)
    _render_top_kpis(broker, live_trading, positions)
    render_section_divider()

    left_col, divider_col, right_col = st.columns([1.35, 0.05, 1], gap="medium")
    with left_col:
        _render_live_trading_status(broker, live_trading)
        render_subtle_divider()
        _render_account_and_positions(broker, positions)
    with divider_col:
        render_vertical_divider()
    with right_col:
        _render_research_snapshot(research)

    if st.session_state.get("dashboard_auto_refresh"):
        interval = int(st.session_state.get("dashboard_refresh_interval", 10))
        st.caption(f"Auto-refresh is on. Refreshing every {interval} seconds.")
        time.sleep(interval)
        st.rerun()


def _initialize_dashboard_state() -> None:
    if "dashboard_auto_refresh" not in st.session_state:
        st.session_state.dashboard_auto_refresh = False
    if "dashboard_refresh_interval" not in st.session_state:
        st.session_state.dashboard_refresh_interval = 10


def _render_header(live_trading: dict[str, Any]) -> None:
    title_col, toolbar_col = st.columns([3.2, 2], vertical_alignment="bottom")

    with title_col:
        st.title("AI Gold Research Dashboard")
        st.caption("Monitor account health, live trading state, positions, and research readiness in one place.")
        badge_col, detail_col = st.columns([1, 4], vertical_alignment="center")
        with badge_col:
            st.markdown(_build_state_badge(live_trading), unsafe_allow_html=True)
        with detail_col:
            st.caption(
                f"Last candle: `{live_trading.get('last_processed_candle') or 'None'}`"
                f" | Age: `{_format_age(live_trading.get('last_candle_age_seconds'))}`"
            )

    with toolbar_col:
        st.caption(" ")
        spacer_col, refresh_col, settings_col = st.columns([3.6, 0.8, 1.2], vertical_alignment="bottom")
        with refresh_col:
            st.caption(" ")
            if st.button("↻", key="dashboard_refresh_now", use_container_width=True, help="Refresh dashboard now"):
                st.rerun()
        with settings_col:
            auto_refresh_enabled = bool(st.session_state.dashboard_auto_refresh)
            settings_label = "Auto On" if auto_refresh_enabled else "Auto Off"
            with st.popover(settings_label, use_container_width=True):
                st.session_state.dashboard_auto_refresh = st.toggle(
                    "Enable auto-refresh",
                    value=auto_refresh_enabled,
                    key="dashboard_auto_refresh_toggle",
                )
                selected_interval = st.selectbox(
                    "Refresh every",
                    AUTO_REFRESH_OPTIONS,
                    index=AUTO_REFRESH_OPTIONS.index(int(st.session_state.dashboard_refresh_interval)),
                    key="dashboard_refresh_interval_select",
                    disabled=not bool(st.session_state.dashboard_auto_refresh),
                )
                st.session_state.dashboard_refresh_interval = int(selected_interval)

    st.markdown("")


def _build_state_badge(live_trading: dict[str, Any]) -> str:
    if not live_trading.get("running"):
        label = "Stopped"
        background = "#e5e7eb"
        color = "#374151"
    elif str(live_trading.get("market_state") or "") == "market_closed_or_stale":
        label = "Stale"
        background = "#fef3c7"
        color = "#92400e"
    else:
        label = "Live"
        background = "#dcfce7"
        color = "#166534"

    return (
        f"<div style='display:inline-block;padding:0.35rem 0.7rem;border-radius:999px;"
        f"background:{background};color:{color};font-size:0.85rem;font-weight:600;'>{label}</div>"
    )



def _render_top_kpis(broker: dict[str, Any], live_trading: dict[str, Any], positions: dict[str, Any]) -> None:
    st.subheader("Top KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Balance", _format_currency(broker.get("balance")))
    col2.metric("Current Equity", _format_currency(broker.get("equity")))
    col3.metric("Open PnL", _format_currency(positions.get("open_positions_profit_total")))
    col4.metric("Auto Trader State", _format_auto_trader_state(live_trading))


def _render_live_trading_status(broker: dict[str, Any], live_trading: dict[str, Any]) -> None:
    st.subheader("Live Trading Status")
    rows = [
        ("Active Broker", broker.get("active_broker") or "None"),
        ("Broker Connected", "Yes" if broker.get("broker_connected") else "No"),
        ("Symbol", live_trading.get("symbol") or "None"),
        ("Timeframe", live_trading.get("timeframe") or "None"),
        ("Auto Trader Running", "Yes" if live_trading.get("running") else "No"),
        ("Market State", _humanize_market_state(live_trading.get("market_state"))),
        ("Managed Positions", positions_or_zero(live_trading.get("managed_positions"))),
        ("Last Processed Candle", live_trading.get("last_processed_candle") or "None"),
        ("Last Candle Age", _format_age(live_trading.get("last_candle_age_seconds"))),
        ("Latest Action", live_trading.get("latest_action") or "None"),
    ]
    st.table(_rows_to_table(rows))

    if str(live_trading.get("market_state") or "") == "market_closed_or_stale":
        st.info(
            f"Market appears closed or stale. Last closed candle age: "
            f"{_format_age(live_trading.get('last_candle_age_seconds'))}."
        )


def _render_account_and_positions(broker: dict[str, Any], positions: dict[str, Any]) -> None:
    st.subheader("Account and Positions")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Free Margin", _format_currency(broker.get("margin_free")))
    col2.metric("Leverage", _format_leverage(broker.get("leverage")))
    col3.metric("Open Positions", f"{positions_or_zero(positions.get('open_positions_count'))}")
    col4.metric(
        "Buy / Sell",
        f"{positions_or_zero(positions.get('open_buy_positions'))} / {positions_or_zero(positions.get('open_sell_positions'))}",
    )

    position_items = positions.get("items") or []
    if not position_items:
        st.info("No open broker positions right now.")
        return

    positions_df = pd.DataFrame(position_items)
    positions_df["Volume"] = positions_df["Volume"].apply(_format_number)
    positions_df["Current Profit"] = positions_df["Current Profit"].apply(_format_currency)
    st.dataframe(positions_df, use_container_width=True, hide_index=True)


def _render_research_snapshot(research: dict[str, Any]) -> None:
    st.subheader("Research Snapshot")
    backtest = research.get("latest_backtest_summary") or {}
    last_import = research.get("last_import_summary") or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Loaded Model", research.get("loaded_model_file") or "None")
    col2.metric("Saved Models", f"{positions_or_zero(research.get('saved_model_files'))}")
    col3.metric("Selected Features", f"{positions_or_zero(research.get('selected_features'))}")
    col4.metric("Dataset Imported", "Yes" if research.get("dataset_imported") else "No")

    rows = [
        ("Import Rows", last_import.get("rows", 0)),
        ("Feature Rows", last_import.get("feature_rows", 0)),
        ("Backtest Trades", backtest.get("total_trades", 0)),
        ("Win Rate", _format_percent(backtest.get("win_rate"))),
        ("Latest Backtest PnL", _format_currency(backtest.get("total_pnl"))),
        ("Sharpe Ratio", _format_number(backtest.get("sharpe_ratio"))),
    ]
    st.table(_rows_to_table(rows))

    latest_artifacts = research.get("latest_backtest_artifacts") or {}
    if latest_artifacts:
        render_artifact_summary(latest_artifacts)


def _rows_to_table(rows: list[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{"Field": field, "Value": _stringify(value)} for field, value in rows])


def _format_auto_trader_state(live_trading: dict[str, Any]) -> str:
    if not live_trading.get("running"):
        return "Stopped"
    return _humanize_market_state(live_trading.get("market_state")) or "Running"


def _humanize_market_state(value: Any) -> str:
    text = str(value or "idle").replace("_", " ").strip()
    return text.title() if text else "Idle"


def _format_currency(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"${numeric:,.2f}"


def _format_number(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:,.2f}"


def _format_percent(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.1%}"


def _format_leverage(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None or numeric <= 0:
        return "N/A"
    return f"1:{numeric:,.0f}"


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


def positions_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify(value: Any) -> str:
    return "None" if value is None else str(value)
