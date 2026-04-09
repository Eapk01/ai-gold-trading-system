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
    settings_catalog = service.get_auto_trader_settings_catalog().get("data") or {}

    page_result = st.session_state.get("auto_trader_result")
    if page_result:
        show_response(page_result)
        st.session_state.auto_trader_result = None

    settings_tab, monitor_tab = st.tabs(["Settings", "Monitor"])
    with settings_tab:
        _render_settings_tab(service, status, settings_catalog)
    with monitor_tab:
        _render_monitor_tab(service, status, snapshot, events)


def _render_settings_tab(service: ResearchAppService, status: dict[str, Any], catalog: dict[str, Any]) -> None:
    _initialize_settings_form_state(catalog)

    presets = list(catalog.get("built_in_presets") or []) + list(catalog.get("custom_presets") or [])
    preset_options = {preset["id"]: preset for preset in presets}
    selected_default = st.session_state.get("auto_trader_form_selected_preset_id") or catalog.get("selected_preset_id")
    selected_preset_id = st.selectbox(
        "Preset",
        options=list(preset_options.keys()),
        index=_safe_option_index(list(preset_options.keys()), selected_default),
        format_func=lambda preset_id: preset_options[preset_id]["display_name"],
        key="auto_trader_form_selected_preset_id",
    )
    if selected_preset_id and selected_preset_id != st.session_state.get("auto_trader_form_applied_preset_id"):
        _load_form_values_into_session((preset_options.get(selected_preset_id) or {}).get("values") or {})
        st.session_state.auto_trader_form_applied_preset_id = selected_preset_id

    selected_preset = preset_options.get(selected_preset_id)
    if selected_preset:
        st.caption(selected_preset.get("description", ""))

    if bool(status.get("running")):
        st.warning("Auto Trader is running. Applied or saved setting changes will take effect after restart.")

    current_values = _get_form_values()
    differs_from_session = current_values != (catalog.get("session_values") or {})
    differs_from_defaults = current_values != (catalog.get("saved_values") or {})
    dirty_bits = []
    if differs_from_session:
        dirty_bits.append("differs from session settings")
    if differs_from_defaults:
        dirty_bits.append("differs from saved defaults")
    if dirty_bits:
        st.caption("Current form state: " + " | ".join(dirty_bits))
    else:
        st.caption("Current form state matches the active session settings.")

    entry_col, exit_col = st.columns(2)
    with entry_col:
        st.subheader("Entry / Risk")
        st.number_input("Stop Loss", min_value=0.01, step=0.5, key="auto_trader_form_stop_loss_pips")
        st.number_input("Take Profit", min_value=0.01, step=0.5, key="auto_trader_form_take_profit_pips")
        st.number_input(
            "Signal Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="auto_trader_form_signal_confidence_threshold",
        )
    with exit_col:
        st.subheader("Exit Management")
        st.selectbox(
            "Exit Mode",
            options=["disabled", "trailing_stop"],
            key="auto_trader_form_exit_management_mode",
        )
        st.checkbox("Break-even Enabled", key="auto_trader_form_break_even_enabled")
        st.number_input("Break-even Trigger", min_value=0.0, step=0.5, key="auto_trader_form_break_even_trigger_pips")
        st.number_input("Break-even Offset", min_value=0.0, step=0.1, key="auto_trader_form_break_even_offset_pips")
        st.checkbox("Trailing Enabled", key="auto_trader_form_trailing_enabled")
        st.number_input("Trailing Activation", min_value=0.0, step=0.5, key="auto_trader_form_trailing_activation_pips")
        st.number_input("Trailing Distance", min_value=0.0, step=0.5, key="auto_trader_form_trailing_distance_pips")
        st.number_input("Trailing Step", min_value=0.0, step=0.1, key="auto_trader_form_trailing_step_pips")
        st.checkbox("Keep Take Profit", key="auto_trader_form_keep_take_profit")

    preset_name = st.text_input(
        "Custom Preset Name",
        value=st.session_state.get("auto_trader_form_preset_name", ""),
        key="auto_trader_form_preset_name",
        help="Used when saving the current form as a custom preset.",
    )

    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("Apply For This Session", use_container_width=True):
            validation_error = _validate_form_values(current_values)
            if validation_error:
                st.session_state.auto_trader_result = {"success": False, "message": validation_error}
            else:
                st.session_state.auto_trader_result = service.apply_auto_trader_settings(current_values)
            st.rerun()
    with action_cols[1]:
        if st.button("Save As Defaults", use_container_width=True):
            validation_error = _validate_form_values(current_values)
            if validation_error:
                st.session_state.auto_trader_result = {"success": False, "message": validation_error}
            else:
                st.session_state.auto_trader_result = service.save_auto_trader_settings_as_defaults(current_values)
            st.rerun()
    with action_cols[2]:
        if st.button("Save As Preset", use_container_width=True):
            validation_error = _validate_form_values(current_values)
            if validation_error:
                st.session_state.auto_trader_result = {"success": False, "message": validation_error}
            elif not preset_name.strip():
                st.session_state.auto_trader_result = {"success": False, "message": "Preset name is required"}
            else:
                st.session_state.auto_trader_result = service.save_auto_trader_preset(preset_name, current_values)
            st.rerun()
    with action_cols[3]:
        delete_disabled = not bool(selected_preset and selected_preset.get("kind") == "custom")
        if st.button("Delete Preset", use_container_width=True, disabled=delete_disabled):
            st.session_state.auto_trader_result = service.delete_auto_trader_preset(selected_preset_id)
            st.session_state.auto_trader_form_selected_preset_id = catalog.get("selected_preset_id") or "current_defaults"
            st.session_state.auto_trader_form_applied_preset_id = None
            st.rerun()


def _render_monitor_tab(
    service: ResearchAppService,
    status: dict[str, Any],
    snapshot: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
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
        st.caption(
            f"Protection manager: `{status.get('exit_management_mode', 'disabled')}` | "
            f"Last protection action: `{status.get('last_protection_action', 'idle')}`"
        )
        if latest_signal:
            st.caption(
                f"Latest signal: `{latest_signal.get('side')}` @ "
                f"`{float(latest_signal.get('price', 0.0)):.2f}` "
                f"with confidence `{float(latest_signal.get('confidence', 0.0)):.3f}`"
            )
        else:
            st.caption("Latest signal: `None`")

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
    _render_positions(snapshot.get("positions") or [], status.get("symbol") or service.config["trading"]["symbol"])

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
        {"Field": "Exit Management", "Value": status.get("exit_management_mode") or "disabled"},
        {"Field": "Last Managed Stop", "Value": status.get("last_managed_stop_loss")},
        {"Field": "Active Poll Interval", "Value": f"{status.get('active_poll_interval_seconds', 0)}s"},
        {"Field": "Inactive Poll Interval", "Value": f"{status.get('inactive_poll_interval_seconds', 0)}s"},
        {"Field": "Loaded Feature Count", "Value": status.get("loaded_feature_count", 0)},
    ]
    rows = [{"Field": row["Field"], "Value": _stringify_value(row["Value"])} for row in rows]
    st.table(pd.DataFrame(rows))


def _initialize_settings_form_state(catalog: dict[str, Any]) -> None:
    defaults = dict(catalog.get("session_values") or {})
    for field_name, value in defaults.items():
        session_key = f"auto_trader_form_{field_name}"
        if session_key not in st.session_state:
            st.session_state[session_key] = value
    if "auto_trader_form_selected_preset_id" not in st.session_state:
        st.session_state.auto_trader_form_selected_preset_id = catalog.get("selected_preset_id") or "current_defaults"
    if "auto_trader_form_applied_preset_id" not in st.session_state:
        st.session_state.auto_trader_form_applied_preset_id = catalog.get("selected_preset_id")
    if "auto_trader_form_preset_name" not in st.session_state:
        st.session_state.auto_trader_form_preset_name = ""


def _load_form_values_into_session(values: dict[str, Any]) -> None:
    for field_name, value in values.items():
        st.session_state[f"auto_trader_form_{field_name}"] = value


def _get_form_values() -> dict[str, Any]:
    field_names = [
        "stop_loss_pips",
        "take_profit_pips",
        "signal_confidence_threshold",
        "exit_management_mode",
        "break_even_enabled",
        "break_even_trigger_pips",
        "break_even_offset_pips",
        "trailing_enabled",
        "trailing_activation_pips",
        "trailing_distance_pips",
        "trailing_step_pips",
        "keep_take_profit",
    ]
    return {field_name: st.session_state.get(f"auto_trader_form_{field_name}") for field_name in field_names}


def _validate_form_values(values: dict[str, Any]) -> str | None:
    try:
        stop_loss = float(values.get("stop_loss_pips", 0.0))
        take_profit = float(values.get("take_profit_pips", 0.0))
        confidence = float(values.get("signal_confidence_threshold", 0.0))
        break_even_trigger = float(values.get("break_even_trigger_pips", 0.0))
        break_even_offset = float(values.get("break_even_offset_pips", 0.0))
        trailing_activation = float(values.get("trailing_activation_pips", 0.0))
        trailing_distance = float(values.get("trailing_distance_pips", 0.0))
        trailing_step = float(values.get("trailing_step_pips", 0.0))
    except (TypeError, ValueError):
        return "Settings contain an invalid numeric value"

    if stop_loss <= 0:
        return "Stop loss must be greater than zero"
    if take_profit <= 0:
        return "Take profit must be greater than zero"
    if confidence < 0 or confidence > 1:
        return "Signal confidence threshold must be between 0 and 1"
    if min(break_even_trigger, break_even_offset, trailing_activation, trailing_distance, trailing_step) < 0:
        return "Exit-management values must be zero or greater"
    exit_mode = str(values.get("exit_management_mode", "disabled"))
    if exit_mode not in {"disabled", "trailing_stop"}:
        return "Exit mode must be disabled or trailing_stop"
    return None


def _safe_option_index(options: list[str], selected: str | None) -> int:
    try:
        return options.index(selected) if selected in options else 0
    except ValueError:
        return 0


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
                "Stop Loss": float(position.get("sl", 0.0)),
                "Take Profit": float(position.get("tp", 0.0)),
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
