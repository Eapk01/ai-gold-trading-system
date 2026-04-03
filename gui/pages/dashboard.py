"""Dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.headers import render_page_header
from gui.components.summaries import render_artifact_summary, render_key_value_summary, render_raw_expander
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "AI Gold Research Dashboard",
        "Monitor models, brokers, imports, and recent backtest outputs.",
    )
    status = service.get_system_status()["data"]
    config = service.get_configuration_summary()["data"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Saved Models", status.get("saved_model_files", 0))
    col2.metric("Broker Profiles", status.get("saved_broker_profiles", 0))
    col3.metric("Selected Features", status.get("selected_features", 0))

    st.subheader("Current Status")
    dashboard_rows = [
        {"Field": "Trading Symbol", "Value": config.get("trading_symbol")},
        {"Field": "Timeframe", "Value": config.get("timeframe")},
        {"Field": "Loaded Model", "Value": status.get("loaded_model_file") or "None"},
        {"Field": "Active Broker", "Value": status.get("active_broker") or "None"},
        {"Field": "Dataset Imported", "Value": "Yes" if status.get("last_import_summary") else "No"},
    ]
    st.table(pd.DataFrame(dashboard_rows))

    last_import = status.get("last_import_summary") or {}
    if last_import:
        st.subheader("Last Import Summary")
        render_key_value_summary(
            [
                ("Dataset Path", last_import.get("path"), "auto"),
                ("Rows", last_import.get("rows"), "integer"),
                ("Feature Rows", last_import.get("feature_rows"), "integer"),
                ("Selected Features", last_import.get("selected_features"), "integer"),
                ("Data Valid", last_import.get("data_valid"), "boolean"),
            ]
        )
        render_raw_expander("Raw details", last_import)

    latest_artifacts = status.get("latest_backtest_artifacts") or {}
    if latest_artifacts:
        st.subheader("Latest Backtest Artifacts")
        render_artifact_summary(latest_artifacts)
