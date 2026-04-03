"""Backtesting page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from gui.components.summaries import render_artifact_summary, render_backtest_summary
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Backtesting",
        "Run a backtest on the prepared feature set and review the result.",
    )

    if st.button("Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            st.session_state.last_backtest_result = service.run_backtest()

    result = st.session_state.get("last_backtest_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            st.subheader("Summary")
            render_backtest_summary(summary)

        artifacts = result.get("artifacts") or {}
        if artifacts:
            st.subheader("Artifacts")
            render_artifact_summary(artifacts)

            chart_file = artifacts.get("chart_file")
            if chart_file and Path(chart_file).exists():
                st.image(str(Path(chart_file)), caption=Path(chart_file).name, use_container_width=True)
