"""Reports page."""

from __future__ import annotations

import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from gui.components.summaries import render_backtest_summary
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Reports",
        "Open historical backtest reports and inspect their summaries.",
    )
    render_section_divider()
    reports_result = service.list_backtest_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No backtest report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a report", list(report_options.keys()))

    if st.button("Load Report", use_container_width=True):
        with st.spinner("Loading report..."):
            st.session_state.last_report_result = service.get_backtest_report(report_options[selected_report])

    result = st.session_state.get("last_report_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            render_subtle_divider()
            st.subheader("Backtest Summary")
            render_backtest_summary(summary)
