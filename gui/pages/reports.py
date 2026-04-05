"""Reports page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from gui.components.summaries import render_backtest_summary, render_model_test_summary
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Reports",
        "Open historical backtest and model-test reports and inspect their summaries.",
    )
    render_section_divider()
    backtest_tab, model_test_tab = st.tabs(["Backtest Reports", "Model Test Reports"])

    with backtest_tab:
        _render_backtest_reports(service)
    with model_test_tab:
        _render_model_test_reports(service)


def _render_backtest_reports(service: ResearchAppService) -> None:
    reports_result = service.list_backtest_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No backtest report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a backtest report", list(report_options.keys()))

    if st.button("Load Backtest Report", use_container_width=True):
        with st.spinner("Loading backtest report..."):
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


def _render_model_test_reports(service: ResearchAppService) -> None:
    reports_result = service.list_model_test_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No model test report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a model test report", list(report_options.keys()))

    if st.button("Load Model Test Report", use_container_width=True):
        with st.spinner("Loading model test report..."):
            st.session_state.last_model_test_report_result = service.get_model_test_report(report_options[selected_report])

    result = st.session_state.get("last_model_test_report_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            render_subtle_divider()
            st.subheader("Model Test Summary")
            render_model_test_summary(summary)

        threshold_rows = data.get("threshold_performance") or []
        if threshold_rows:
            render_subtle_divider()
            st.subheader("Threshold Performance")
            st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)

        bucket_rows = data.get("confidence_buckets") or []
        if bucket_rows:
            render_subtle_divider()
            st.subheader("Confidence Buckets")
            st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)
