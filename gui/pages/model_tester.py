"""Model tester page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from gui.components.summaries import render_artifact_summary, render_model_test_summary
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Model Tester",
        "Evaluate prediction quality on every scorable prepared-feature row.",
    )
    render_section_divider()

    if st.button("Run Model Test", use_container_width=True):
        with st.spinner("Running model test..."):
            st.session_state.last_model_test_result = service.run_model_test()

    result = st.session_state.get("last_model_test_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            render_subtle_divider()
            st.subheader("Summary")
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

        evaluation_preview = data.get("evaluation_preview") or []
        if evaluation_preview:
            render_subtle_divider()
            st.subheader("Evaluation Preview")
            st.dataframe(pd.DataFrame(evaluation_preview), use_container_width=True, hide_index=True)

        artifacts = result.get("artifacts") or {}
        if artifacts:
            render_subtle_divider()
            st.subheader("Artifacts")
            render_artifact_summary(artifacts)
