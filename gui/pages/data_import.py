"""Data import page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Data Import",
        "Load the default dataset, validate it, and inspect prepared features.",
    )
    config = service.get_configuration_summary()["data"]
    st.caption(f"Dataset directory: {config.get('dataset_directory')}")
    render_section_divider()

    if st.button("Import Default Dataset", use_container_width=True):
        with st.spinner("Importing dataset and preparing features..."):
            st.session_state.last_import_result = service.import_and_prepare_data()

    result = st.session_state.get("last_import_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            cols = st.columns(3)
            cols[0].metric("Rows", summary.get("rows", 0))
            cols[1].metric("Feature Rows", summary.get("feature_rows", 0))
            cols[2].metric("Selected Features", summary.get("selected_features", 0))

        selected_features = data.get("selected_features") or []
        if selected_features:
            render_subtle_divider()
            st.subheader("Selected Features")
            st.write(", ".join(selected_features))

        preview = data.get("preview") or []
        if preview:
            render_subtle_divider()
            st.subheader("Preview")
            st.dataframe(pd.DataFrame(preview), use_container_width=True)
