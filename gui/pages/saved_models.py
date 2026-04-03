"""Saved models page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Saved Models",
        "Browse saved model artifacts and load one into the current session.",
    )
    render_section_divider()
    result = service.list_saved_models()
    models = result.get("data") or []

    if not models:
        st.info("No saved model files found.")
        return

    model_df = pd.DataFrame(models)
    st.dataframe(model_df, use_container_width=True)

    render_subtle_divider()
    options = {f"{model['name']} ({model['modified_at']})": model["path"] for model in models}
    selected_label = st.selectbox("Select a model to load", list(options.keys()))

    if st.button("Load Selected Model", use_container_width=True):
        with st.spinner("Loading model..."):
            load_result = service.load_saved_model(options[selected_label])
            show_response(load_result)
