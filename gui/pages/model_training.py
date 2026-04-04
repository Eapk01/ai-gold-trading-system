"""Model training page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from gui.components.summaries import render_artifact_summary
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Model Training",
        "Train the configured ensemble and save a named model artifact.",
    )
    render_section_divider()
    model_name = st.text_input("Saved model name", value="default")

    if st.button("Train Models", use_container_width=True):
        with st.spinner("Training models..."):
            st.session_state.last_training_result = service.train_models(model_name)

    result = st.session_state.get("last_training_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        training_results = data.get("training_results") or {}
        if training_results:
            render_subtle_divider()
            training_df = pd.DataFrame.from_dict(training_results, orient="index").reset_index()
            training_df = training_df.rename(columns={"index": "trained_model"})
            if "model_name" in training_df.columns:
                training_df = training_df.rename(columns={"model_name": "reported_model"})
            training_df = training_df.loc[:, ~training_df.columns.duplicated()]
            st.dataframe(training_df, use_container_width=True)

        artifacts = result.get("artifacts") or {}
        if artifacts:
            render_subtle_divider()
            st.subheader("Artifacts")
            render_artifact_summary(
                artifacts,
                saved_model_name=data.get("saved_model_name"),
            )
