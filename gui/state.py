"""Session-state bootstrap helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from main import setup_logging
from src.app_service import ResearchAppService


def bootstrap() -> ResearchAppService:
    if "logging_initialized" not in st.session_state:
        setup_logging()
        st.session_state.logging_initialized = True

    if "service" not in st.session_state:
        st.session_state.service = ResearchAppService()
        st.session_state.last_import_result = None
        st.session_state.last_training_result = None
        st.session_state.last_backtest_result = None
        st.session_state.last_report_result = None
        st.session_state.broker_action_result = None

    return st.session_state.service
