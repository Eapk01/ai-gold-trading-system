"""Session-state bootstrap helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from main import setup_logging
from src.app_service import ResearchAppService


@st.cache_resource(show_spinner=False)
def _get_shared_service() -> ResearchAppService:
    """Create one shared backend service for the running Streamlit process."""
    return ResearchAppService()


def _service_supports_current_ui(service: ResearchAppService) -> bool:
    """Detect stale cached service instances after code changes."""
    return (
        getattr(service, "api_compatibility_version", None)
        == ResearchAppService.API_COMPATIBILITY_VERSION
    )


def _initialize_session_defaults() -> None:
    defaults = {
        "last_import_result": None,
        "last_training_result": None,
        "last_model_test_result": None,
        "last_backtest_result": None,
        "last_report_result": None,
        "last_model_test_report_result": None,
        "broker_action_result": None,
        "manual_trade_result": None,
        "manual_trade_log": [],
        "auto_trader_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def bootstrap() -> ResearchAppService:
    if "logging_initialized" not in st.session_state:
        setup_logging()
        st.session_state.logging_initialized = True

    _initialize_session_defaults()
    service = _get_shared_service()
    if not _service_supports_current_ui(service):
        _get_shared_service.clear()
        service = _get_shared_service()
    return service
