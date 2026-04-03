"""Sidebar navigation and status rendering."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.app_service import ResearchAppService


def inject_sidebar_nav_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .stButton {
            margin-bottom: 0.15rem !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start !important;
            font-size: 0.9rem !important;
            padding: 0.26rem 0.55rem !important;
            border-radius: 0.5rem !important;
            min-height: 2.05rem !important;
        }

        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"] {
            border: 1px solid transparent !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"]:hover {
            border-color: transparent !important;
            background: rgba(49, 51, 63, 0.04) !important;
        }

        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] {
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            background: rgba(255, 255, 255, 0.04) !important;
            color: rgba(255, 255, 255, 0.85) !important;
            transition: all 0.15s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_nav_page(page: str) -> None:
    st.session_state["nav_page"] = page


def render_sidebar_navigation(service: ResearchAppService, pages: dict[str, Any]) -> str:
    inject_sidebar_nav_styles()
    st.sidebar.title("Navigation")

    current_page = st.session_state.get("nav_page", "Dashboard")
    for page_name in pages:
        button_type = "secondary" if page_name == current_page else "tertiary"
        if st.sidebar.button(
            page_name,
            key=f"nav_{page_name.lower()}",
            type=button_type,
            use_container_width=True,
            on_click=set_nav_page,
            args=(page_name,),
        ):
            st.rerun()

    status = service.get_system_status()["data"]
    active_broker = status.get("active_broker")
    if active_broker:
        st.sidebar.caption(f"Broker: Connected to {active_broker}")
    else:
        st.sidebar.caption("Broker: Not connected")
    st.sidebar.caption("Launch with `streamlit run gui_app.py`")
    return st.session_state.get("nav_page", "Dashboard")
