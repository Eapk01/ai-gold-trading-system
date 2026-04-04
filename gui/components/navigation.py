"""Sidebar navigation and status rendering."""

from __future__ import annotations

from typing import Any

import streamlit as st

from gui.theme import get_theme_mode, inject_theme_styles, set_theme_mode
from src.app_service import ResearchAppService


def set_nav_page(page: str) -> None:
    st.session_state["nav_page"] = page


def render_sidebar_navigation(service: ResearchAppService, pages: dict[str, Any]) -> str:
    inject_theme_styles()
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

    current_mode = get_theme_mode()
    theme_enabled = st.sidebar.toggle(
        "Light mode",
        value=current_mode == "light",
        key="theme_mode_toggle",
        help="Switch between the default dark theme and a lighter alternative.",
    )
    selected_mode = "light" if theme_enabled else "dark"
    if selected_mode != current_mode:
        set_theme_mode(selected_mode)
        st.rerun()

    status = service.get_system_status()["data"]
    active_broker = status.get("active_broker")
    if active_broker:
        st.sidebar.caption(f"Broker: Connected to {active_broker}")
    else:
        st.sidebar.caption("Broker: Not connected")
    st.sidebar.caption("Launch with `streamlit run gui_app.py`")
    return st.session_state.get("nav_page", "Dashboard")
