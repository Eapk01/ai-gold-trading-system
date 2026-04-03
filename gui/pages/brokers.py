"""Broker profiles page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from src.app_service import ResearchAppService


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Broker Profiles",
        "Manage saved Exness profiles and current connection status.",
    )
    render_section_divider()

    with st.form("add_broker_profile"):
        st.subheader("Add or Replace Exness Profile")
        name = st.text_input("Profile Name")
        login = st.text_input("MT5 Login")
        password = st.text_input("MT5 Password", type="password")
        server = st.text_input("MT5 Server")
        terminal_path = st.text_input("MT5 Terminal Path (optional)")
        overwrite = st.checkbox("Overwrite if profile exists")
        submitted = st.form_submit_button("Save Profile", use_container_width=True)

    if submitted:
        with st.spinner("Saving broker profile..."):
            st.session_state.broker_action_result = service.save_broker_profile(
                name=name,
                login=login,
                password=password,
                server=server,
                terminal_path=terminal_path,
                overwrite=overwrite,
            )
        st.rerun()

    pending_broker_result = st.session_state.get("broker_action_result")
    if pending_broker_result:
        show_response(pending_broker_result)
        st.session_state.broker_action_result = None

    profiles_result = service.list_broker_profiles()
    profiles = profiles_result.get("data") or []
    if profiles:
        render_subtle_divider()
        st.subheader("Saved Profiles")
        profiles_df = pd.DataFrame(profiles)
        if "connected" in profiles_df.columns:
            profiles_df["connected"] = profiles_df["connected"].astype(bool)
        if "is_active" in profiles_df.columns:
            profiles_df["is_active"] = profiles_df["is_active"].astype(bool)
        profiles_df = profiles_df.rename(
            columns={
                "name": "Name",
                "type": "Type",
                "connected": "Connected",
                "sandbox": "Sandbox",
                "last_heartbeat": "Last Heartbeat",
                "is_active": "Active",
            }
        )
        st.dataframe(profiles_df, use_container_width=True)

        profile_names = [profile["name"] for profile in profiles]
        selected_profile = st.selectbox("Select profile", profile_names, key="broker_selected_profile")
        col1, col2, col3 = st.columns(3)

        if col1.button("Connect", key="broker_connect", use_container_width=True):
            with st.spinner("Connecting broker..."):
                st.session_state.broker_action_result = service.connect_broker(selected_profile)
            st.rerun()
        if col2.button("Disconnect All", key="broker_disconnect_all", use_container_width=True):
            with st.spinner("Disconnecting brokers..."):
                st.session_state.broker_action_result = service.disconnect_all_brokers()
            st.rerun()
        if col3.button("Delete", key="broker_delete", use_container_width=True):
            with st.spinner("Deleting profile..."):
                st.session_state.broker_action_result = service.delete_broker_profile(selected_profile)
            st.rerun()
    else:
        st.info("No saved broker profiles found.")
