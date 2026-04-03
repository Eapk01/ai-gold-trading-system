"""Page header helpers."""

from __future__ import annotations

import streamlit as st


def render_page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
