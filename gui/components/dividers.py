"""Subtle divider helpers for page layout."""

from __future__ import annotations

import streamlit as st

from gui.theme import get_theme_tokens


def render_section_divider() -> None:
    border = get_theme_tokens()["border_strong"]
    st.markdown(
        f"<div style='height:1px;background:{border};margin:0.85rem 0 1.05rem 0;'></div>",
        unsafe_allow_html=True,
    )


def render_subtle_divider() -> None:
    border = get_theme_tokens()["border_subtle"]
    st.markdown(
        f"<div style='height:1px;background:{border};margin:0.75rem 0 0.95rem 0;'></div>",
        unsafe_allow_html=True,
    )


def render_vertical_divider(*, min_height: str = "38rem") -> None:
    border = get_theme_tokens()["border_subtle"]
    st.markdown(
        f"<div style='width:1px;height:100%;min-height:{min_height};background:{border};"
        "margin:0 auto;'></div>",
        unsafe_allow_html=True,
    )
