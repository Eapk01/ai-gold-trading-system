"""Subtle divider helpers for page layout."""

from __future__ import annotations

import streamlit as st


def render_section_divider() -> None:
    st.markdown(
        "<div style='height:1px;background:rgba(100,116,139,0.26);margin:0.85rem 0 1.05rem 0;'></div>",
        unsafe_allow_html=True,
    )


def render_subtle_divider() -> None:
    st.markdown(
        "<div style='height:1px;background:rgba(100,116,139,0.20);margin:0.75rem 0 0.95rem 0;'></div>",
        unsafe_allow_html=True,
    )


def render_vertical_divider(*, min_height: str = "38rem") -> None:
    st.markdown(
        f"<div style='width:1px;height:100%;min-height:{min_height};background:rgba(100,116,139,0.18);"
        "margin:0 auto;'></div>",
        unsafe_allow_html=True,
    )
