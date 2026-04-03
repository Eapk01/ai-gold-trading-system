"""Feedback and response rendering helpers."""

from __future__ import annotations

import streamlit as st


def show_response(result: dict) -> None:
    if result["success"]:
        st.success(result["message"])
    else:
        st.error(result["message"])

    errors = result.get("errors") or []
    for error in errors:
        st.warning(error)
