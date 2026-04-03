"""Feedback and response rendering helpers."""

from __future__ import annotations

import streamlit as st


def show_response(result: dict) -> None:
    if result["success"]:
        st.success(result["message"])
    else:
        st.error(result["message"])

    message = str(result.get("message", "")).strip()
    errors = result.get("errors") or []
    seen: set[str] = set()
    for error in errors:
        normalized = str(error).strip()
        if not normalized or normalized == message or normalized in seen:
            continue
        seen.add(normalized)
        st.warning(normalized)
