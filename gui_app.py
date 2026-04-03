"""
Streamlit GUI for the AI Gold Research System.
"""

from __future__ import annotations

import streamlit as st
from loguru import logger

from gui.components.navigation import render_sidebar_navigation
from gui.pages import auto_trader, backtesting, brokers, dashboard, data_import, manual_trading, model_training, reports, saved_models
from gui.state import bootstrap


def main() -> None:
    st.set_page_config(
        page_title="AI Gold Research System",
        page_icon=":bar_chart:",
        layout="wide",
    )

    service = bootstrap()

    pages = {
        "Dashboard": dashboard.render,
        "Import": data_import.render,
        "Training": model_training.render,
        "Backtest": backtesting.render,
        "Models": saved_models.render,
        "Reports": reports.render,
        "Brokers": brokers.render,
        "Auto Trader": auto_trader.render,
        "Manual Trader": manual_trading.render,
    }

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in pages:
        st.session_state["nav_page"] = "Dashboard"

    selected_page = render_sidebar_navigation(service, pages)

    try:
        pages[selected_page](service)
    except Exception as exc:
        logger.exception("GUI page failed")
        st.error(f"Page failed: {exc}")


if __name__ == "__main__":
    main()
