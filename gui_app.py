"""
Streamlit GUI for the AI Gold Research System.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

from main import setup_logging
from src.app_service import ResearchAppService


def bootstrap() -> ResearchAppService:
    if "logging_initialized" not in st.session_state:
        setup_logging()
        st.session_state.logging_initialized = True

    if "service" not in st.session_state:
        st.session_state.service = ResearchAppService()
        st.session_state.last_import_result = None
        st.session_state.last_training_result = None
        st.session_state.last_backtest_result = None
        st.session_state.last_report_result = None

    return st.session_state.service


def show_response(result: dict) -> None:
    if result["success"]:
        st.success(result["message"])
    else:
        st.error(result["message"])

    errors = result.get("errors") or []
    for error in errors:
        st.warning(error)


def render_dashboard(service: ResearchAppService) -> None:
    st.title("AI Gold Research Dashboard")
    status = service.get_system_status()["data"]
    config = service.get_configuration_summary()["data"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Saved Models", status.get("saved_model_files", 0))
    col2.metric("Broker Profiles", status.get("saved_broker_profiles", 0))
    col3.metric("Selected Features", status.get("selected_features", 0))

    st.subheader("Current Status")
    dashboard_rows = [
        {"Field": "Trading Symbol", "Value": config.get("trading_symbol")},
        {"Field": "Timeframe", "Value": config.get("timeframe")},
        {"Field": "Loaded Model", "Value": status.get("loaded_model_file") or "None"},
        {"Field": "Active Broker", "Value": status.get("active_broker") or "None"},
        {"Field": "Dataset Imported", "Value": "Yes" if status.get("last_import_summary") else "No"},
    ]
    st.table(pd.DataFrame(dashboard_rows))

    last_import = status.get("last_import_summary") or {}
    if last_import:
        st.subheader("Last Import Summary")
        st.json(last_import)

    latest_artifacts = status.get("latest_backtest_artifacts") or {}
    if latest_artifacts:
        st.subheader("Latest Backtest Artifacts")
        st.json(latest_artifacts)


def render_data_import(service: ResearchAppService) -> None:
    st.title("Data Import")
    config = service.get_configuration_summary()["data"]
    st.caption(f"Dataset directory: {config.get('dataset_directory')}")

    if st.button("Import Default Dataset", use_container_width=True):
        with st.spinner("Importing dataset and preparing features..."):
            st.session_state.last_import_result = service.import_and_prepare_data()

    result = st.session_state.get("last_import_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            cols = st.columns(3)
            cols[0].metric("Rows", summary.get("rows", 0))
            cols[1].metric("Feature Rows", summary.get("feature_rows", 0))
            cols[2].metric("Selected Features", summary.get("selected_features", 0))

        selected_features = data.get("selected_features") or []
        if selected_features:
            st.subheader("Selected Features")
            st.write(", ".join(selected_features))

        preview = data.get("preview") or []
        if preview:
            st.subheader("Preview")
            st.dataframe(pd.DataFrame(preview), use_container_width=True)


def render_model_training(service: ResearchAppService) -> None:
    st.title("Model Training")
    model_name = st.text_input("Saved model name", value="default")

    if st.button("Train Models", use_container_width=True):
        with st.spinner("Training models..."):
            st.session_state.last_training_result = service.train_models(model_name)

    result = st.session_state.get("last_training_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        training_results = data.get("training_results") or {}
        if training_results:
            training_df = pd.DataFrame(training_results).T.reset_index().rename(columns={"index": "model_name"})
            st.dataframe(training_df, use_container_width=True)

        artifacts = result.get("artifacts") or {}
        if artifacts:
            st.subheader("Artifacts")
            st.json(artifacts)


def render_backtesting(service: ResearchAppService) -> None:
    st.title("Backtesting")

    if st.button("Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            st.session_state.last_backtest_result = service.run_backtest()

    result = st.session_state.get("last_backtest_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            cols = st.columns(4)
            cols[0].metric("Trades", summary.get("total_trades", 0))
            cols[1].metric("Win Rate", f"{summary.get('win_rate', 0):.1%}")
            cols[2].metric("PnL", f"${summary.get('total_pnl', 0):.2f}")
            cols[3].metric("Sharpe", f"{summary.get('sharpe_ratio', 0):.2f}")
            st.json(summary)

        artifacts = result.get("artifacts") or {}
        if artifacts:
            st.subheader("Artifacts")
            st.json(artifacts)

            chart_file = artifacts.get("chart_file")
            if chart_file and Path(chart_file).exists():
                st.image(str(Path(chart_file)), caption=Path(chart_file).name, use_container_width=True)


def render_saved_models(service: ResearchAppService) -> None:
    st.title("Saved Models")
    result = service.list_saved_models()
    models = result.get("data") or []

    if not models:
        st.info("No saved model files found.")
        return

    model_df = pd.DataFrame(models)
    st.dataframe(model_df, use_container_width=True)

    options = {f"{model['name']} ({model['modified_at']})": model["path"] for model in models}
    selected_label = st.selectbox("Select a model to load", list(options.keys()))

    if st.button("Load Selected Model", use_container_width=True):
        with st.spinner("Loading model..."):
            load_result = service.load_saved_model(options[selected_label])
            show_response(load_result)


def render_reports(service: ResearchAppService) -> None:
    st.title("Reports")
    reports_result = service.list_backtest_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No backtest report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a report", list(report_options.keys()))

    if st.button("Load Report", use_container_width=True):
        with st.spinner("Loading report..."):
            st.session_state.last_report_result = service.get_backtest_report(report_options[selected_report])

    result = st.session_state.get("last_report_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            st.subheader("Backtest Summary")
            st.json(summary)


def render_broker_profiles(service: ResearchAppService) -> None:
    st.title("Broker Profiles")

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
            result = service.save_broker_profile(
                name=name,
                login=login,
                password=password,
                server=server,
                terminal_path=terminal_path,
                overwrite=overwrite,
            )
            show_response(result)

    profiles_result = service.list_broker_profiles()
    profiles = profiles_result.get("data") or []
    if profiles:
        st.subheader("Saved Profiles")
        st.dataframe(pd.DataFrame(profiles), use_container_width=True)

        profile_names = [profile["name"] for profile in profiles]
        selected_profile = st.selectbox("Select profile", profile_names)
        col1, col2, col3 = st.columns(3)

        if col1.button("Connect", use_container_width=True):
            with st.spinner("Connecting broker..."):
                show_response(service.connect_broker(selected_profile))
        if col2.button("Disconnect All", use_container_width=True):
            with st.spinner("Disconnecting brokers..."):
                show_response(service.disconnect_all_brokers())
        if col3.button("Delete", use_container_width=True):
            with st.spinner("Deleting profile..."):
                show_response(service.delete_broker_profile(selected_profile))
    else:
        st.info("No saved broker profiles found.")


def main() -> None:
    st.set_page_config(
        page_title="AI Gold Research System",
        page_icon=":bar_chart:",
        layout="wide",
    )

    service = bootstrap()

    pages = {
        "Dashboard": render_dashboard,
        "Data Import": render_data_import,
        "Model Training": render_model_training,
        "Backtesting": render_backtesting,
        "Saved Models": render_saved_models,
        "Reports": render_reports,
        "Broker Profiles": render_broker_profiles,
    }

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.caption("Launch with `streamlit run gui_app.py`")

    try:
        pages[selected_page](service)
    except Exception as exc:
        logger.exception("GUI page failed")
        st.error(f"Page failed: {exc}")


if __name__ == "__main__":
    main()
