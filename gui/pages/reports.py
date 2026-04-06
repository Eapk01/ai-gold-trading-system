"""Reports page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gui.components.dividers import render_section_divider, render_subtle_divider
from gui.components.feedback import show_response
from gui.components.headers import render_page_header
from gui.components.summaries import (
    render_artifact_summary,
    render_backtest_summary,
    render_diagnostics_summary,
    render_integrity_summary,
    render_model_test_summary,
    render_search_summary,
    render_training_experiment_summary,
)
from src.app_service import ResearchAppService
from src.research import (
    get_feature_set_description,
    get_feature_set_display_name,
    get_stage5_preset_description,
    get_stage5_preset_display_name,
)


def _format_elapsed_seconds(value: float | int | None) -> str:
    if value is None:
        return "0.0s"
    total_seconds = float(value)
    minutes = int(total_seconds // 60)
    seconds = total_seconds - (minutes * 60)
    if minutes:
        return f"{minutes}m {seconds:04.1f}s"
    return f"{seconds:.1f}s"


def render(service: ResearchAppService) -> None:
    render_page_header(
        "Reports",
        "Search is the primary research workflow now. Candidate reports remain available because Stage 5 produces them and promotion depends on them.",
    )
    render_section_divider()
    backtest_tab, model_test_tab, search_tab, training_experiment_tab = st.tabs(
        [
            "Backtests",
            "Model Tests",
            "Research Search",
            "Candidate Models",
        ]
    )

    with backtest_tab:
        _render_backtest_reports(service)
    with model_test_tab:
        _render_model_test_reports(service)
    with search_tab:
        _render_search_reports(service)
    with training_experiment_tab:
        _render_training_experiment_reports(service)


def _render_backtest_reports(service: ResearchAppService) -> None:
    reports_result = service.list_backtest_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No backtest report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a backtest report", list(report_options.keys()))

    if st.button("Load Backtest Report", use_container_width=True):
        with st.spinner("Loading backtest report..."):
            st.session_state.last_report_result = service.get_backtest_report(report_options[selected_report])

    result = st.session_state.get("last_report_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            render_subtle_divider()
            st.subheader("Backtest Summary")
            render_backtest_summary(summary)


def _render_model_test_reports(service: ResearchAppService) -> None:
    reports_result = service.list_model_test_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No model test report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a model test report", list(report_options.keys()))

    if st.button("Load Model Test Report", use_container_width=True):
        with st.spinner("Loading model test report..."):
            st.session_state.last_model_test_report_result = service.get_model_test_report(report_options[selected_report])

    result = st.session_state.get("last_model_test_report_result")
    if result:
        show_response(result)
        data = result.get("data") or {}
        summary = data.get("summary") or {}
        if summary:
            render_subtle_divider()
            st.subheader("Model Test Summary")
            render_model_test_summary(summary)

        threshold_rows = data.get("threshold_performance") or []
        if threshold_rows:
            render_subtle_divider()
            st.subheader("Threshold Performance")
            st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)

        bucket_rows = data.get("confidence_buckets") or []
        if bucket_rows:
            render_subtle_divider()
            st.subheader("Confidence Buckets")
            st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)


def _render_training_experiment_reports(service: ResearchAppService) -> None:
    _render_feature_set_glossary()
    reports_result = service.list_training_experiment_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No training experiment report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a training experiment report", list(report_options.keys()))

    if st.button("Load Training Experiment Report", use_container_width=True):
        with st.spinner("Loading training experiment report..."):
            st.session_state.last_training_experiment_report_result = service.get_training_experiment_report(
                report_options[selected_report]
            )

    result = st.session_state.get("last_training_experiment_report_result")
    if not result:
        return

    show_response(result)
    data = result.get("data") or {}
    summary = data.get("summary") or {}
    if summary:
        render_subtle_divider()
        st.subheader("Training Experiment Summary")
        render_training_experiment_summary(summary)

    integrity = data.get("integrity") or {}
    if integrity:
        render_subtle_divider()
        st.subheader("Integrity Proof")
        render_integrity_summary(integrity)
        integrity_rows = integrity.get("fold_rows") or []
        if integrity_rows:
            render_subtle_divider()
            st.caption("Per-Fold Integrity")
            st.dataframe(pd.DataFrame(integrity_rows), use_container_width=True, hide_index=True)

    aggregate_metrics = data.get("aggregate_metrics") or {}
    if aggregate_metrics:
        render_subtle_divider()
        st.subheader("Aggregate Metrics")
        st.json(aggregate_metrics)

    comparison_runs = data.get("comparison_runs") or []
    if comparison_runs:
        render_subtle_divider()
        st.subheader("Comparison Feature Set Runs")
        st.dataframe(pd.DataFrame(comparison_runs), use_container_width=True, hide_index=True)

    candidate_artifact = data.get("candidate_artifact") or {}
    if candidate_artifact:
        render_subtle_divider()
        st.subheader("Candidate Artifact")
        st.json(candidate_artifact)

    diagnostics = data.get("diagnostics") or {}
    if diagnostics:
        render_subtle_divider()
        st.subheader("Diagnostics")
        render_diagnostics_summary(diagnostics)

        fold_rows = ((diagnostics.get("target_balance") or {}).get("fold_rows")) or []
        if fold_rows:
            render_subtle_divider()
            st.caption("Per-Fold Target Balance")
            st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)

        feature_health_rows = diagnostics.get("feature_health_rows") or []
        if feature_health_rows:
            render_subtle_divider()
            st.caption("Feature Health")
            st.dataframe(pd.DataFrame(feature_health_rows), use_container_width=True, hide_index=True)

        prediction_health_rows = diagnostics.get("prediction_health_rows") or []
        if prediction_health_rows:
            render_subtle_divider()
            st.caption("Prediction Health")
            st.dataframe(pd.DataFrame(prediction_health_rows), use_container_width=True, hide_index=True)

    artifacts = data.get("artifact_paths") or {}
    if artifacts:
        render_subtle_divider()
        st.subheader("Artifacts")
        render_artifact_summary(artifacts)


def _render_search_reports(service: ResearchAppService) -> None:
    _render_feature_set_glossary()
    _render_preset_glossary()

    st.subheader("Run Primary Research Search")
    st.caption("Use this as the default workflow. The other research tabs are optional diagnostics for drilling into one candidate, target family, or feature family.")
    search_name = st.text_input(
        "Search run name",
        value="research_search",
        key="search_run_name_input",
        help="Used in the saved report and candidate artifact names.",
    )
    progress_bar = st.progress(0.0, text="Idle")
    progress_status = st.empty()
    progress_details = st.empty()

    def _render_progress_snapshot(snapshot: dict | None) -> None:
        payload = snapshot or {}
        phase = str(payload.get("phase") or "idle").title()
        step_label = str(payload.get("step_label") or "Waiting to start")
        current = int(payload.get("current") or 0)
        total = int(payload.get("total") or 0)
        ratio = float(payload.get("progress_ratio") or 0.0)
        elapsed = _format_elapsed_seconds(payload.get("elapsed_seconds"))
        details = payload.get("details") or {}

        progress_bar.progress(min(max(ratio, 0.0), 1.0), text=f"{phase}: {step_label}")
        progress_status.caption(
            f"Phase: {phase} | Progress: {current}/{total if total else '?'} | Elapsed: {elapsed}"
        )

        detail_parts: list[str] = []
        if details.get("execution_mode"):
            detail_parts.append(f"Mode: {details['execution_mode']}")
        if details.get("resolved_max_workers") is not None:
            detail_parts.append(f"Workers: {details['resolved_max_workers']}")
        if details.get("completed_count") is not None:
            detail_parts.append(f"Completed: {details['completed_count']}")
        if details.get("failed_count") is not None:
            detail_parts.append(f"Failed: {details['failed_count']}")
        if details.get("active_count") is not None:
            detail_parts.append(f"Active: {details['active_count']}")
        if details.get("target_display_name"):
            detail_parts.append(f"Target: {details['target_display_name']}")
        if details.get("feature_set_name"):
            detail_parts.append(f"Feature Set: {get_feature_set_display_name(str(details['feature_set_name']))}")
        if details.get("preset_name"):
            detail_parts.append(f"Preset: {get_stage5_preset_display_name(str(details['preset_name']))}")
        if details.get("target_count") is not None:
            detail_parts.append(f"Targets: {details['target_count']}")
        if details.get("feature_set_count") is not None:
            detail_parts.append(f"Feature Sets: {details['feature_set_count']}")
        if details.get("preset_count") is not None:
            detail_parts.append(f"Presets: {details['preset_count']}")
        if details.get("winner_status"):
            detail_parts.append(f"Winner Status: {details['winner_status']}")
        if details.get("error"):
            detail_parts.append(f"Error: {details['error']}")
        progress_details.caption(" | ".join(detail_parts) if detail_parts else "No extra details yet.")

    _render_progress_snapshot(st.session_state.get("last_search_progress"))

    if st.button("Run Automated Search", use_container_width=True):
        st.session_state.last_search_progress = {
            "phase": "setup",
            "step_label": "Preparing search run",
            "current": 0,
            "total": 0,
            "progress_ratio": 0.0,
            "elapsed_seconds": 0.0,
            "details": {},
        }
        _render_progress_snapshot(st.session_state.last_search_progress)

        def _progress_callback(payload: dict) -> None:
            st.session_state.last_search_progress = dict(payload or {})
            _render_progress_snapshot(st.session_state.last_search_progress)

        with st.spinner("Running primary research search... This may take a while on real data."):
            st.session_state.last_search_run_result = service.run_automated_search(
                search_name.strip() or "research_search",
                progress_callback=_progress_callback,
            )

        final_progress = dict(st.session_state.get("last_search_progress") or {})
        if str(final_progress.get("phase") or "").lower() != "failed":
            final_progress.setdefault("phase", "complete")
            final_progress.setdefault("step_label", "Search finished")
            final_progress["progress_ratio"] = 1.0
        st.session_state.last_search_progress = final_progress
        _render_progress_snapshot(final_progress)

    run_result = st.session_state.get("last_search_run_result")
    if run_result:
        show_response(run_result)
        run_data = run_result.get("data") or {}
        run_summary = run_data.get("summary") or {}
        if run_summary:
            render_subtle_divider()
            st.subheader("Latest Search Run")
            render_search_summary(run_summary)

        run_integrity = run_data.get("integrity") or {}
        if run_integrity:
            render_subtle_divider()
            st.subheader("Latest Search Integrity")
            render_integrity_summary(run_integrity)

        run_artifacts = run_result.get("artifacts") or {}
        if run_artifacts:
            render_subtle_divider()
            st.subheader("Latest Search Artifacts")
            render_artifact_summary(run_artifacts)

    render_subtle_divider()
    st.subheader("Open Saved Search Report")
    reports_result = service.list_search_reports(limit=20)
    reports = reports_result.get("data") or []
    if not reports:
        st.info("No search report files found.")
        return

    report_options = {report["name"]: report["path"] for report in reports}
    selected_report = st.selectbox("Select a search report", list(report_options.keys()))

    if st.button("Load Search Report", use_container_width=True):
        with st.spinner("Loading search report..."):
            st.session_state.last_search_report_result = service.get_search_report(report_options[selected_report])

    result = st.session_state.get("last_search_report_result")
    if not result:
        return

    show_response(result)
    data = result.get("data") or {}
    summary = data.get("summary") or {}
    if summary:
        render_subtle_divider()
        st.subheader("Search Summary")
        render_search_summary(summary)

    integrity = data.get("integrity") or {}
    if integrity:
        render_subtle_divider()
        st.subheader("Integrity Proof")
        render_integrity_summary(integrity)
        integrity_rows = integrity.get("fold_rows") or []
        if integrity_rows:
            render_subtle_divider()
            st.caption("Candidate Integrity")
            st.dataframe(pd.DataFrame(integrity_rows), use_container_width=True, hide_index=True)

    diagnostics = data.get("diagnostics") or {}
    if diagnostics:
        render_subtle_divider()
        st.subheader("Why Candidates Failed")
        render_diagnostics_summary(diagnostics)
        candidate_highlights = diagnostics.get("candidate_highlights") or []
        if candidate_highlights:
            highlight_frame = pd.DataFrame(candidate_highlights)
            preferred_columns = [
                "target_display_name",
                "feature_set_display_name",
                "preset_display_name",
                "execution_status",
                "error_message",
                "elapsed_seconds",
                "proof_status",
                "integrity_contract_ok",
                "integrity_failure_reasons",
                "passed_truth_gate",
                "truth_gate_failures",
                "overall_mean_test_accuracy",
                "majority_baseline_mean_test_accuracy",
                "selected_threshold_test_mean_f1",
                "selected_threshold_test_mean_coverage",
                "one_class_fold_count",
                "undefined_selected_threshold_metric_count",
                "constant_feature_fold_count",
                "near_constant_feature_fold_count",
            ]
            available_columns = [column for column in preferred_columns if column in highlight_frame.columns]
            st.dataframe(highlight_frame.loc[:, available_columns], use_container_width=True, hide_index=True)

    leaderboard_rows = data.get("leaderboard_rows") or []
    if leaderboard_rows:
        render_subtle_divider()
        st.subheader("Leaderboard")
        leaderboard_frame = pd.DataFrame(leaderboard_rows)
        if not leaderboard_frame.empty:
            preferred_columns = [
                "rank",
                "target_display_name",
                "feature_set_display_name",
                "preset_display_name",
                "execution_status",
                "error_message",
                "elapsed_seconds",
                "selected_threshold",
                "validation_beat_rate",
                "validation_f1_std",
                "validation_mean_f1",
                "validation_mean_coverage",
                "test_mean_f1",
                "test_mean_coverage",
                "overall_mean_test_accuracy",
                "majority_baseline_mean_test_accuracy",
                "test_best_baseline_mean_f1",
                "passed_test_guardrail",
                "passed_truth_gate",
                "truth_gate_failures",
                "proof_status",
                "integrity_contract_ok",
                "diagnostics",
                "is_recommended",
                "report_file",
            ]
            available_columns = [column for column in preferred_columns if column in leaderboard_frame.columns]
            st.dataframe(leaderboard_frame.loc[:, available_columns], use_container_width=True, hide_index=True)

    recommended_winner = data.get("recommended_winner") or {}
    if recommended_winner:
        render_subtle_divider()
        st.subheader("Recommended Winner")
        if recommended_winner.get("status") == "no_winner":
            st.warning(recommended_winner.get("reason") or "No winner was recommended.")
            gate_summary = data.get("gate_summary") or recommended_winner.get("gate_summary") or {}
            if gate_summary:
                st.json(gate_summary)
        st.json(recommended_winner)

    artifacts = data.get("artifact_paths") or {}
    if artifacts:
        render_subtle_divider()
        st.subheader("Artifacts")
        render_artifact_summary(artifacts)


def _render_feature_set_glossary() -> None:
    with st.expander("What The Feature Sets Mean"):
        rows = [
            {
                "Feature Set": get_feature_set_display_name(name),
                "Internal Name": name,
                "Meaning": get_feature_set_description(name),
            }
            for name in ["baseline_core", "momentum", "volatility", "context", "lag_statistical", "all_eligible"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_preset_glossary() -> None:
    with st.expander("What The Search Presets Mean"):
        rows = [
            {
                "Preset": get_stage5_preset_display_name(name),
                "Internal Name": name,
                "Meaning": get_stage5_preset_description(name),
            }
            for name in ["conservative", "balanced", "capacity"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
