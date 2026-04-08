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
from src.research import get_feature_set_display_name, get_stage5_preset_display_name


def _format_elapsed_seconds(value: float | int | None) -> str:
    if value is None:
        return "0.0s"
    total_seconds = float(value)
    minutes = int(total_seconds // 60)
    seconds = total_seconds - (minutes * 60)
    if minutes:
        return f"{minutes}m {seconds:04.1f}s"
    return f"{seconds:.1f}s"


def _format_threshold_list(values: list[object]) -> str:
    thresholds: list[str] = []
    for value in values:
        try:
            thresholds.append(f"{float(value):.2f}")
        except (TypeError, ValueError):
            thresholds.append(str(value))
    return ", ".join(thresholds) if thresholds else "None"


def _format_split_summary(defaults: dict[str, object]) -> str:
    train_fraction = float(defaults.get("train_fraction") or 0.0)
    validation_fraction = float(defaults.get("validation_fraction") or 0.0)
    test_fraction = float(defaults.get("test_fraction") or 0.0)
    return (
        f"{train_fraction:.0%} train / "
        f"{validation_fraction:.0%} validation / "
        f"{test_fraction:.0%} test"
    )


def _format_worker_policy(worker_policy: dict[str, object]) -> str:
    min_auto_workers = int(worker_policy.get("min_auto_workers") or 0)
    max_worker_cap = int(worker_policy.get("max_worker_cap") or 0)
    return f"Auto ({min_auto_workers} to {max_worker_cap} workers)"


def _search_form_key(name: str) -> str:
    return f"stage5_search_form_{name}"


def _initialize_search_form_state(catalog: dict[str, object]) -> None:
    defaults = dict(catalog.get("defaults") or {})
    if st.session_state.get(_search_form_key("initialized")):
        return

    st.session_state[_search_form_key("initialized")] = True
    st.session_state.setdefault(_search_form_key("trainer_name"), str(defaults.get("trainer_name") or ""))
    st.session_state.setdefault(_search_form_key("target_ids"), list(defaults.get("target_ids") or []))
    st.session_state.setdefault(
        _search_form_key("feature_set_names"),
        list(defaults.get("feature_set_names") or []),
    )
    st.session_state.setdefault(_search_form_key("preset_names"), list(defaults.get("preset_names") or []))
    st.session_state.setdefault(
        _search_form_key("threshold_text"),
        _format_threshold_list(list(defaults.get("threshold_list") or [])),
    )
    st.session_state.setdefault(_search_form_key("selector_name"), str(defaults.get("selector_name") or ""))
    st.session_state.setdefault(
        _search_form_key("selector_max_features"),
        int(defaults.get("selector_max_features") or 20),
    )
    st.session_state.setdefault(_search_form_key("train_fraction"), float(defaults.get("train_fraction") or 0.0))
    st.session_state.setdefault(
        _search_form_key("validation_fraction"),
        float(defaults.get("validation_fraction") or 0.0),
    )
    st.session_state.setdefault(_search_form_key("test_fraction"), float(defaults.get("test_fraction") or 0.0))
    st.session_state.setdefault(
        _search_form_key("expanding_window"),
        bool(defaults.get("expanding_window")),
    )
    st.session_state.setdefault(
        _search_form_key("use_auto_workers"),
        defaults.get("max_workers") in (None, 0),
    )
    st.session_state.setdefault(
        _search_form_key("max_workers"),
        int(((defaults.get("worker_policy") or {}).get("max_worker_cap")) or 1),
    )


def _parse_threshold_text(raw_value: str) -> tuple[list[float], str | None]:
    parts = [part.strip() for part in str(raw_value or "").split(",") if part.strip()]
    if not parts:
        return [], "Enter at least one threshold."
    try:
        return [float(part) for part in parts], None
    except ValueError:
        return [], "Thresholds must be comma-separated numbers."


def _build_search_form_overrides(thresholds: list[float]) -> dict[str, object]:
    use_auto_workers = bool(st.session_state.get(_search_form_key("use_auto_workers"), True))
    return {
        "trainer_name": str(st.session_state.get(_search_form_key("trainer_name")) or "").strip(),
        "target_ids": list(st.session_state.get(_search_form_key("target_ids")) or []),
        "feature_set_names": list(st.session_state.get(_search_form_key("feature_set_names")) or []),
        "preset_names": list(st.session_state.get(_search_form_key("preset_names")) or []),
        "threshold_list": list(thresholds),
        "selector_name": str(st.session_state.get(_search_form_key("selector_name")) or "").strip(),
        "selector_max_features": int(st.session_state.get(_search_form_key("selector_max_features")) or 0),
        "train_fraction": float(st.session_state.get(_search_form_key("train_fraction")) or 0.0),
        "validation_fraction": float(st.session_state.get(_search_form_key("validation_fraction")) or 0.0),
        "test_fraction": float(st.session_state.get(_search_form_key("test_fraction")) or 0.0),
        "expanding_window": bool(st.session_state.get(_search_form_key("expanding_window"))),
        "max_workers": None if use_auto_workers else int(st.session_state.get(_search_form_key("max_workers")) or 0),
    }


def _validate_search_form_inputs(thresholds: list[float], threshold_error: str | None) -> list[str]:
    errors: list[str] = []
    if threshold_error:
        errors.append(threshold_error)
    if not list(st.session_state.get(_search_form_key("target_ids")) or []):
        errors.append("Select at least one target.")
    if not list(st.session_state.get(_search_form_key("feature_set_names")) or []):
        errors.append("Select at least one feature set.")
    if not list(st.session_state.get(_search_form_key("preset_names")) or []):
        errors.append("Select at least one preset.")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        errors.append("Threshold values must stay between 0 and 1.")

    train_fraction = float(st.session_state.get(_search_form_key("train_fraction")) or 0.0)
    validation_fraction = float(st.session_state.get(_search_form_key("validation_fraction")) or 0.0)
    test_fraction = float(st.session_state.get(_search_form_key("test_fraction")) or 0.0)
    if train_fraction <= 0.0 or validation_fraction <= 0.0 or test_fraction <= 0.0:
        errors.append("Train, validation, and test fractions must all be positive.")
    if (train_fraction + validation_fraction + test_fraction) > 1.0:
        errors.append("Train, validation, and test fractions must sum to 1.0 or less.")

    if int(st.session_state.get(_search_form_key("selector_max_features")) or 0) <= 0:
        errors.append("Max selected features must be positive.")
    if not bool(st.session_state.get(_search_form_key("use_auto_workers"), True)):
        if int(st.session_state.get(_search_form_key("max_workers")) or 0) <= 0:
            errors.append("Manual worker count must be positive.")
    return errors


def _render_search_space_editor(service: ResearchAppService, base_catalog: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    _initialize_search_form_state(base_catalog)
    defaults = dict(base_catalog.get("defaults") or {})

    st.subheader("Search Setup")
    st.caption(
        "Adjust the bounded Stage 5 search space for this run only. These selections do not write back to `config.yaml`."
    )

    trainer_rows = list(base_catalog.get("available_trainers") or [])
    trainer_labels = {str(row.get("id")): str(row.get("display_name") or row.get("id") or "") for row in trainer_rows}
    st.selectbox(
        "Trainer",
        options=list(trainer_labels.keys()),
        index=max(
            list(trainer_labels.keys()).index(str(st.session_state.get(_search_form_key("trainer_name")) or ""))
            if str(st.session_state.get(_search_form_key("trainer_name")) or "") in trainer_labels
            else 0,
            0,
        ),
        format_func=lambda trainer_id: trainer_labels.get(trainer_id, trainer_id),
        key=_search_form_key("trainer_name"),
        help="Trainer family for this run. Preset choices update automatically.",
    )

    current_catalog = (
        service.get_search_catalog(
            {
                "trainer_name": str(st.session_state.get(_search_form_key("trainer_name")) or defaults.get("trainer_name") or ""),
            }
        ).get("data")
        or {}
    )
    available_targets = list(current_catalog.get("available_targets") or base_catalog.get("available_targets") or [])
    available_feature_sets = list(current_catalog.get("available_feature_sets") or base_catalog.get("available_feature_sets") or [])
    available_presets = list(current_catalog.get("available_presets") or [])
    available_selectors = list(current_catalog.get("available_selectors") or base_catalog.get("available_selectors") or [])

    valid_preset_ids = {str(row.get("id") or "") for row in available_presets}
    st.session_state[_search_form_key("preset_names")] = [
        preset_name
        for preset_name in list(st.session_state.get(_search_form_key("preset_names")) or [])
        if preset_name in valid_preset_ids
    ]

    target_labels = {
        str(row.get("id")): str(row.get("display_name") or row.get("id") or "")
        for row in available_targets
    }
    feature_set_labels = {
        str(row.get("id")): (
            f"{row.get('display_name') or row.get('id')} ({int(row.get('feature_count') or 0)} features)"
        )
        for row in available_feature_sets
    }
    preset_labels = {
        str(row.get("id")): str(row.get("display_name") or row.get("id") or "")
        for row in available_presets
    }
    selector_labels = {
        str(row.get("id")): str(row.get("display_name") or row.get("id") or "")
        for row in available_selectors
    }

    target_col, feature_col = st.columns(2)
    with target_col:
        st.multiselect(
            "Targets",
            options=list(target_labels.keys()),
            default=list(st.session_state.get(_search_form_key("target_ids")) or []),
            format_func=lambda target_id: target_labels.get(target_id, target_id),
            key=_search_form_key("target_ids"),
        )
    with feature_col:
        st.multiselect(
            "Feature Sets",
            options=list(feature_set_labels.keys()),
            default=list(st.session_state.get(_search_form_key("feature_set_names")) or []),
            format_func=lambda feature_set_id: feature_set_labels.get(feature_set_id, feature_set_id),
            key=_search_form_key("feature_set_names"),
        )

    st.multiselect(
        "Presets",
        options=list(preset_labels.keys()),
        default=list(st.session_state.get(_search_form_key("preset_names")) or []),
        format_func=lambda preset_id: preset_labels.get(preset_id, preset_id),
        key=_search_form_key("preset_names"),
        help="Preset options are driven by the selected trainer.",
    )

    selected_target_count = len(list(st.session_state.get(_search_form_key("target_ids")) or []))
    selected_feature_set_count = len(list(st.session_state.get(_search_form_key("feature_set_names")) or []))
    selected_preset_count = len(list(st.session_state.get(_search_form_key("preset_names")) or []))
    candidate_count = selected_target_count * selected_feature_set_count * selected_preset_count
    summary_col, detail_col = st.columns([1, 2])
    with summary_col:
        st.metric("Candidate Count", candidate_count)
        st.caption(
            f"{selected_target_count} targets x {selected_feature_set_count} feature sets x {selected_preset_count} presets"
        )
    with detail_col:
        if candidate_count > 50:
            st.warning("This search space is large. Expect longer runtimes and more candidate reports.")
        elif candidate_count > 24:
            st.info("This search space is starting to get wide. Runtime grows with each extra option.")
        else:
            st.info("This remains a bounded search space designed for understandable comparisons.")

    with st.expander("Advanced Search Settings", expanded=False):
        st.text_input(
            "Threshold list",
            key=_search_form_key("threshold_text"),
            help="Comma-separated probability thresholds, for example `0.50, 0.55, 0.60`.",
        )
        selector_columns = st.columns(2)
        with selector_columns[0]:
            st.selectbox(
                "Selector",
                options=list(selector_labels.keys()),
                index=max(
                    list(selector_labels.keys()).index(
                        str(st.session_state.get(_search_form_key("selector_name")) or defaults.get("selector_name") or "")
                    )
                    if str(st.session_state.get(_search_form_key("selector_name")) or "") in selector_labels
                    else 0,
                    0,
                ),
                format_func=lambda selector_id: selector_labels.get(selector_id, selector_id),
                key=_search_form_key("selector_name"),
            )
        with selector_columns[1]:
            st.number_input(
                "Max selected features",
                min_value=1,
                step=1,
                key=_search_form_key("selector_max_features"),
            )

        split_columns = st.columns(3)
        with split_columns[0]:
            st.number_input(
                "Train fraction",
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key=_search_form_key("train_fraction"),
            )
        with split_columns[1]:
            st.number_input(
                "Validation fraction",
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key=_search_form_key("validation_fraction"),
            )
        with split_columns[2]:
            st.number_input(
                "Test fraction",
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key=_search_form_key("test_fraction"),
            )

        st.checkbox(
            "Use expanding window",
            key=_search_form_key("expanding_window"),
            help="When off, the walk-forward splitter uses rolling windows instead.",
        )
        st.checkbox(
            "Use automatic worker count",
            key=_search_form_key("use_auto_workers"),
            help=_format_worker_policy(dict(defaults.get("worker_policy") or {})),
        )
        if not bool(st.session_state.get(_search_form_key("use_auto_workers"), True)):
            st.number_input(
                "Max workers",
                min_value=1,
                step=1,
                key=_search_form_key("max_workers"),
            )

    with st.expander("Where These Values Come From"):
        st.markdown(
            "\n".join(
                [
                    "- Stage 5 defaults come from `config/config.yaml`.",
                    "- Available targets come from `src/research/catalog/stage5_targets.py`.",
                    "- Available feature sets come from `src/research/feature_sets.py`.",
                    "- Available presets come from `src/research/catalog/search_presets.py`.",
                    "- The GUI reads everything through `ResearchAppService.get_search_catalog()`.",
                ]
            )
        )
    thresholds, threshold_error = _parse_threshold_text(
        str(st.session_state.get(_search_form_key("threshold_text")) or "")
    )
    overrides = _build_search_form_overrides(thresholds)
    validation_errors = _validate_search_form_inputs(thresholds, threshold_error)
    return overrides, {
        **current_catalog,
        "candidate_count": candidate_count,
        "validation_errors": validation_errors,
    }


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
    _render_feature_set_glossary({})
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
    catalog = (service.get_search_catalog().get("data") or {})
    search_overrides, live_catalog = _render_search_space_editor(service, catalog)
    _render_feature_set_glossary(live_catalog)
    _render_preset_glossary(live_catalog)
    render_subtle_divider()

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

    validation_errors = list(live_catalog.get("validation_errors") or [])
    if validation_errors:
        st.error("Please fix the search setup before running:")
        for error_message in validation_errors:
            st.caption(f"- {error_message}")

    if service.feature_data is None or service.feature_data.empty:
        st.info("Prepare data first to run search. You can still edit the bounded search space now.")

    if st.button("Run Automated Search", use_container_width=True):
        if validation_errors:
            return
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
                search_overrides=search_overrides,
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


def _render_feature_set_glossary(catalog: dict[str, object]) -> None:
    with st.expander("What The Feature Sets Mean"):
        rows = [
            {
                "Feature Set": row.get("display_name"),
                "Internal Name": row.get("id"),
                "Meaning": row.get("description"),
            }
            for row in list(catalog.get("available_feature_sets") or [])
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_preset_glossary(catalog: dict[str, object]) -> None:
    with st.expander("What The Search Presets Mean"):
        rows = [
            {
                "Preset": row.get("display_name"),
                "Internal Name": row.get("id"),
                "Meaning": row.get("description"),
            }
            for row in list(catalog.get("available_presets") or [])
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
