"""Summary and detail rendering helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st


def format_value(value: Any, format_type: str = "auto") -> str:
    if value is None or value == "":
        return "None"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if format_type == "boolean":
        return "Yes" if bool(value) else "No"

    if format_type == "percent":
        return f"{float(value):.1%}"

    if format_type == "currency":
        return f"${float(value):,.2f}"

    if format_type == "integer":
        return f"{int(value):,}"

    if format_type == "float":
        return f"{float(value):,.2f}"

    return str(value)


def render_key_value_summary(items: list[tuple[str, Any, str]]) -> None:
    for label, value, format_type in items:
        label_col, value_col = st.columns([1, 2])
        with label_col:
            st.caption(label)
        with value_col:
            st.write(format_value(value, format_type))


def render_raw_expander(label: str, payload: Any) -> None:
    with st.expander(label):
        st.json(payload)


def render_backtest_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(4)
    cols[0].metric("Trades", summary.get("total_trades", 0))
    cols[1].metric("Win Rate", format_value(summary.get("win_rate", 0), "percent"))
    cols[2].metric("PnL", format_value(summary.get("total_pnl", 0), "currency"))
    cols[3].metric("Sharpe", format_value(summary.get("sharpe_ratio", 0), "float"))

    detail_items = [
        ("Winning Trades", summary.get("winning_trades"), "integer"),
        ("Losing Trades", summary.get("losing_trades"), "integer"),
        ("Max Drawdown", summary.get("max_drawdown"), "percent"),
        ("Sortino Ratio", summary.get("sortino_ratio"), "float"),
        ("Calmar Ratio", summary.get("calmar_ratio"), "float"),
        ("Profit Factor", summary.get("profit_factor"), "float"),
        ("Avg Winning Trade", summary.get("avg_winning_trade"), "currency"),
        ("Avg Losing Trade", summary.get("avg_losing_trade"), "currency"),
        ("Max Consecutive Wins", summary.get("max_consecutive_wins"), "integer"),
        ("Max Consecutive Losses", summary.get("max_consecutive_losses"), "integer"),
        ("Start Date", summary.get("start_date"), "auto"),
        ("End Date", summary.get("end_date"), "auto"),
    ]

    if "equity_curve_points" in summary:
        detail_items.append(("Equity Curve Points", summary.get("equity_curve_points"), "integer"))

    render_key_value_summary(detail_items)

    baseline_summary = summary.get("baseline_summary") or {}
    if baseline_summary:
        st.caption("Baseline Comparison")
        baseline_items = [
            ("Baseline Trades", baseline_summary.get("total_trades"), "integer"),
            ("Baseline Win Rate", baseline_summary.get("win_rate"), "percent"),
            ("Baseline PnL", baseline_summary.get("total_pnl"), "currency"),
            ("Baseline Sharpe", baseline_summary.get("sharpe_ratio"), "float"),
        ]
        render_key_value_summary(baseline_items)

    comparison_summary = summary.get("comparison_summary") or {}
    if comparison_summary:
        comparison_items = [
            ("Baseline Name", comparison_summary.get("baseline_name"), "auto"),
            ("Curve Rows", comparison_summary.get("equity_curve_rows"), "integer"),
            ("PnL vs Baseline", comparison_summary.get("model_minus_baseline_pnl"), "currency"),
            ("Win Rate vs Baseline", comparison_summary.get("model_minus_baseline_win_rate"), "percent"),
            ("Sharpe vs Baseline", comparison_summary.get("model_minus_baseline_sharpe"), "float"),
        ]
        render_key_value_summary(comparison_items)

    render_raw_expander("Raw details", summary)


def render_model_test_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(4)
    cols[0].metric("Scored Rows", format_value(summary.get("scored_rows", 0), "integer"))
    cols[1].metric("Coverage", format_value(summary.get("coverage_rate", 0), "percent"))
    cols[2].metric("Accuracy", format_value(summary.get("accuracy", 0), "percent"))
    cols[3].metric("F1", format_value(summary.get("f1", 0), "float"))

    confusion = summary.get("confusion_matrix") or {}
    detail_items = [
        ("Total Rows", summary.get("total_rows"), "integer"),
        ("Valid Prediction Rows", summary.get("valid_prediction_rows"), "integer"),
        ("Invalid Rows", summary.get("invalid_rows"), "integer"),
        ("Target Missing Rows", summary.get("target_missing_rows"), "integer"),
        ("Precision", summary.get("precision"), "float"),
        ("Recall", summary.get("recall"), "float"),
        ("Positive Precision", summary.get("positive_precision"), "float"),
        ("Positive Recall", summary.get("positive_recall"), "float"),
        ("Negative Precision", summary.get("negative_precision"), "float"),
        ("Negative Recall", summary.get("negative_recall"), "float"),
        ("True Negatives", confusion.get("tn"), "integer"),
        ("False Positives", confusion.get("fp"), "integer"),
        ("False Negatives", confusion.get("fn"), "integer"),
        ("True Positives", confusion.get("tp"), "integer"),
    ]

    render_key_value_summary(detail_items)
    render_raw_expander("Raw details", summary)


def render_experiment_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(4)
    cols[0].metric("Experiment", format_value(summary.get("experiment_name"), "auto"))
    cols[1].metric("Target", format_value(summary.get("target_column"), "auto"))
    cols[2].metric("Features", format_value(summary.get("feature_count", 0), "integer"))
    cols[3].metric("Mean Test Accuracy", format_value(summary.get("mean_test_accuracy", 0), "percent"))

    detail_items = [
        ("Trainer", summary.get("trainer_name"), "auto"),
        ("Fold Count", summary.get("fold_count"), "integer"),
        ("Integrity Proof", summary.get("proof_status"), "auto"),
        ("Invalid Folds", summary.get("invalid_fold_count"), "integer"),
    ]
    render_key_value_summary(detail_items)
    render_raw_expander("Raw details", summary)


def render_target_study_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(4)
    cols[0].metric("Study", format_value(summary.get("study_name"), "auto"))
    cols[1].metric("Targets", format_value(summary.get("target_count", 0), "integer"))
    cols[2].metric("Successful", format_value(summary.get("successful_targets", 0), "integer"))
    cols[3].metric(
        "Best Mean Accuracy",
        format_value(summary.get("best_mean_test_accuracy", 0), "percent"),
    )

    detail_items = [
        ("Best Target", summary.get("best_target_name"), "auto"),
        ("Integrity Proof", summary.get("proof_status"), "auto"),
    ]
    render_key_value_summary(detail_items)
    render_raw_expander("Raw details", summary)


def render_feature_study_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(4)
    cols[0].metric("Study", format_value(summary.get("study_name"), "auto"))
    cols[1].metric("Feature Sets", format_value(summary.get("feature_set_count", 0), "integer"))
    cols[2].metric("Successful Runs", format_value(summary.get("successful_runs", 0), "integer"))
    cols[3].metric(
        "Best Mean Accuracy",
        format_value(summary.get("best_mean_test_accuracy", 0), "percent"),
    )

    detail_items = [
        ("Working Target", summary.get("working_target_id"), "auto"),
        ("Best Feature Set", summary.get("best_feature_set_name"), "auto"),
        ("Target Count", summary.get("target_count"), "integer"),
    ]
    render_key_value_summary(detail_items)
    render_raw_expander("Raw details", summary)


def render_training_experiment_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(4)
    cols[0].metric("Experiment", format_value(summary.get("experiment_name"), "auto"))
    cols[1].metric("Feature Set", format_value(summary.get("feature_set_display_name") or summary.get("feature_set_name"), "auto"))
    cols[2].metric("Threshold", format_value(summary.get("selected_threshold"), "float"))
    cols[3].metric("Mean Test Accuracy", format_value(summary.get("mean_test_accuracy", 0), "percent"))

    detail_items = [
        ("Experiment ID", summary.get("experiment_id"), "auto"),
        ("Target Spec", summary.get("target_spec_id"), "auto"),
        ("Feature Set Meaning", summary.get("feature_set_description"), "auto"),
        ("Comparison Set", summary.get("comparison_feature_set_display_name") or summary.get("comparison_feature_set_name"), "auto"),
        ("Trainer", summary.get("trainer_name"), "auto"),
        ("Features", summary.get("feature_count"), "integer"),
        ("Fold Count", summary.get("fold_count"), "integer"),
        ("Integrity Proof", summary.get("proof_status"), "auto"),
        ("Invalid Folds", summary.get("invalid_fold_count"), "integer"),
        ("One-Class Fold Segments", summary.get("one_class_fold_count"), "integer"),
        ("Constant-Feature Folds", summary.get("constant_feature_fold_count"), "integer"),
    ]
    render_key_value_summary(detail_items)
    render_raw_expander("Raw details", summary)


def render_promotion_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(3)
    cols[0].metric("Experiment", format_value(summary.get("experiment_name"), "auto"))
    cols[1].metric("Feature Set", format_value(summary.get("feature_set_name"), "auto"))
    cols[2].metric("Threshold", format_value(summary.get("selected_threshold"), "float"))

    detail_items = [
        ("Experiment ID", summary.get("experiment_id"), "auto"),
        ("Promoted Model Path", summary.get("promoted_model_path"), "auto"),
        ("Integrity Proof", summary.get("proof_status"), "auto"),
    ]
    render_key_value_summary(detail_items)
    render_raw_expander("Raw details", summary)


def render_search_summary(summary: dict) -> None:
    if not summary:
        return

    cols = st.columns(4)
    cols[0].metric("Search", format_value(summary.get("search_name"), "auto"))
    cols[1].metric("Candidates", format_value(summary.get("candidate_count", 0), "integer"))
    cols[2].metric("Targets", format_value(summary.get("target_count", 0), "integer"))
    cols[3].metric("Winner Status", format_value(summary.get("winner_status"), "auto"))

    detail_items = [
        ("Search ID", summary.get("search_id"), "auto"),
        ("Target Spec", summary.get("target_spec_id"), "auto"),
        ("Searched Targets", ", ".join(summary.get("searched_target_display_names") or []), "auto"),
        ("Trainer", summary.get("trainer_name"), "auto"),
        ("Execution Mode", summary.get("execution_mode"), "auto"),
        ("Workers", summary.get("resolved_max_workers"), "integer"),
        ("Feature Sets", summary.get("feature_set_count"), "integer"),
        ("Successful Candidates", summary.get("successful_candidate_count"), "integer"),
        ("Failed Candidates", summary.get("failed_candidate_count"), "integer"),
        ("Recommended Experiment", summary.get("recommended_experiment_id"), "auto"),
        ("Recommended Target", summary.get("recommended_target_display_name") or summary.get("recommended_target_spec_id"), "auto"),
        ("Recommended Feature Set", summary.get("recommended_feature_set_display_name") or summary.get("recommended_feature_set_name"), "auto"),
        ("Feature Set Meaning", summary.get("recommended_feature_set_description"), "auto"),
        ("Recommended Preset", summary.get("recommended_preset_display_name") or summary.get("recommended_preset_name"), "auto"),
        ("Preset Meaning", summary.get("recommended_preset_description"), "auto"),
        ("Recommended Threshold", summary.get("recommended_selected_threshold"), "float"),
        ("Winner Reason", summary.get("winner_reason"), "auto"),
        ("Elapsed Seconds", summary.get("elapsed_seconds"), "float"),
        ("Truth-Gate Passes", summary.get("truth_gate_pass_count"), "integer"),
        ("Test-Guardrail Passes", summary.get("test_guardrail_pass_count"), "integer"),
        ("Truth-Gate Failure Counts", summary.get("truth_gate_failures"), "auto"),
        ("Integrity-Failed Candidates", summary.get("integrity_failure_candidate_count"), "integer"),
        ("Low-Coverage Candidates", summary.get("low_coverage_candidate_count"), "integer"),
        ("Below-Majority Candidates", summary.get("majority_dominance_candidate_count"), "integer"),
        ("One-Class Candidates", summary.get("one_class_candidate_count"), "integer"),
    ]
    render_key_value_summary(detail_items)
    render_raw_expander("Raw details", summary)


def render_integrity_summary(integrity: dict) -> None:
    if not integrity:
        return

    proof_status = str(integrity.get("proof_status") or "missing").lower()
    warnings = integrity.get("warnings") or []
    if proof_status == "passed":
        st.success("Integrity proof passed.")
    elif proof_status == "missing":
        st.warning("Integrity proof is missing for this saved report.")
    else:
        st.error("Integrity proof failed for this saved report.")

    if warnings:
        for warning in warnings:
            severity = str(warning.get("severity") or "info").lower()
            message = str(warning.get("message") or warning.get("code") or "Integrity warning")
            if severity == "critical":
                st.error(message)
            elif severity == "warning":
                st.warning(message)
            else:
                st.info(message)

    overview = integrity.get("overview") or {}
    if overview:
        render_key_value_summary([(key.replace("_", " ").title(), value, "auto") for key, value in overview.items()])

    render_raw_expander("Integrity details", integrity)


def render_diagnostics_summary(diagnostics: dict) -> None:
    if not diagnostics:
        return

    warnings = diagnostics.get("warnings") or []
    if warnings:
        for warning in warnings:
            severity = str(warning.get("severity") or "info").lower()
            message = str(warning.get("message") or warning.get("code") or "Diagnostics warning")
            if severity == "critical":
                st.error(message)
            elif severity == "warning":
                st.warning(message)
            else:
                st.info(message)

    overview = diagnostics.get("overview") or diagnostics.get("summary") or {}
    if overview:
        render_key_value_summary([(key.replace("_", " ").title(), value, "auto") for key, value in overview.items()])

    render_raw_expander("Diagnostics details", diagnostics)


def render_artifact_summary(artifacts: dict, *, saved_model_name: str | None = None) -> None:
    if not artifacts and not saved_model_name:
        return

    items: list[tuple[str, Any, str]] = []
    if saved_model_name:
        items.append(("Saved Model", saved_model_name, "auto"))

    artifact_fields = [
        ("Model Path", artifacts.get("model_path"), "auto"),
        ("Report File", artifacts.get("report_file"), "auto"),
        ("Chart File", artifacts.get("chart_file"), "auto"),
        ("Comparison Chart File", artifacts.get("comparison_chart_file"), "auto"),
        ("Comparison Curve File", artifacts.get("comparison_curve_file"), "auto"),
        ("Trade Summary File", artifacts.get("trade_summary_file"), "auto"),
        ("Evaluation Rows File", artifacts.get("evaluation_rows_file"), "auto"),
        ("Prediction Rows File", artifacts.get("prediction_rows_file"), "auto"),
        ("Threshold Metrics File", artifacts.get("threshold_metrics_file"), "auto"),
        ("Calibration File", artifacts.get("calibration_file"), "auto"),
        ("Inventory File", artifacts.get("inventory_file"), "auto"),
        ("Fold Selection File", artifacts.get("fold_selection_file"), "auto"),
        ("Stability File", artifacts.get("stability_file"), "auto"),
        ("Comparison File", artifacts.get("comparison_file"), "auto"),
        ("Resolved Features File", artifacts.get("resolved_features_file"), "auto"),
        ("Fold Integrity File", artifacts.get("fold_integrity_file"), "auto"),
        ("Fold Diagnostics File", artifacts.get("fold_diagnostics_file"), "auto"),
        ("Threshold Coverage Diagnostics File", artifacts.get("threshold_coverage_diagnostics_file"), "auto"),
        ("Feature Health Diagnostics File", artifacts.get("feature_health_diagnostics_file"), "auto"),
        ("Leaderboard File", artifacts.get("leaderboard_file"), "auto"),
        ("Candidates File", artifacts.get("candidates_file"), "auto"),
    ]
    items.extend([item for item in artifact_fields if item[1]])

    render_key_value_summary(items)
    render_raw_expander("Raw details", artifacts)
