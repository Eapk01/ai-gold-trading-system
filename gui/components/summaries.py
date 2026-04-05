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
        ("Trade Summary File", artifacts.get("trade_summary_file"), "auto"),
        ("Evaluation Rows File", artifacts.get("evaluation_rows_file"), "auto"),
    ]
    items.extend([item for item in artifact_fields if item[1]])

    render_key_value_summary(items)
    render_raw_expander("Raw details", artifacts)
