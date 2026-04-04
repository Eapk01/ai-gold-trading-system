"""Theme helpers for the Streamlit GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - used only in non-GUI test environments
    class _StreamlitFallback:
        def __init__(self) -> None:
            self.session_state: dict[str, str] = {}

        def markdown(self, *_args, **_kwargs) -> None:
            return None

    st = _StreamlitFallback()


ThemeMode = Literal["dark", "light"]

DEFAULT_THEME_MODE: ThemeMode = "dark"
VALID_THEME_MODES: tuple[ThemeMode, ThemeMode] = ("dark", "light")
PREFERENCES_PATH = Path.home() / ".ai-gold-research-system" / "ui_preferences.json"

_THEME_TOKENS: dict[ThemeMode, dict[str, str]] = {
    "dark": {
        "app_background": "#0e1117",
        "sidebar_background": "#262730",
        "surface_background": "#111827",
        "app_text": "#fafafa",
        "muted_text": "rgba(250, 250, 250, 0.72)",
        "input_background": "#111827",
        "input_text": "#fafafa",
        "input_border": "rgba(100, 116, 139, 0.30)",
        "input_placeholder": "rgba(250, 250, 250, 0.45)",
        "button_background": "rgba(255, 255, 255, 0.04)",
        "button_hover": "rgba(255, 255, 255, 0.08)",
        "border_subtle": "rgba(100, 116, 139, 0.20)",
        "border_strong": "rgba(100, 116, 139, 0.26)",
        "chart_grid": "rgba(148, 163, 184, 0.14)",
        "chart_axis": "rgba(250, 250, 250, 0.72)",
        "chart_background": "#111827",
        "sidebar_hover": "rgba(255, 255, 255, 0.06)",
        "sidebar_active_border": "rgba(255, 255, 255, 0.12)",
        "sidebar_active_background": "rgba(255, 255, 255, 0.04)",
        "sidebar_active_text": "rgba(255, 255, 255, 0.85)",
        "toggle_background": "rgba(255, 255, 255, 0.03)",
        "toggle_border": "rgba(255, 255, 255, 0.08)",
        "badge_stopped_background": "#e5e7eb",
        "badge_stopped_text": "#374151",
        "badge_stale_background": "#fef3c7",
        "badge_stale_text": "#92400e",
        "badge_live_background": "#dcfce7",
        "badge_live_text": "#166534",
    },
    "light": {
        "app_background": "#f6f8fb",
        "sidebar_background": "#f3f4f6",
        "surface_background": "#ffffff",
        "app_text": "#172033",
        "muted_text": "rgba(23, 32, 51, 0.72)",
        "input_background": "#ffffff",
        "input_text": "#172033",
        "input_border": "rgba(71, 85, 105, 0.24)",
        "input_placeholder": "rgba(23, 32, 51, 0.42)",
        "button_background": "#ffffff",
        "button_hover": "rgba(148, 163, 184, 0.18)",
        "border_subtle": "rgba(71, 85, 105, 0.18)",
        "border_strong": "rgba(71, 85, 105, 0.26)",
        "chart_grid": "rgba(71, 85, 105, 0.14)",
        "chart_axis": "rgba(23, 32, 51, 0.74)",
        "chart_background": "#ffffff",
        "sidebar_hover": "rgba(15, 23, 42, 0.05)",
        "sidebar_active_border": "rgba(30, 41, 59, 0.12)",
        "sidebar_active_background": "rgba(148, 163, 184, 0.15)",
        "sidebar_active_text": "#0f172a",
        "toggle_background": "rgba(255, 255, 255, 0.88)",
        "toggle_border": "rgba(71, 85, 105, 0.18)",
        "badge_stopped_background": "#e2e8f0",
        "badge_stopped_text": "#334155",
        "badge_stale_background": "#fde68a",
        "badge_stale_text": "#92400e",
        "badge_live_background": "#bbf7d0",
        "badge_live_text": "#166534",
    },
}


def get_preferences_path() -> Path:
    return PREFERENCES_PATH


def normalize_theme_mode(value: str | None) -> ThemeMode:
    if value in VALID_THEME_MODES:
        return value
    return DEFAULT_THEME_MODE


def load_theme_mode(preferences_path: Path | None = None) -> ThemeMode:
    path = preferences_path or get_preferences_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return DEFAULT_THEME_MODE

    return normalize_theme_mode(data.get("theme_mode"))


def save_theme_mode(mode: str, preferences_path: Path | None = None) -> ThemeMode:
    normalized_mode = normalize_theme_mode(mode)
    path = preferences_path or get_preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"theme_mode": normalized_mode}, indent=2), encoding="utf-8")
    return normalized_mode


def initialize_theme_mode() -> ThemeMode:
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = load_theme_mode()
    return get_theme_mode()


def get_theme_mode() -> ThemeMode:
    return normalize_theme_mode(st.session_state.get("theme_mode"))


def set_theme_mode(mode: str) -> ThemeMode:
    normalized_mode = save_theme_mode(mode)
    st.session_state["theme_mode"] = normalized_mode
    return normalized_mode


def get_theme_tokens(mode: str | None = None) -> dict[str, str]:
    active_mode = normalize_theme_mode(mode or get_theme_mode())
    return dict(_THEME_TOKENS[active_mode])


def style_altair_chart(chart: Any) -> Any:
    tokens = get_theme_tokens()
    return chart.configure(
        background=tokens["chart_background"],
        padding={"left": 8, "right": 8, "top": 10, "bottom": 8},
    ).configure_view(
        stroke=None,
    ).configure_axis(
        domainColor=tokens["border_subtle"],
        gridColor=tokens["chart_grid"],
        labelColor=tokens["chart_axis"],
        tickColor=tokens["border_subtle"],
        titleColor=tokens["app_text"],
    ).configure_legend(
        labelColor=tokens["app_text"],
        titleColor=tokens["app_text"],
    ).configure_title(
        color=tokens["app_text"],
    )


def inject_theme_styles(mode: str | None = None) -> None:
    tokens = get_theme_tokens(mode)
    st.markdown(
        f"""
        <style>
        :root {{
            --app-background: {tokens["app_background"]};
            --sidebar-background: {tokens["sidebar_background"]};
            --surface-background: {tokens["surface_background"]};
            --app-text: {tokens["app_text"]};
            --muted-text: {tokens["muted_text"]};
            --input-background: {tokens["input_background"]};
            --input-text: {tokens["input_text"]};
            --input-border: {tokens["input_border"]};
            --input-placeholder: {tokens["input_placeholder"]};
            --button-background: {tokens["button_background"]};
            --button-hover: {tokens["button_hover"]};
            --border-subtle: {tokens["border_subtle"]};
            --border-strong: {tokens["border_strong"]};
            --sidebar-hover: {tokens["sidebar_hover"]};
            --sidebar-active-border: {tokens["sidebar_active_border"]};
            --sidebar-active-background: {tokens["sidebar_active_background"]};
            --sidebar-active-text: {tokens["sidebar_active_text"]};
            --toggle-background: {tokens["toggle_background"]};
            --toggle-border: {tokens["toggle_border"]};
        }}

        html,
        body,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"] {{
            background: var(--app-background) !important;
            color: var(--app-text);
        }}

        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stHeader"] {{
            background: transparent !important;
            color: var(--app-text) !important;
        }}

        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {{
            background: var(--sidebar-background) !important;
            color: var(--app-text) !important;
        }}

        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] div,
        [data-testid="stAppViewContainer"] th,
        [data-testid="stAppViewContainer"] td,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: inherit !important;
        }}

        [data-testid="stCaptionContainer"] {{
            color: var(--muted-text) !important;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMetricLabel"] div,
        [data-testid="stMetricValue"] div,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabelHelpInline"] {{
            color: var(--app-text) !important;
        }}

        [data-testid="stMetric"],
        [data-testid="stTable"],
        [data-testid="stDataFrame"],
        [data-testid="stDataFrameResizable"],
        [data-testid="stExpander"],
        [data-testid="stPopover"] {{
            color: var(--app-text) !important;
        }}

        [data-testid="stTable"] table,
        [data-testid="stTable"] thead tr,
        [data-testid="stTable"] tbody tr,
        [data-testid="stDataFrameResizable"],
        [data-testid="stDataFrame"] div[role="grid"],
        [data-testid="stDataFrame"] div[role="row"],
        [data-testid="stDataFrame"] div[role="gridcell"] {{
            background: var(--surface-background) !important;
            color: var(--app-text) !important;
            border-color: var(--border-subtle) !important;
        }}

        .stDataFrameGlideDataEditor {{
            --gdg-text-dark: var(--app-text) !important;
            --gdg-text-medium: var(--muted-text) !important;
            --gdg-text-light: var(--muted-text) !important;
            --gdg-text-bubble: var(--muted-text) !important;
            --gdg-bg-icon-header: var(--muted-text) !important;
            --gdg-fg-icon-header: var(--app-text) !important;
            --gdg-text-header: var(--muted-text) !important;
            --gdg-text-group-header: var(--muted-text) !important;
            --gdg-text-header-selected: var(--app-text) !important;
            --gdg-bg-cell: var(--surface-background) !important;
            --gdg-bg-cell-medium: var(--surface-background) !important;
            --gdg-bg-header: var(--surface-background) !important;
            --gdg-bg-header-has-focus: var(--button-hover) !important;
            --gdg-bg-header-hovered: var(--button-hover) !important;
            --gdg-bg-bubble: var(--button-background) !important;
            --gdg-bg-bubble-selected: var(--button-hover) !important;
            --gdg-bg-search-result: var(--button-hover) !important;
            --gdg-border-color: var(--border-subtle) !important;
            --gdg-horizontal-border-color: var(--border-subtle) !important;
            --gdg-drilldown-border: var(--input-border) !important;
        }}

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {{
            background: transparent !important;
        }}

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button,
        [data-baseweb="button"],
        [data-testid="stPopoverButton"],
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-secondaryFormSubmit"] {{
            background: var(--button-background) !important;
            border-color: var(--input-border) !important;
            color: var(--app-text) !important;
        }}

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stDownloadButton > button:hover,
        [data-baseweb="button"]:hover,
        [data-testid="stPopoverButton"]:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stBaseButton-secondaryFormSubmit"]:hover {{
            background: var(--button-hover) !important;
        }}

        .stSelectbox label,
        .stToggle label,
        .stCheckbox label {{
            color: var(--app-text) !important;
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] > div,
        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stNumberInputRootElement"] > div {{
            background: var(--surface-background) !important;
            color: var(--app-text) !important;
            border-color: var(--input-border) !important;
        }}

        input[data-testid="stNumberInputField"],
        input[type="text"],
        input[type="number"],
        textarea {{
            background: var(--input-background) !important;
            color: var(--input-text) !important;
            caret-color: var(--input-text) !important;
        }}

        input[data-testid="stNumberInputField"]::placeholder,
        input[type="text"]::placeholder,
        input[type="number"]::placeholder,
        textarea::placeholder {{
            color: var(--input-placeholder) !important;
        }}

        [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child,
        [data-testid="stToggle"] [data-baseweb="checkbox"] > div:first-child {{
            background: var(--surface-background) !important;
            border-color: var(--input-border) !important;
        }}

        [data-testid="stCheckbox"] [data-baseweb="checkbox"] > span:first-child {{
            background: var(--surface-background) !important;
            border-color: var(--input-border) !important;
            box-shadow: none !important;
        }}

        [data-testid="stPopover"] [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stPopover"] .st-bt {{
            background: var(--surface-background) !important;
            color: var(--app-text) !important;
        }}

        button[title="Show password text"],
        button[title="Hide password text"] {{
            background: var(--surface-background) !important;
            color: var(--app-text) !important;
            border: 1px solid var(--input-border) !important;
        }}

        [data-testid="stVegaLiteChart"],
        [data-testid="stVegaLiteChart"] details,
        [data-testid="stVegaLiteChart"] summary,
        [data-testid="stVegaLiteChart"] .vega-actions,
        [data-testid="stVegaLiteChart"] .vega-actions a {{
            background: var(--surface-background) !important;
            color: var(--app-text) !important;
            border-color: var(--input-border) !important;
        }}

        [data-testid="stVegaLiteChart"] details {{
            border-radius: 0.5rem;
        }}

        [data-testid="stVegaLiteChart"] canvas,
        .chart-wrapper {{
            background: var(--chart-background) !important;
        }}

        [data-testid="stSidebar"] .stButton {{
            margin-bottom: 0.15rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            justify-content: flex-start !important;
            font-size: 0.9rem !important;
            padding: 0.26rem 0.55rem !important;
            border-radius: 0.5rem !important;
            min-height: 2.05rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"] {{
            border: 1px solid transparent !important;
            background: transparent !important;
            box-shadow: none !important;
        }}

        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-tertiary"]:hover {{
            border-color: transparent !important;
            background: var(--sidebar-hover) !important;
        }}

        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] {{
            border: 1px solid var(--sidebar-active-border) !important;
            background: var(--sidebar-active-background) !important;
            color: var(--sidebar-active-text) !important;
            transition: all 0.15s ease;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
