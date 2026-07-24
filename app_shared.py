"""Shared constants, label maps, and helper functions used across all ODS pages.

Everything here is pure Python — no st.* calls — so it is safe to import at
module level from any page without triggering Streamlit side-effects.
"""

from __future__ import annotations

from html import escape
import re
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from ods import ChartSuggestion, CleaningAction, build_chart_data, build_chart_figure

# ── File paths ────────────────────────────────────────────────────────────────

# app_shared.py lives at  <root>/app_shared.py
# sample data lives at    <root>/examples/sample_customers.csv
SAMPLE_DATA_PATH = (
    Path(__file__).resolve().parent / "examples" / "sample_customers.csv"
)

# ── UI label maps ─────────────────────────────────────────────────────────────

ROLE_LABELS: dict[str, str] = {
    "identifier": "Identifier / ID",
    "numeric":    "Numeric measure",
    "categorical":"Category",
    "datetime":   "Date / time",
    "text":       "Free text",
    "ignore":     "Ignore",
}

ROLE_VALUES: dict[str, str] = {label: role for role, label in ROLE_LABELS.items()}

AGGREGATIONS: dict[str, str] = {
    "Average": "mean",
    "Median":  "median",
    "Total":   "sum",
    "Count":   "count",
    "Minimum": "min",
    "Maximum": "max",
}

FILTER_OPERATORS: dict[str, str] = {
    "Equals":        "eq",
    "Does not equal":"ne",
    "Greater than":  "gt",
    "At least":      "gte",
    "Less than":     "lt",
    "At most":       "lte",
    "Contains":      "contains",
    "Starts with":   "starts_with",
    "Is missing":    "is_null",
    "Is not missing":"is_not_null",
}

SUMMARY_CALCULATIONS: dict[str, str] = {
    "Count rows":       "count_rows",
    "Count non-missing":"count_values",
    "Average":          "mean",
    "Median":           "median",
    "Total":            "sum",
    "Minimum":          "min",
    "Maximum":          "max",
}

STUDIO_CARD_LABELS: dict[str, str] = {
    "kpi":          "KPI card",
    "bar":          "Bar chart",
    "pie":          "Pie chart",
    "line":         "Line chart",
    "scatter":      "Scatter plot",
    "distribution": "Distribution",
}

STUDIO_CARD_VALUES: dict[str, str] = {
    label: value for value, label in STUDIO_CARD_LABELS.items()
}

STUDIO_METRIC_LABELS: dict[str, str] = {
    "row_count":     "Row count",
    "sum":           "Total",
    "mean":          "Average",
    "median":        "Median",
    "distinct_count":"Distinct count",
}

STUDIO_METRIC_VALUES: dict[str, str] = {
    label: value for value, label in STUDIO_METRIC_LABELS.items()
}

STUDIO_DATE_GRAINS: dict[str, str] = {
    "Day":     "day",
    "Week":    "week",
    "Month":   "month",
    "Quarter": "quarter",
    "Year":    "year",
}

# ── Cleaning session state ────────────────────────────────────────────────────

def get_cleaning_state(dataset_key: str) -> dict[str, object]:
    """Return (and initialise if absent) the cleaning history for one file."""
    state_key = f"cleaning-state-{dataset_key}"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or "batches" not in state:
        state = {"batches": [], "preview": None}
        st.session_state[state_key] = state
    return state


def cleaning_history_fingerprint(
    batches: list[tuple[CleaningAction, ...]],
) -> str:
    """Stable short hash of the applied cleaning history — used as a cache key."""
    serialized = tuple(
        tuple((action.action_id, action.parameters) for action in batch)
        for batch in batches
    )
    return sha256(repr(serialized).encode()).hexdigest()[:10]

# ── Formatting ────────────────────────────────────────────────────────────────

def format_bytes(size: int) -> str:
    """Human-readable byte size (mirrors _format_bytes in loader.py — kept separate
    so the modules stay independent)."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"

# ── Chart rendering ───────────────────────────────────────────────────────────

def render_chart(dataframe: pd.DataFrame, suggestion: ChartSuggestion) -> None:
    """Render one ChartSuggestion as a themed Plotly figure + audit expander."""
    chart_data = build_chart_data(dataframe, suggestion)
    figure = build_chart_figure(dataframe, suggestion, chart_data)
    if chart_data.empty or figure is None:
        st.info("There is not enough usable data to draw this chart.")
        return

    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True})
    st.caption(f"Recommendation confidence: {suggestion.confidence:.0%}")
    with st.expander("Verify calculation"):
        st.dataframe(chart_data, width="stretch", hide_index=True)
        st.caption(
            f"This is the exact {len(chart_data):,}-row summary used by the chart. "
            "No hidden model or paid API is involved."
        )

# ── Filter helpers ────────────────────────────────────────────────────────────

def filter_operator_labels(series: pd.Series) -> list[str]:
    """Sensible filter operator choices for a column's dtype."""
    if is_bool_dtype(series):
        return ["Equals", "Does not equal", "Is missing", "Is not missing"]
    if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
        return [
            "Equals", "Does not equal",
            "Greater than", "At least",
            "Less than", "At most",
            "Is missing", "Is not missing",
        ]
    return [
        "Equals", "Does not equal",
        "Contains", "Starts with",
        "Is missing", "Is not missing",
    ]


def default_aggregate_alias(function: str, column: str | None) -> str:
    """Short readable result-column name for a summary calculation."""
    labels = {
        "count_rows":   "row_count",
        "count_values": "count",
        "mean":         "average",
        "median":       "median",
        "sum":          "total",
        "min":          "minimum",
        "max":          "maximum",
    }
    return labels[function] if column is None else f"{labels[function]}_{column}"

# ── Dashboard Studio UI helpers ───────────────────────────────────────────────

def option_index(options: list[str], value: str) -> int:
    """Safe selectbox index — falls back to 0 if value not in list."""
    return options.index(value) if value in options else 0


def safe_download_stem(value: str) -> str:
    """Cross-platform filename stem (≤60 chars, no special chars)."""
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return (normalized or "ods-dashboard")[:60]

# ── Shared visual theme ───────────────────────────────────────────────────────

THEME_CSS = """
<style>
:root {
  --ods-bg: #f7f3e7;
  --ods-surface: #fffdf6;
  --ods-border: rgba(20, 83, 45, .16);
  --ods-text: #16281e;
  --ods-muted: #5f6f5f;
  --ods-accent: #15803d;
  --ods-pitch: #123a26;
  --ods-cream: #f3efdf;
}
.stApp { background: var(--ods-bg); color: var(--ods-text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
/* Central reading column, like a match-day programme page */
[data-testid="stMainBlockContainer"], .stMain .block-container {
  max-width: 1080px; margin: 0 auto;
}
/* Pitch-green sidebar with cream navigation */
[data-testid="stSidebar"] { background: var(--ods-pitch); border-right: none; }
[data-testid="stSidebar"] a { background: transparent !important; }
[data-testid="stSidebar"] a span { color: var(--ods-cream); }
[data-testid="stSidebar"] a:hover { background: rgba(243, 239, 223, .1) !important; }
[data-testid="stSidebar"] a[aria-current="page"] {
  background: rgba(243, 239, 223, .16) !important;
}
[data-testid="stSidebar"] a[aria-current="page"] span { color: #ffffff; }
[data-testid="stSidebar"] button svg { color: var(--ods-cream); fill: var(--ods-cream); }
/* Sidebar headings, widget labels, and captions must stay readable on green;
   widget inputs keep their own cream background and dark text. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] :is(h1, h2, h3, h4, p) {
  color: var(--ods-cream);
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--ods-cream); }
/* ...but button labels sit on cream chips, so they keep dark text */
[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] p {
  color: var(--ods-text);
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color: rgba(243, 239, 223, .75);
}
/* Streamlit's st.info is blue by default; tint it to the palette instead.
   Success, warning, and error keep their universal colors. */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
  background: rgba(21, 128, 61, .09); color: #1d3b29; border-radius: 12px;
}
h1, h2, h3 { letter-spacing: -.02em; }
[data-testid="stMetric"] {
  background: var(--ods-surface); border: 1px solid var(--ods-border);
  border-radius: 12px; padding: .85rem 1rem;
}
[data-testid="stMetricLabel"] p {
  color: var(--ods-muted); font-size: .72rem;
  text-transform: uppercase; letter-spacing: .08em;
}
[data-testid="stMetricValue"] { color: var(--ods-accent); }
.stButton button, .stDownloadButton button { border-radius: 10px; }
[data-testid="stExpander"] details {
  border: 1px solid var(--ods-border); border-radius: 12px; background: var(--ods-surface);
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--ods-border) !important; border-radius: 14px; background: var(--ods-surface);
}
[data-testid="stFileUploader"] section {
  background: var(--ods-surface); border: 1px dashed rgba(21, 128, 61, .45); border-radius: 12px;
}
hr { border-color: var(--ods-border); }
.ods-kicker {
  color: var(--ods-accent); font-size: .68rem; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase;
}
.ods-page {
  text-align: center; padding-bottom: .8rem;
  border-bottom: 2px solid var(--ods-accent); margin-bottom: 1.2rem;
}
.ods-page h1 { font-size: 1.9rem; margin: .15rem 0 .3rem; }
.ods-page p { color: var(--ods-muted); max-width: 780px; margin: 0 auto; line-height: 1.55; }
.ods-hero { text-align: center; padding-top: .6rem; }
.ods-hero h1 {
  color: var(--ods-text);
  font-size: clamp(2.2rem, 4.5vw, 3.4rem); line-height: 1.04;
  letter-spacing: -.035em; margin: .4rem 0 .7rem;
}
/* Streamlit wraps heading text in its own span, so target the class, not bare spans */
.ods-hero h1 .ods-accent { color: var(--ods-accent); }
.ods-hero p {
  color: var(--ods-muted); max-width: 660px; margin: 0 auto;
  font-size: 1.02rem; line-height: 1.65;
}
</style>
"""


def apply_theme() -> None:
    """Inject the shared visual theme (call once near the top of each page)."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str) -> None:
    """Render the consistent page header: theme, kicker, title, muted subtitle."""
    apply_theme()
    st.markdown(
        '<div class="ods-page"><div class="ods-kicker">Open Data Scientist</div>'
        f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )
