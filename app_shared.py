"""Shared constants, label maps, and helper functions used across all ODS pages.

Everything here is pure Python — no st.* calls — so it is safe to import at
module level from any page without triggering Streamlit side-effects.
"""

from __future__ import annotations

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

from ods import ChartSuggestion, CleaningAction, build_chart_data

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
    """Render one ChartSuggestion using Streamlit native charts + audit expander."""
    chart_data = build_chart_data(dataframe, suggestion)
    if chart_data.empty:
        st.info("There is not enough usable data to draw this chart.")
        return

    if suggestion.kind == "time_series":
        value_col = suggestion.y or "count"
        st.line_chart(chart_data.set_index(suggestion.x)[value_col], height=300)
    elif suggestion.kind == "scatter" and suggestion.y:
        st.scatter_chart(chart_data, x=suggestion.x, y=suggestion.y, height=300)
    else:
        value_col = suggestion.y or "count"
        st.bar_chart(chart_data.set_index(suggestion.x)[value_col], height=300)

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
