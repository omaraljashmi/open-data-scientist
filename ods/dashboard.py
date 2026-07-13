"""Explainable automatic chart recommendations for uploaded datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

ChartKind = Literal[
    "missingness",
    "category_count",
    "histogram",
    "time_series",
    "category_average",
    "scatter",
]


@dataclass(frozen=True)
class ColumnRoles:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    datetime: tuple[str, ...]
    identifiers: tuple[str, ...]


@dataclass(frozen=True)
class ChartSuggestion:
    kind: ChartKind
    title: str
    explanation: str
    x: str
    y: str | None = None


def infer_column_roles(df: pd.DataFrame) -> ColumnRoles:
    """Infer simple semantic roles without modifying the uploaded DataFrame."""
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    identifiers: list[str] = []
    row_count = len(df)

    for raw_name in df.columns:
        name = str(raw_name)
        series = df[raw_name]
        unique = int(series.nunique(dropna=True))
        lowered = name.lower()
        looks_like_id = lowered == "id" or lowered.endswith("_id") or lowered.startswith("id_")
        is_unique = row_count >= 20 and unique == row_count

        if looks_like_id or is_unique:
            identifiers.append(name)
            continue
        if is_datetime64_any_dtype(series) or _looks_like_datetime(series, lowered):
            datetime.append(name)
            continue
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            numeric.append(name)
            continue
        if (
            is_bool_dtype(series)
            or is_string_dtype(series)
            or is_object_dtype(series)
        ) and 1 < unique <= 50:
            categorical.append(name)

    return ColumnRoles(tuple(numeric), tuple(categorical), tuple(datetime), tuple(identifiers))


def suggest_dashboard(df: pd.DataFrame, max_charts: int = 4) -> tuple[ChartSuggestion, ...]:
    """Recommend a concise dashboard using transparent, deterministic rules."""
    if df.empty or max_charts <= 0:
        return ()

    roles = infer_column_roles(df)
    suggestions: list[ChartSuggestion] = []

    if bool(df.isna().any().any()):
        suggestions.append(
            ChartSuggestion(
                "missingness",
                "Missing values by column",
                "Highlights columns that may require cleaning before analysis.",
                "column",
                "missing_percent",
            )
        )
    if roles.categorical:
        category = roles.categorical[0]
        suggestions.append(
            ChartSuggestion(
                "category_count",
                f"Records by {category}",
                "Counts the most common categories, including missing values.",
                category,
                "count",
            )
        )
    if roles.numeric:
        measure = roles.numeric[0]
        suggestions.append(
            ChartSuggestion(
                "histogram",
                f"Distribution of {measure}",
                "Groups numeric values into bins to show their overall distribution.",
                measure,
                "count",
            )
        )
    if roles.datetime:
        date = roles.datetime[0]
        measure = roles.numeric[0] if roles.numeric else None
        suggestions.append(
            ChartSuggestion(
                "time_series",
                f"{measure or 'Records'} over time",
                f"Shows {'the daily average of ' + measure if measure else 'daily record counts'} by {date}.",
                date,
                measure,
            )
        )
    if len(suggestions) < max_charts and roles.categorical and roles.numeric:
        category, measure = roles.categorical[0], roles.numeric[0]
        suggestions.append(
            ChartSuggestion(
                "category_average",
                f"Average {measure} by {category}",
                "Compares the average numeric value across categories.",
                category,
                measure,
            )
        )
    if len(suggestions) < max_charts and len(roles.numeric) >= 2:
        x, y = roles.numeric[:2]
        suggestions.append(
            ChartSuggestion(
                "scatter",
                f"{y} versus {x}",
                "Shows whether the two numeric variables move together.",
                x,
                y,
            )
        )

    return tuple(suggestions[:max_charts])


def build_chart_data(df: pd.DataFrame, suggestion: ChartSuggestion) -> pd.DataFrame:
    """Prepare the small DataFrame consumed by a native Streamlit chart."""
    if suggestion.kind == "missingness":
        missing = (df.isna().mean() * 100).round(2)
        return (
            missing[missing > 0]
            .sort_values(ascending=False)
            .rename("missing_percent")
            .rename_axis("column")
            .reset_index()
        )

    if suggestion.kind == "category_count":
        values = df[suggestion.x].astype("string").fillna("(missing)")
        return (
            values.value_counts(dropna=False)
            .head(15)
            .rename("count")
            .rename_axis(suggestion.x)
            .reset_index()
        )

    if suggestion.kind == "histogram":
        values = pd.to_numeric(df[suggestion.x], errors="coerce").dropna()
        if values.empty:
            return pd.DataFrame(columns=[suggestion.x, "count"])
        unique = max(int(values.nunique()), 1)
        bins = min(12, max(3, round(unique**0.5)))
        buckets = pd.cut(values, bins=bins, duplicates="drop")
        counts = buckets.value_counts(sort=False)
        return pd.DataFrame({suggestion.x: counts.index.astype(str), "count": counts.values})

    if suggestion.kind == "time_series":
        dates = pd.to_datetime(df[suggestion.x], errors="coerce", format="mixed").dt.floor("D")
        if suggestion.y:
            values = pd.to_numeric(df[suggestion.y], errors="coerce")
            prepared = pd.DataFrame({suggestion.x: dates, suggestion.y: values}).dropna()
            return prepared.groupby(suggestion.x, as_index=False)[suggestion.y].mean()
        prepared = pd.DataFrame({suggestion.x: dates}).dropna()
        return prepared.value_counts().rename("count").reset_index().sort_values(suggestion.x)

    if suggestion.kind == "category_average" and suggestion.y:
        prepared = pd.DataFrame(
            {
                suggestion.x: df[suggestion.x].astype("string").fillna("(missing)"),
                suggestion.y: pd.to_numeric(df[suggestion.y], errors="coerce"),
            }
        ).dropna(subset=[suggestion.y])
        return (
            prepared.groupby(suggestion.x, as_index=False)[suggestion.y]
            .mean()
            .sort_values(suggestion.y, ascending=False)
            .head(15)
        )

    if suggestion.kind == "scatter" and suggestion.y:
        return df[[suggestion.x, suggestion.y]].apply(pd.to_numeric, errors="coerce").dropna().head(5000)

    return pd.DataFrame()


def _looks_like_datetime(series: pd.Series, lowered_name: str) -> bool:
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False
    name_hint = any(token in lowered_name for token in ("date", "time", "day", "month", "year"))
    sample = series.dropna().head(100)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    threshold = 0.8 if name_hint else 0.95
    return float(parsed.notna().mean()) >= threshold
