"""Guided, explainable dashboard recommendations for uploaded datasets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import re
from typing import Literal, Mapping

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
    "category_aggregate",
    "scatter",
]
ColumnRole = Literal["identifier", "numeric", "categorical", "datetime", "text", "ignore"]
Aggregation = Literal["mean", "median", "sum", "count", "min", "max"]
DateGrain = Literal["day", "week", "month", "quarter", "year"]


@dataclass(frozen=True)
class ColumnSemantic:
    column: str
    role: ColumnRole
    confidence: float
    display_format: str
    reason: str


@dataclass(frozen=True)
class ColumnRoles:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    datetime: tuple[str, ...]
    identifiers: tuple[str, ...]
    text: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChartSuggestion:
    kind: ChartKind
    title: str
    explanation: str
    x: str
    y: str | None = None
    aggregation: Aggregation = "count"
    date_grain: DateGrain = "month"
    confidence: float = 0.8


INTENTS = (
    "Overview",
    "Trends over time",
    "Compare categories",
    "Understand distributions",
    "Explore relationships",
    "Review data quality",
)


def infer_column_semantics(df: pd.DataFrame) -> tuple[ColumnSemantic, ...]:
    """Infer roles with a visible confidence and reason for user review."""
    semantics: list[ColumnSemantic] = []

    for raw_name in df.columns:
        name = str(raw_name)
        series = df[raw_name]
        non_null = series.dropna()
        unique = int(non_null.nunique())
        lowered = name.lower()
        name_tokens = _name_tokens(lowered)
        unique_ratio = unique / max(len(non_null), 1)

        id_name = (
            lowered == "id"
            or lowered.endswith("_id")
            or lowered.startswith("id_")
            or bool(name_tokens & {"uuid", "guid"})
            or (
                unique_ratio >= 0.8
                and bool(name_tokens & {"key", "code", "number"})
            )
        )
        if id_name:
            semantics.append(ColumnSemantic(name, "identifier", 0.99, "ID", "The column name indicates an identifier."))
            continue
        if non_null.empty:
            semantics.append(
                ColumnSemantic(name, "ignore", 0.99, "Empty", "The column has no non-null values to analyze.")
            )
            continue
        if unique == 1:
            semantics.append(
                ColumnSemantic(name, "ignore", 0.97, "Constant", "The column has only one non-null value and cannot vary in a chart.")
            )
            continue
        if is_datetime64_any_dtype(series):
            semantics.append(ColumnSemantic(name, "datetime", 0.99, "Date/time", "Pandas loaded this column as a date/time."))
            continue

        if _looks_like_numeric_year(series, lowered):
            semantics.append(
                ColumnSemantic(
                    name,
                    "datetime",
                    0.96,
                    "Year",
                    "The column name and four-digit values indicate calendar years.",
                )
            )
            continue

        date_score = _datetime_parse_score(series)
        date_hint = bool(name_tokens & {"date", "datetime", "timestamp", "time", "day", "month", "year"})
        date_shape = _has_date_shape(series)
        if (date_hint and date_score >= 0.8) or (date_shape and date_score >= 0.95):
            confidence = min(0.98, 0.72 + date_score * 0.24 + (0.04 if date_hint else 0))
            semantics.append(ColumnSemantic(name, "datetime", confidence, "Date/time", "Most sampled values parse as dates."))
            continue

        categorical_hint = bool(
            name_tokens & {"category", "class", "group", "segment", "status", "tier", "type"}
        )
        if is_numeric_dtype(series) and not is_bool_dtype(series) and categorical_hint and unique <= 50:
            semantics.append(
                ColumnSemantic(
                    name,
                    "categorical",
                    0.86,
                    "Category",
                    "The column name indicates a coded category with a limited set of values.",
                )
            )
            continue

        if is_numeric_dtype(series) and not is_bool_dtype(series):
            display_format, reason = _numeric_format(series, lowered)
            semantics.append(ColumnSemantic(name, "numeric", 0.94, display_format, reason))
            continue

        if is_bool_dtype(series):
            semantics.append(ColumnSemantic(name, "categorical", 0.99, "Category", "Boolean values form two categories."))
            continue

        if is_string_dtype(series) or is_object_dtype(series):
            if 1 < unique <= 50 and unique_ratio <= 0.6:
                semantics.append(ColumnSemantic(name, "categorical", 0.88, "Category", "The column has a limited set of repeated values."))
            else:
                semantics.append(ColumnSemantic(name, "text", 0.82, "Text", "The values are mostly free-form or high-cardinality text."))
            continue

        semantics.append(ColumnSemantic(name, "ignore", 0.55, "Unknown", "ODS could not safely assign an analytical role."))

    return tuple(semantics)


def infer_column_roles(df: pd.DataFrame) -> ColumnRoles:
    return roles_from_mapping(df, {item.column: item.role for item in infer_column_semantics(df)})


def roles_from_mapping(df: pd.DataFrame, role_mapping: Mapping[str, str]) -> ColumnRoles:
    """Build validated chart roles from user-reviewed role selections."""
    buckets: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "identifier": [],
        "text": [],
        "ignore": [],
    }
    for raw_name in df.columns:
        name = str(raw_name)
        role = role_mapping.get(name, "ignore")
        if role not in buckets:
            role = "ignore"
        buckets[role].append(name)
    return ColumnRoles(
        numeric=tuple(buckets["numeric"]),
        categorical=tuple(buckets["categorical"]),
        datetime=tuple(buckets["datetime"]),
        identifiers=tuple(buckets["identifier"]),
        text=tuple(buckets["text"]),
        ignored=tuple(buckets["ignore"]),
    )


def suggest_dashboard(
    df: pd.DataFrame,
    max_charts: int = 4,
    *,
    roles: ColumnRoles | None = None,
    intent: str = "Overview",
    aggregation: Aggregation = "mean",
    date_grain: DateGrain = "month",
) -> tuple[ChartSuggestion, ...]:
    """Recommend charts based on reviewed roles and the user's analysis intent."""
    if df.empty or max_charts <= 0:
        return ()
    roles = roles or infer_column_roles(df)
    suggestions: list[ChartSuggestion] = []
    has_missing = bool(df.isna().any().any())

    def add_missingness() -> None:
        if has_missing:
            suggestions.append(
                ChartSuggestion(
                    "missingness",
                    "Missing values by column",
                    "Percent of rows missing in each affected column; no imputation is applied.",
                    "column",
                    "missing_percent",
                    confidence=1.0,
                )
            )

    def add_category_counts(limit: int = 1) -> None:
        for category in roles.categorical[:limit]:
            suggestions.append(
                ChartSuggestion(
                    "category_count",
                    f"Record count by {category}",
                    f"Exact row count grouped by {category}; missing categories are labeled explicitly. "
                    "Up to the 15 most common groups are shown.",
                    category,
                    "count",
                    confidence=0.95,
                )
            )

    def add_distributions(limit: int = 1) -> None:
        for measure in roles.numeric[:limit]:
            suggestions.append(
                ChartSuggestion(
                    "histogram",
                    f"Distribution of {measure}",
                    "Non-null numeric values grouped with a data-driven bin width; no aggregation is applied.",
                    measure,
                    "count",
                    confidence=0.92,
                )
            )

    def add_trends(limit: int = 1) -> None:
        if not roles.datetime:
            return
        date = roles.datetime[0]
        if roles.numeric:
            for measure in roles.numeric[:limit]:
                label = _aggregation_label(aggregation)
                suggestions.append(
                    ChartSuggestion(
                        "time_series",
                        f"{label} {measure} by {date_grain}",
                        f"{label} of non-null {measure}, grouped by {date} at {date_grain} grain.",
                        date,
                        measure,
                        aggregation,
                        date_grain,
                        0.88,
                    )
                )
        else:
            suggestions.append(
                ChartSuggestion(
                    "time_series",
                    f"Record count by {date_grain}",
                    f"Exact row count grouped by {date} at {date_grain} grain.",
                    date,
                    None,
                    "count",
                    date_grain,
                    0.92,
                )
            )

    def add_category_aggregates(limit: int = 2) -> None:
        if not (roles.categorical and roles.numeric):
            return
        category = roles.categorical[0]
        for measure in roles.numeric[:limit]:
            label = _aggregation_label(aggregation)
            suggestions.append(
                ChartSuggestion(
                    "category_aggregate",
                    f"{label} {measure} by {category}",
                    f"{label} of non-null {measure}, grouped by {category}; missing categories remain visible. "
                    "Up to the top 15 groups by result are shown.",
                    category,
                    measure,
                    aggregation,
                    confidence=0.9,
                )
            )

    def add_scatterplots(limit: int = 2) -> None:
        numeric = roles.numeric
        for index in range(min(max(len(numeric) - 1, 0), limit)):
            x, y = numeric[index], numeric[index + 1]
            suggestions.append(
                ChartSuggestion(
                    "scatter",
                    f"{y} versus {x}",
                    f"A deterministic sample of up to 5,000 complete rows using {x} and {y}; "
                    "no aggregation is applied.",
                    x,
                    y,
                    confidence=0.85,
                )
            )

    if intent == "Trends over time":
        add_trends(3)
    elif intent == "Compare categories":
        add_category_aggregates(3)
        add_category_counts(1)
    elif intent == "Understand distributions":
        add_distributions(4)
    elif intent == "Explore relationships":
        add_scatterplots(3)
        add_category_aggregates(1)
    elif intent == "Review data quality":
        add_missingness()
        add_category_counts(2)
    else:
        add_missingness()
        add_category_counts(1)
        add_distributions(1)
        add_trends(1)
        if len(suggestions) < max_charts:
            add_category_aggregates(1)
        if len(suggestions) < max_charts:
            add_scatterplots(1)

    return tuple(suggestions[:max_charts])


def build_chart_data(df: pd.DataFrame, suggestion: ChartSuggestion) -> pd.DataFrame:
    """Prepare the auditable summary DataFrame consumed by a Streamlit chart."""
    if suggestion.kind == "missingness":
        missing = (df.isna().mean() * 100).round(2)
        return missing[missing > 0].sort_values(ascending=False).rename("missing_percent").rename_axis("column").reset_index()

    if suggestion.kind == "category_count":
        values = df[suggestion.x].astype("string").fillna("(missing)")
        return values.value_counts(dropna=False).head(15).rename("count").rename_axis(suggestion.x).reset_index()

    if suggestion.kind == "histogram":
        values = pd.to_numeric(df[suggestion.x], errors="coerce").dropna()
        if values.empty:
            return pd.DataFrame(columns=[suggestion.x, "count"])
        bins = _histogram_bin_count(values)
        buckets = pd.cut(values, bins=bins, duplicates="drop")
        counts = buckets.value_counts(sort=False)
        return pd.DataFrame({suggestion.x: counts.index.astype(str), "count": counts.values})

    if suggestion.kind == "time_series":
        dates = _to_datetime_series(df[suggestion.x])
        grouped_dates = _group_dates(dates, suggestion.date_grain)
        if suggestion.y:
            values = pd.to_numeric(df[suggestion.y], errors="coerce")
            prepared = pd.DataFrame({suggestion.x: grouped_dates, suggestion.y: values}).dropna()
            return _aggregate(prepared, suggestion.x, suggestion.y, suggestion.aggregation)
        prepared = pd.DataFrame({suggestion.x: grouped_dates}).dropna()
        return prepared.value_counts().rename("count").reset_index().sort_values(suggestion.x)

    if suggestion.kind == "category_aggregate" and suggestion.y:
        prepared = pd.DataFrame(
            {
                suggestion.x: df[suggestion.x].astype("string").fillna("(missing)"),
                suggestion.y: pd.to_numeric(df[suggestion.y], errors="coerce"),
            }
        ).dropna(subset=[suggestion.y])
        return _aggregate(prepared, suggestion.x, suggestion.y, suggestion.aggregation).sort_values(suggestion.y, ascending=False).head(15)

    if suggestion.kind == "scatter" and suggestion.y:
        complete = df[[suggestion.x, suggestion.y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(complete) > 5000:
            return complete.sample(n=5000, random_state=42).sort_index()
        return complete

    return pd.DataFrame()


def _datetime_parse_score(series: pd.Series) -> float:
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return 0.0
    sample = series.dropna().head(100)
    if sample.empty:
        return 0.0
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _name_tokens(lowered_name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", lowered_name) if token}


def _has_date_shape(series: pd.Series) -> bool:
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False
    sample = series.dropna().astype("string").str.strip().head(100)
    if sample.empty:
        return False
    date_pattern = (
        r"(?:^\d{4}[-/]\d{1,2}[-/]\d{1,2})"
        r"|(?:^\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    )
    return float(sample.str.contains(date_pattern, case=False, regex=True).mean()) >= 0.8


def _looks_like_numeric_year(series: pd.Series, lowered_name: str) -> bool:
    if "year" not in _name_tokens(lowered_name) or not is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return False
    return bool(values.between(1900, 2100).all() and (values % 1 == 0).all())


def _to_datetime_series(series: pd.Series) -> pd.Series:
    """Parse dates while treating plausible numeric years as calendar years."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if (
        not non_null.empty
        and len(non_null) == int(series.notna().sum())
        and bool(non_null.between(1900, 2100).all())
        and bool((non_null % 1 == 0).all())
    ):
        return pd.to_datetime(numeric.astype("Int64").astype("string"), errors="coerce", format="%Y")
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _numeric_format(series: pd.Series, lowered_name: str) -> tuple[str, str]:
    if any(token in lowered_name for token in ("price", "cost", "revenue", "sales", "spend", "amount", "income")):
        return "Currency", "The numeric column name indicates a monetary measure."
    if any(token in lowered_name for token in ("percent", "percentage", "pct", "rate", "ratio")):
        return "Percentage", "The numeric column name indicates a percentage or rate."
    values = pd.to_numeric(series, errors="coerce").dropna()
    if not values.empty and bool(values.between(0, 1).all()) and values.nunique() > 2:
        return "Percentage", "All sampled numeric values fall between zero and one."
    return "Number", "Pandas loaded this column as numeric."


def _histogram_bin_count(values: pd.Series) -> int:
    if values.nunique() <= 1:
        return 1
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return min(12, max(3, ceil(values.nunique() ** 0.5)))
    width = 2 * iqr * (len(values) ** (-1 / 3))
    if width <= 0:
        return 8
    count = ceil((float(values.max()) - float(values.min())) / width)
    return min(20, max(3, count))


def _group_dates(dates: pd.Series, grain: DateGrain) -> pd.Series:
    # W-SUN periods start on Monday, which matches the common business week.
    frequencies = {"day": "D", "week": "W-SUN", "month": "M", "quarter": "Q", "year": "Y"}
    return dates.dt.to_period(frequencies[grain]).dt.start_time


def _aggregate(data: pd.DataFrame, group: str, value: str, aggregation: Aggregation) -> pd.DataFrame:
    grouped = data.groupby(group, as_index=False)[value]
    operations = {
        "mean": grouped.mean,
        "median": grouped.median,
        "sum": grouped.sum,
        "count": grouped.count,
        "min": grouped.min,
        "max": grouped.max,
    }
    return operations[aggregation]()


def _aggregation_label(aggregation: Aggregation) -> str:
    return {
        "mean": "Average",
        "median": "Median",
        "sum": "Total",
        "count": "Count of",
        "min": "Minimum",
        "max": "Maximum",
    }[aggregation]
