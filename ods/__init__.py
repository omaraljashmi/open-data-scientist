"""Core analysis utilities for Open Data Scientist."""

from .loader import DatasetLoadError, load_dataset
from .dashboard import (
    ChartSuggestion,
    ColumnRoles,
    ColumnSemantic,
    INTENTS,
    build_chart_data,
    infer_column_roles,
    infer_column_semantics,
    roles_from_mapping,
    suggest_dashboard,
)
from .profiler import DatasetProfile, profile_dataset
from .query_builder import (
    AggregateRule,
    BuiltQuery,
    FilterRule,
    QueryBuilderError,
    QueryResult,
    QuerySpec,
    build_query,
    execute_query,
)
from .reporting import build_markdown_report

__all__ = [
    "DatasetLoadError",
    "DatasetProfile",
    "AggregateRule",
    "BuiltQuery",
    "ChartSuggestion",
    "ColumnRoles",
    "ColumnSemantic",
    "INTENTS",
    "FilterRule",
    "QueryBuilderError",
    "QueryResult",
    "QuerySpec",
    "build_markdown_report",
    "build_chart_data",
    "build_query",
    "execute_query",
    "infer_column_roles",
    "infer_column_semantics",
    "load_dataset",
    "profile_dataset",
    "roles_from_mapping",
    "suggest_dashboard",
]
