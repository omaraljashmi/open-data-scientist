"""Core analysis utilities for Open Data Scientist."""

from .loader import DatasetLoadError, load_dataset
from .dashboard import ChartSuggestion, build_chart_data, infer_column_roles, suggest_dashboard
from .profiler import DatasetProfile, profile_dataset
from .reporting import build_markdown_report

__all__ = [
    "DatasetLoadError",
    "DatasetProfile",
    "ChartSuggestion",
    "build_markdown_report",
    "build_chart_data",
    "infer_column_roles",
    "load_dataset",
    "profile_dataset",
    "suggest_dashboard",
]
