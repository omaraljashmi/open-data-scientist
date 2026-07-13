"""Dataset profiling and explainable data-quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    title: str
    detail: str
    column: str | None = None


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    columns: int
    memory_bytes: int
    duplicate_rows: int
    health_score: int
    column_profile: pd.DataFrame
    numeric_summary: pd.DataFrame
    issues: tuple[QualityIssue, ...]

    @property
    def missing_cells(self) -> int:
        if self.column_profile.empty:
            return 0
        return int(self.column_profile["missing_count"].sum())


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Create a compact profile and a transparent quality score."""
    rows, columns = df.shape
    duplicate_rows = int(df.duplicated().sum()) if rows else 0
    column_profile = _profile_columns(df)
    issues = tuple(_find_quality_issues(df, column_profile, duplicate_rows))
    health_score = _calculate_health_score(df, column_profile, duplicate_rows)

    numeric = df.select_dtypes(include="number")
    numeric_summary = numeric.describe().T if not numeric.empty else pd.DataFrame()

    return DatasetProfile(
        rows=rows,
        columns=columns,
        memory_bytes=int(df.memory_usage(index=True, deep=True).sum()),
        duplicate_rows=duplicate_rows,
        health_score=health_score,
        column_profile=column_profile,
        numeric_summary=numeric_summary,
        issues=issues,
    )


def _profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row_count = len(df)
    for name in df.columns:
        series = df[name]
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        examples = [str(value)[:60] for value in non_null.unique()[:3]]
        rows.append(
            {
                "column": str(name),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percent": round((missing_count / row_count * 100) if row_count else 0, 2),
                "unique_count": int(series.nunique(dropna=True)),
                "examples": ", ".join(examples) if examples else "—",
            }
        )
    return pd.DataFrame(rows)


def _find_quality_issues(
    df: pd.DataFrame,
    columns: pd.DataFrame,
    duplicate_rows: int,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    row_count = len(df)

    if row_count == 0:
        return [QualityIssue("critical", "Empty dataset", "The file contains columns but no data rows.")]

    if duplicate_rows:
        percent = duplicate_rows / row_count * 100
        issues.append(
            QualityIssue(
                "warning" if percent < 10 else "critical",
                "Duplicate rows detected",
                f"{duplicate_rows:,} rows ({percent:.1f}%) are exact duplicates.",
            )
        )

    for _, info in columns.iterrows():
        name = str(info["column"])
        missing_percent = float(info["missing_percent"])
        unique_count = int(info["unique_count"])

        if missing_percent > 0:
            issues.append(
                QualityIssue(
                    "critical" if missing_percent >= 30 else "warning",
                    "High missingness" if missing_percent >= 30 else "Missing values",
                    f"{missing_percent:.1f}% of values are missing.",
                    name,
                )
            )
        if unique_count <= 1:
            issues.append(
                QualityIssue(
                    "warning",
                    "Constant or empty column",
                    "This column has one or fewer distinct non-null values.",
                    name,
                )
            )
        if row_count >= 20 and unique_count == row_count:
            issues.append(
                QualityIssue(
                    "info",
                    "Possible identifier",
                    "Every value is unique; this column may be an ID rather than a feature.",
                    name,
                )
            )

    return issues


def _calculate_health_score(
    df: pd.DataFrame,
    columns: pd.DataFrame,
    duplicate_rows: int,
) -> int:
    if df.empty or df.shape[1] == 0:
        return 0
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_cells = int(columns["missing_count"].sum())
    missing_penalty = min(55.0, missing_cells / total_cells * 100)
    duplicate_penalty = min(30.0, duplicate_rows / max(len(df), 1) * 100)
    constant_columns = int((columns["unique_count"] <= 1).sum())
    constant_penalty = min(15.0, constant_columns / max(df.shape[1], 1) * 20)
    return max(0, round(100 - missing_penalty - duplicate_penalty - constant_penalty))

