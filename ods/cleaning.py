"""Deterministic, review-first data-cleaning recommendations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import re
from typing import Literal, Sequence

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from .dashboard import infer_column_semantics
from .profiler import profile_dataset


CleaningKind = Literal[
    "drop_duplicates",
    "trim_whitespace",
    "blank_to_missing",
    "normalize_categories",
    "convert_numeric",
    "convert_datetime",
    "fill_numeric_median",
    "fill_category_missing",
    "drop_column",
    "flag_outliers",
]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CleaningAction:
    """One inspectable cleaning operation that can be previewed and applied."""

    action_id: str
    kind: CleaningKind
    title: str
    evidence: str
    recommendation: str
    affected_rows: int
    affected_percent: float
    confidence: Confidence
    column: str | None = None
    parameters: tuple[tuple[str, str], ...] = ()

    @property
    def parameter_map(self) -> dict[str, str]:
        return dict(self.parameters)

    def as_recipe_step(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "operation": self.kind,
            "column": self.column,
            "title": self.title,
            "confidence": self.confidence,
            "estimated_affected_rows": self.affected_rows,
            "estimated_affected_percent": self.affected_percent,
            "parameters": self.parameter_map,
            "evidence": self.evidence,
        }


class CleaningError(ValueError):
    """Raised when a cleaning action cannot be applied safely."""


ACTION_PRIORITY: dict[str, int] = {
    "drop_duplicates": 10,
    "trim_whitespace": 20,
    "blank_to_missing": 30,
    "normalize_categories": 40,
    "convert_numeric": 50,
    "convert_datetime": 50,
    "fill_numeric_median": 60,
    "fill_category_missing": 60,
    "flag_outliers": 70,
    "drop_column": 80,
}

PLAIN_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
LEADING_ZERO_PATTERN = re.compile(r"^[+-]?0\d+$")
DATE_SHAPE_PATTERN = re.compile(
    r"(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2}(?:,?\s+\d{2,4})?)"
)
IDENTIFIER_TOKENS = {"id", "uuid", "guid", "key", "code", "zip", "postal", "phone"}


def suggest_cleaning_actions(dataframe: pd.DataFrame) -> tuple[CleaningAction, ...]:
    """Recommend conservative operations using transparent local rules."""
    frame = _normalized_frame(dataframe, deep=False)
    if frame.empty or frame.shape[1] == 0:
        return ()

    row_count = len(frame)
    suggestions: list[CleaningAction] = []
    duplicate_count = int(frame.duplicated(keep="first").sum())
    if duplicate_count:
        suggestions.append(
            _action(
                "drop_duplicates",
                None,
                "Remove exact duplicate rows",
                (
                    f"{duplicate_count:,} row{'s are' if duplicate_count != 1 else ' is'} an exact "
                    "duplicate of an earlier row across every column."
                ),
                "Keep the first copy and remove only later exact duplicates.",
                duplicate_count,
                row_count,
                "high",
            )
        )

    semantics = {item.column: item for item in infer_column_semantics(frame)}
    for column in frame.columns:
        series = frame[column]
        semantic = semantics[column]
        non_null = series.dropna()

        if _is_string_series(series):
            strings = non_null[non_null.map(lambda value: isinstance(value, str))]
            if len(strings) == len(non_null):
                suggestions.extend(
                    _string_actions(frame, column, strings, semantic.role)
                )

        missing_count = int(series.isna().sum())
        missing_rate = missing_count / row_count
        unique_count = int(non_null.nunique(dropna=True)) if not non_null.empty else 0

        if unique_count == 0:
            suggestions.append(
                _action(
                    "drop_column",
                    column,
                    f"Remove empty column · {column}",
                    "Every row is missing in this column, so it cannot support analysis.",
                    "Drop the column after confirming it is not required by a downstream schema.",
                    row_count,
                    row_count,
                    "high",
                    (("reason", "empty"),),
                )
            )
            continue

        if unique_count == 1:
            suggestions.append(
                _action(
                    "drop_column",
                    column,
                    f"Review constant column · {column}",
                    "The column has only one distinct non-missing value.",
                    "Remove it only if the constant does not carry required metadata.",
                    len(frame),
                    row_count,
                    "low",
                    (("reason", "constant"),),
                )
            )
            continue

        if missing_count:
            suggestions.extend(
                _missing_value_actions(
                    frame,
                    column,
                    series,
                    semantic.role,
                    missing_count,
                    missing_rate,
                )
            )

        if (
            is_numeric_dtype(series)
            and not is_bool_dtype(series)
            and semantic.role != "identifier"
        ):
            outlier_action = _outlier_action(frame, column, series)
            if outlier_action is not None:
                suggestions.append(outlier_action)

    return tuple(
        sorted(
            suggestions,
            key=lambda item: (
                ACTION_PRIORITY[item.kind],
                item.column or "",
                item.action_id,
            ),
        )
    )


def apply_cleaning_actions(
    dataframe: pd.DataFrame,
    actions: Sequence[CleaningAction],
) -> pd.DataFrame:
    """Apply an explicit action selection in a stable, dependency-safe order."""
    result = _normalized_frame(dataframe, deep=True)
    if len({action.action_id for action in actions}) != len(actions):
        raise CleaningError("Each cleaning action can be applied only once per batch.")

    dropped_columns = {
        action.column for action in actions if action.kind == "drop_column"
    }
    conflicting_columns = {
        action.column
        for action in actions
        if action.column in dropped_columns and action.kind != "drop_column"
    }
    if conflicting_columns:
        names = ", ".join(sorted(str(column) for column in conflicting_columns))
        raise CleaningError(
            f"Choose either dropping or cleaning these columns in one batch, not both: {names}."
        )

    ordered = sorted(
        actions,
        key=lambda item: (ACTION_PRIORITY[item.kind], item.column or "", item.action_id),
    )
    for action in ordered:
        result = _apply_action(result, action)
    return result


def replay_cleaning_batches(
    original: pd.DataFrame,
    batches: Sequence[Sequence[CleaningAction]],
) -> pd.DataFrame:
    """Rebuild a cleaned dataset from the original and its auditable action history."""
    result = _normalized_frame(original, deep=True)
    for batch in batches:
        result = apply_cleaning_actions(result, batch)
    return result


def build_cleaning_recipe(
    file_name: str,
    source_sha256: str,
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    batches: Sequence[Sequence[CleaningAction]],
) -> str:
    """Serialize an inspectable recipe with operations and before/after evidence."""
    original_profile = profile_dataset(original)
    cleaned_profile = profile_dataset(cleaned)
    operations = [
        action.as_recipe_step()
        for batch in batches
        for action in sorted(
            batch,
            key=lambda item: (
                ACTION_PRIORITY[item.kind],
                item.column or "",
                item.action_id,
            ),
        )
    ]
    recipe = {
        "format": "open-data-scientist-cleaning-recipe",
        "version": 1,
        "source": {
            "file_name": file_name,
            "sha256": source_sha256,
            "rows": original_profile.rows,
            "columns": original_profile.columns,
            "quality_score": original_profile.health_score,
        },
        "result": {
            "rows": cleaned_profile.rows,
            "columns": cleaned_profile.columns,
            "quality_score": cleaned_profile.health_score,
            "column_types": {
                str(column): str(dtype)
                for column, dtype in cleaned.dtypes.items()
            },
        },
        "operations": operations,
        "notes": [
            "Operations are applied in the listed order.",
            "Review assumptions against domain rules before using cleaned data for decisions.",
            "Outlier actions add evidence flags; they do not delete or cap source values.",
        ],
    }
    return json.dumps(recipe, indent=2, ensure_ascii=False) + "\n"


def cleaning_actions_from_recipe(text: str) -> tuple[tuple[CleaningAction, ...], ...]:
    """Rebuild replayable cleaning batches from an exported recipe.

    The recipe's flat operation list is replayed as one single-action batch
    per step, preserving the exact recorded order. Recommendation text is not
    stored in recipes, so replays carry an empty recommendation.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CleaningError("This is not a valid cleaning recipe JSON file.") from exc
    if not isinstance(payload, dict):
        raise CleaningError("A cleaning recipe must contain one JSON object.")
    if (
        payload.get("format") != "open-data-scientist-cleaning-recipe"
        or payload.get("version") != 1
    ):
        raise CleaningError("Only cleaning recipe format version 1 is supported.")
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise CleaningError("The cleaning recipe operations must be a JSON array.")

    batches: list[tuple[CleaningAction, ...]] = []
    for index, step in enumerate(operations, start=1):
        if not isinstance(step, dict):
            raise CleaningError(f"Recipe operation {index} must be an object.")
        kind = step.get("operation")
        if kind not in ACTION_PRIORITY:
            raise CleaningError(f"Recipe operation {index} has unsupported type: {kind!r}.")
        confidence = step.get("confidence")
        if confidence not in {"high", "medium", "low"}:
            raise CleaningError(f"Recipe operation {index} has an invalid confidence level.")
        parameters = step.get("parameters", {})
        if not isinstance(parameters, dict):
            raise CleaningError(f"Recipe operation {index} has invalid parameters.")
        column = step.get("column")
        if column is not None and not isinstance(column, str):
            raise CleaningError(f"Recipe operation {index} has an invalid column name.")
        batches.append(
            (
                CleaningAction(
                    action_id=str(step.get("action_id") or f"{kind}:{column or 'dataset'}"),
                    kind=kind,
                    title=str(step.get("title") or kind),
                    evidence=str(step.get("evidence") or ""),
                    recommendation="",
                    affected_rows=int(step.get("estimated_affected_rows") or 0),
                    affected_percent=float(step.get("estimated_affected_percent") or 0.0),
                    confidence=confidence,
                    column=column,
                    parameters=tuple(
                        (str(key), str(value)) for key, value in sorted(parameters.items())
                    ),
                ),
            )
        )
    return tuple(batches)


def _string_actions(
    frame: pd.DataFrame,
    column: str,
    strings: pd.Series,
    semantic_role: str,
) -> list[CleaningAction]:
    actions: list[CleaningAction] = []
    row_count = len(frame)
    trimmed = strings.map(str.strip)
    whitespace_count = int(strings.ne(trimmed).sum())
    if whitespace_count:
        actions.append(
            _action(
                "trim_whitespace",
                column,
                f"Trim surrounding whitespace · {column}",
                f"{whitespace_count:,} value{'s have' if whitespace_count != 1 else ' has'} leading or trailing whitespace.",
                "Remove only whitespace at the beginning and end of text values.",
                whitespace_count,
                row_count,
                "high",
            )
        )

    blank_count = int(trimmed.eq("").sum())
    if blank_count:
        actions.append(
            _action(
                "blank_to_missing",
                column,
                f"Standardize blank values · {column}",
                f"{blank_count:,} value{'s are' if blank_count != 1 else ' is'} empty or whitespace-only text.",
                "Represent blank text as a true missing value so quality checks count it consistently.",
                blank_count,
                row_count,
                "high",
            )
        )

    meaningful = trimmed[trimmed.ne("")]
    if meaningful.empty:
        return actions

    category_action = _category_action(
        frame,
        column,
        strings,
        meaningful,
        semantic_role,
    )
    if category_action is not None:
        actions.append(category_action)

    numeric_action = _numeric_conversion_action(
        frame,
        column,
        meaningful,
        semantic_role,
    )
    if numeric_action is not None:
        actions.append(numeric_action)
    else:
        datetime_action = _datetime_conversion_action(
            frame,
            column,
            meaningful,
            semantic_role,
        )
        if datetime_action is not None:
            actions.append(datetime_action)
    return actions


def _category_action(
    frame: pd.DataFrame,
    column: str,
    original_strings: pd.Series,
    meaningful: pd.Series,
    _semantic_role: str,
) -> CleaningAction | None:
    unique_count = int(meaningful.nunique())
    if unique_count > 50 or unique_count > max(10, round(len(frame) * 0.4)):
        return None

    counts = meaningful.value_counts()
    groups: dict[str, list[tuple[str, int]]] = {}
    for value, count in counts.items():
        groups.setdefault(str(value).casefold(), []).append((str(value), int(count)))
    collisions = {key: values for key, values in groups.items() if len(values) > 1}
    if not collisions:
        return None

    canonical = {
        key: sorted(values, key=lambda item: (-item[1], item[0]))[0][0]
        for key, values in collisions.items()
    }

    def normalized(value: object) -> object:
        if not isinstance(value, str):
            return value
        return canonical.get(value.strip().casefold(), value)

    affected = int(original_strings.map(normalized).ne(original_strings).sum())
    variant_count = sum(len(values) for values in collisions.values())
    return _action(
        "normalize_categories",
        column,
        f"Merge inconsistent category labels · {column}",
        (
            f"{variant_count:,} spellings collapse into {len(collisions):,} case-insensitive "
            f"group{'s' if len(collisions) != 1 else ''}, affecting {affected:,} rows."
        ),
        "Use the most frequent trimmed spelling in each matching group; unrelated labels stay unchanged.",
        affected,
        len(frame),
        "medium",
        (("canonical_map", json.dumps(canonical, ensure_ascii=False, sort_keys=True)),),
    )


def _numeric_conversion_action(
    frame: pd.DataFrame,
    column: str,
    meaningful: pd.Series,
    semantic_role: str,
) -> CleaningAction | None:
    values = meaningful.astype(str)
    if len(values) < 3 or semantic_role == "identifier" or _name_suggests_identifier(column):
        return None
    if semantic_role == "categorical" and int(values.nunique()) <= 10:
        return None
    if not values.map(lambda value: bool(PLAIN_NUMBER_PATTERN.fullmatch(value))).all():
        return None
    if values.map(lambda value: bool(LEADING_ZERO_PATTERN.fullmatch(value))).any():
        return None
    parsed = pd.to_numeric(values, errors="coerce")
    if not parsed.notna().all():
        return None
    return _action(
        "convert_numeric",
        column,
        f"Convert numeric text · {column}",
        f"All {len(values):,} non-blank values follow a plain numeric format and parse successfully.",
        "Convert the column to a numeric data type; identifier-like names and leading-zero values are excluded.",
        len(values),
        len(frame),
        "high",
    )


def _datetime_conversion_action(
    frame: pd.DataFrame,
    column: str,
    meaningful: pd.Series,
    semantic_role: str,
) -> CleaningAction | None:
    values = meaningful.astype(str)
    if len(values) < 3 or semantic_role != "datetime":
        return None
    shape_rate = float(values.map(lambda value: bool(DATE_SHAPE_PATTERN.search(value))).mean())
    if shape_rate < 0.9:
        return None
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if not parsed.notna().all():
        return None
    return _action(
        "convert_datetime",
        column,
        f"Convert date text · {column}",
        f"All {len(values):,} non-blank values parse as dates and at least 90% have a recognizable date shape.",
        "Convert the column to pandas datetime values so sorting and time analysis use calendar order.",
        len(values),
        len(frame),
        "high",
    )


def _missing_value_actions(
    frame: pd.DataFrame,
    column: str,
    series: pd.Series,
    semantic_role: str,
    missing_count: int,
    missing_rate: float,
) -> list[CleaningAction]:
    actions: list[CleaningAction] = []
    if missing_rate >= 0.6:
        actions.append(
            _action(
                "drop_column",
                column,
                f"Review mostly-missing column · {column}",
                f"{missing_count:,} rows ({missing_rate:.1%}) are missing.",
                "Consider removing the column only after confirming its sparse values are not analytically important.",
                len(frame),
                len(frame),
                "low",
                (("reason", "mostly_missing"),),
            )
        )
        return actions

    if missing_rate > 0.3:
        return actions

    if is_numeric_dtype(series) and not is_bool_dtype(series) and semantic_role != "identifier":
        non_null = pd.to_numeric(series, errors="coerce").dropna()
        finite = non_null[non_null.map(lambda value: isfinite(float(value)))]
        if len(finite) >= 3:
            median = float(finite.median())
            actions.append(
                _action(
                    "fill_numeric_median",
                    column,
                    f"Fill missing numbers with the median · {column}",
                    f"{missing_count:,} rows ({missing_rate:.1%}) are missing; the observed median is {median:g}.",
                    "Use the median as a robust placeholder, but confirm that imputation is valid for this measure.",
                    missing_count,
                    len(frame),
                    "medium",
                    (("value", repr(median)),),
                )
            )
    elif semantic_role == "categorical":
        label = _missing_label(series)
        actions.append(
            _action(
                "fill_category_missing",
                column,
                f"Label missing categories · {column}",
                f"{missing_count:,} rows ({missing_rate:.1%}) have no category.",
                f"Use {label!r} as an explicit category instead of guessing an existing label.",
                missing_count,
                len(frame),
                "medium",
                (("value", label),),
            )
        )
    return actions


def _outlier_action(
    frame: pd.DataFrame,
    column: str,
    series: pd.Series,
) -> CleaningAction | None:
    flag_column = f"{column}_is_outlier"
    if flag_column in {str(existing) for existing in frame.columns}:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    finite_values = values[values.map(lambda value: isfinite(float(value)))]
    if len(finite_values) < 8 or finite_values.nunique() < 4:
        return None
    q1 = float(finite_values.quantile(0.25))
    q3 = float(finite_values.quantile(0.75))
    iqr = q3 - q1
    if not pd.notna(iqr) or iqr <= 0:
        return None
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = (values < lower) | (values > upper)
    count = int(outliers.sum())
    if not count:
        return None
    return _action(
        "flag_outliers",
        column,
        f"Flag possible outliers · {column}",
        f"{count:,} values fall outside the robust IQR range [{lower:g}, {upper:g}].",
        f"Add {flag_column!r} as a Boolean review flag; source values will not be removed or capped.",
        count,
        len(frame),
        "medium",
        (
            ("lower", repr(lower)),
            ("upper", repr(upper)),
            ("flag_column", flag_column),
        ),
    )


def _apply_action(frame: pd.DataFrame, action: CleaningAction) -> pd.DataFrame:
    result = frame.copy(deep=True)
    parameters = action.parameter_map
    if action.kind == "drop_duplicates":
        return result.drop_duplicates(keep="first").reset_index(drop=True)

    column = action.column
    if column is None or column not in result.columns:
        raise CleaningError(
            f"Cleaning action {action.action_id!r} references a column that is no longer available."
        )

    if action.kind == "trim_whitespace":
        result[column] = result[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    elif action.kind == "blank_to_missing":
        result[column] = result[column].map(
            lambda value: pd.NA
            if isinstance(value, str) and not value.strip()
            else value
        )
    elif action.kind == "normalize_categories":
        canonical = json.loads(parameters["canonical_map"])

        def normalize(value: object) -> object:
            if not isinstance(value, str):
                return value
            return canonical.get(value.strip().casefold(), value)

        result[column] = result[column].map(normalize)
    elif action.kind == "convert_numeric":
        cleaned = result[column].map(
            lambda value: pd.NA
            if isinstance(value, str) and not value.strip()
            else value.strip() if isinstance(value, str) else value
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        non_null = parsed.dropna()
        if not non_null.empty and bool(non_null.mod(1).eq(0).all()):
            result[column] = parsed.astype("Int64" if parsed.isna().any() else "int64")
        else:
            result[column] = parsed.astype("Float64")
    elif action.kind == "convert_datetime":
        cleaned = result[column].map(
            lambda value: pd.NA
            if isinstance(value, str) and not value.strip()
            else value.strip() if isinstance(value, str) else value
        )
        result[column] = pd.to_datetime(cleaned, errors="coerce", format="mixed")
    elif action.kind == "fill_numeric_median":
        result[column] = result[column].fillna(float(parameters["value"]))
    elif action.kind == "fill_category_missing":
        label = parameters["value"]
        series = result[column]
        if isinstance(series.dtype, pd.CategoricalDtype) and label not in series.cat.categories:
            series = series.cat.add_categories([label])
        result[column] = series.fillna(label)
    elif action.kind == "drop_column":
        result = result.drop(columns=[column])
    elif action.kind == "flag_outliers":
        lower = float(parameters["lower"])
        upper = float(parameters["upper"])
        numeric = pd.to_numeric(result[column], errors="coerce")
        result[parameters["flag_column"]] = ((numeric < lower) | (numeric > upper)).fillna(False)
    else:  # pragma: no cover - Literal typing and tests guard this boundary.
        raise CleaningError(f"Unsupported cleaning action: {action.kind}")
    return result


def _action(
    kind: CleaningKind,
    column: str | None,
    title: str,
    evidence: str,
    recommendation: str,
    affected_rows: int,
    row_count: int,
    confidence: Confidence,
    parameters: tuple[tuple[str, str], ...] = (),
) -> CleaningAction:
    location = column if column is not None else "dataset"
    return CleaningAction(
        action_id=f"{kind}:{location}",
        kind=kind,
        title=title,
        evidence=evidence,
        recommendation=recommendation,
        affected_rows=int(affected_rows),
        affected_percent=round(affected_rows / max(row_count, 1) * 100, 2),
        confidence=confidence,
        column=column,
        parameters=parameters,
    )


def _normalized_frame(dataframe: pd.DataFrame, *, deep: bool) -> pd.DataFrame:
    columns = [str(column) for column in dataframe.columns]
    if len(columns) != len(set(columns)):
        raise CleaningError("Data Cleaning Studio needs unique column names.")
    frame = dataframe.copy(deep=deep)
    frame.columns = columns
    return frame


def _is_string_series(series: pd.Series) -> bool:
    return bool(is_object_dtype(series) or is_string_dtype(series))


def _name_suggests_identifier(column: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", column.casefold()))
    return bool(tokens & IDENTIFIER_TOKENS)


def _missing_label(series: pd.Series) -> str:
    existing = {str(value).casefold() for value in series.dropna().unique()}
    label = "Missing"
    suffix = 2
    while label.casefold() in existing:
        label = f"Missing ({suffix})"
        suffix += 1
    return label
