"""Safe visual-query construction and in-memory DuckDB execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Literal

import duckdb
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)


FilterOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "starts_with",
    "is_null",
    "is_not_null",
]
AggregateFunction = Literal[
    "count_rows",
    "count_values",
    "mean",
    "median",
    "sum",
    "min",
    "max",
]


@dataclass(frozen=True)
class FilterRule:
    """A single AND-combined filter selected in the visual builder."""

    column: str
    operator: FilterOperator
    value: Any = None


@dataclass(frozen=True)
class AggregateRule:
    """A summary calculation and its output label."""

    function: AggregateFunction
    column: str | None = None
    alias: str | None = None


@dataclass(frozen=True)
class QuerySpec:
    """Declarative input for a safe single-table query."""

    selected_columns: tuple[str, ...] = ()
    filters: tuple[FilterRule, ...] = ()
    group_by: tuple[str, ...] = ()
    aggregates: tuple[AggregateRule, ...] = ()
    sort_by: str | None = None
    sort_descending: bool = False
    limit: int = 100


@dataclass(frozen=True)
class BuiltQuery:
    """Parameterized SQL plus the readable SQL shown to the user."""

    sql: str
    parameters: tuple[Any, ...]
    display_sql: str


@dataclass(frozen=True)
class QueryResult:
    """The executed result and the exact query that produced it."""

    dataframe: pd.DataFrame
    query: BuiltQuery


class QueryBuilderError(ValueError):
    """Raised when a visual query specification is invalid."""


FILTER_SQL = {
    "eq": "=",
    "ne": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}
AGGREGATE_SQL = {
    "count_values": "COUNT",
    "mean": "AVG",
    "median": "MEDIAN",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
}


def build_query(
    dataframe: pd.DataFrame,
    spec: QuerySpec,
    *,
    table_name: str = "uploaded_data",
) -> BuiltQuery:
    """Validate a visual query and compile it to injection-safe DuckDB SQL."""
    columns = _normalized_columns(dataframe)
    available = set(columns)
    _validate_limit(spec.limit)

    selected_columns = _unique_tuple(spec.selected_columns, "selected column")
    group_by = _unique_tuple(spec.group_by, "group-by column")
    _require_columns(selected_columns, available)
    _require_columns(group_by, available)

    select_parts: list[str] = []
    output_names: list[str] = []
    if spec.aggregates:
        if selected_columns:
            raise QueryBuilderError("Summary queries cannot also select ungrouped detail columns.")
        for column in group_by:
            select_parts.append(_quote_identifier(column))
            output_names.append(column)
        for aggregate in spec.aggregates:
            expression, alias = _aggregate_expression(aggregate, available)
            if alias in output_names:
                raise QueryBuilderError(f"Duplicate output column: {alias}.")
            select_parts.append(f"{expression} AS {_quote_identifier(alias)}")
            output_names.append(alias)
    else:
        if group_by:
            raise QueryBuilderError("Group-by columns require at least one summary calculation.")
        detail_columns = selected_columns or tuple(columns)
        select_parts.extend(_quote_identifier(column) for column in detail_columns)
        output_names.extend(detail_columns)

    if not select_parts:
        raise QueryBuilderError("Choose at least one output column or summary calculation.")

    parameterized_filters: list[str] = []
    display_filters: list[str] = []
    parameters: list[Any] = []
    normalized_frame = _normalize_dataframe(dataframe)
    for rule in spec.filters:
        if rule.column not in available:
            raise QueryBuilderError(f"Unknown column: {rule.column}.")
        sql_filter, display_filter, values = _filter_expression(
            normalized_frame[rule.column], rule
        )
        parameterized_filters.append(sql_filter)
        display_filters.append(display_filter)
        parameters.extend(values)

    if spec.sort_by is not None and spec.sort_by not in output_names:
        raise QueryBuilderError(
            f"Sort column must be one of the query outputs: {spec.sort_by}."
        )

    select_clause = ",\n    ".join(select_parts)
    base = f"SELECT\n    {select_clause}\nFROM {_quote_identifier(table_name)}"
    display_base = base
    if parameterized_filters:
        base += "\nWHERE " + "\n  AND ".join(parameterized_filters)
        display_base += "\nWHERE " + "\n  AND ".join(display_filters)
    if group_by:
        group_clause = ", ".join(_quote_identifier(column) for column in group_by)
        base += f"\nGROUP BY {group_clause}"
        display_base += f"\nGROUP BY {group_clause}"
    if spec.sort_by is not None:
        direction = "DESC" if spec.sort_descending else "ASC"
        order_clause = f"\nORDER BY {_quote_identifier(spec.sort_by)} {direction}"
        base += order_clause
        display_base += order_clause
    base += f"\nLIMIT {spec.limit}"
    display_base += f"\nLIMIT {spec.limit}"

    return BuiltQuery(
        sql=base,
        parameters=tuple(parameters),
        display_sql=display_base + ";",
    )


def execute_query(dataframe: pd.DataFrame, spec: QuerySpec) -> QueryResult:
    """Run a validated query against a temporary in-memory DuckDB relation."""
    normalized_frame = _normalize_dataframe(dataframe)
    query = build_query(normalized_frame, spec)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("uploaded_data", normalized_frame)
        result = connection.execute(query.sql, list(query.parameters)).fetchdf()
    except duckdb.Error as exc:
        raise QueryBuilderError(f"DuckDB could not run this query: {exc}") from exc
    finally:
        connection.close()
    return QueryResult(dataframe=result, query=query)


def _normalized_columns(dataframe: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(str(column) for column in dataframe.columns)
    if len(columns) != len(set(columns)):
        raise QueryBuilderError(
            "Column names must be unique after converting them to text before a query can run."
        )
    return columns


def _normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = _normalized_columns(dataframe)
    normalized = dataframe.copy(deep=False)
    normalized.columns = list(columns)
    return normalized


def _unique_tuple(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise QueryBuilderError(f"Choose each {label} only once.")
    return normalized


def _require_columns(columns: tuple[str, ...], available: set[str]) -> None:
    for column in columns:
        if column not in available:
            raise QueryBuilderError(f"Unknown column: {column}.")


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, Integral) or not 1 <= int(limit) <= 5000:
        raise QueryBuilderError("The row limit must be a whole number from 1 to 5,000.")


def _aggregate_expression(
    rule: AggregateRule,
    available: set[str],
) -> tuple[str, str]:
    if rule.function == "count_rows":
        if rule.column is not None:
            raise QueryBuilderError("Count rows does not use a source column.")
        default_alias = "row_count"
        expression = "COUNT(*)"
    else:
        if rule.function not in AGGREGATE_SQL:
            raise QueryBuilderError(f"Unsupported summary calculation: {rule.function}.")
        if not rule.column or rule.column not in available:
            raise QueryBuilderError("Choose a valid column for this summary calculation.")
        default_alias = f"{rule.function}_{rule.column}"
        expression = f"{AGGREGATE_SQL[rule.function]}({_quote_identifier(rule.column)})"
    alias = (rule.alias or default_alias).strip()
    if not alias:
        raise QueryBuilderError("Summary output labels cannot be empty.")
    return expression, alias


def _filter_expression(
    series: pd.Series,
    rule: FilterRule,
) -> tuple[str, str, tuple[Any, ...]]:
    column = _quote_identifier(rule.column)
    if rule.operator == "is_null":
        return f"{column} IS NULL", f"{column} IS NULL", ()
    if rule.operator == "is_not_null":
        return f"{column} IS NOT NULL", f"{column} IS NOT NULL", ()

    if rule.operator in {"contains", "starts_with"}:
        value = str(rule.value if rule.value is not None else "")
        function = "contains" if rule.operator == "contains" else "starts_with"
        expression = f"{function}(lower(CAST({column} AS VARCHAR)), lower(?))"
        display = (
            f"{function}(lower(CAST({column} AS VARCHAR)), "
            f"lower({_sql_literal(value)}))"
        )
        return expression, display, (value,)

    if rule.operator not in FILTER_SQL:
        raise QueryBuilderError(f"Unsupported filter operator: {rule.operator}.")
    value = _coerce_filter_value(series, rule.value)
    operator = FILTER_SQL[rule.operator]
    return (
        f"{column} {operator} ?",
        f"{column} {operator} {_sql_literal(value)}",
        (value,),
    )


def _coerce_filter_value(series: pd.Series, value: Any) -> Any:
    if value is None:
        raise QueryBuilderError("Choose a filter value or use an empty/missing operator.")
    if is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise QueryBuilderError(f"{value!r} is not a valid true/false value.")
    if is_datetime64_any_dtype(series):
        try:
            parsed = pd.to_datetime(value, errors="raise")
        except (TypeError, ValueError) as exc:
            raise QueryBuilderError(f"{value!r} is not a valid date or time.") from exc
        return parsed.to_pydatetime()
    if is_numeric_dtype(series):
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        try:
            if is_integer_dtype(series) and not is_float_dtype(series):
                number = float(value)
                if not number.is_integer():
                    raise ValueError
                return int(number)
            return float(value)
        except (TypeError, ValueError) as exc:
            raise QueryBuilderError(f"{value!r} is not a valid number.") from exc
    return value


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    if value is None or value is pd.NA:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (Integral, Real)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return "'" + value.isoformat().replace("'", "''") + "'"
    return "'" + str(value).replace("'", "''") + "'"
