"""Local, read-only SQL explanation and optimization guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


Severity = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class ClauseExplanation:
    """A plain-English description of one part of a SQL query."""

    clause: str
    explanation: str


@dataclass(frozen=True)
class OptimizationFinding:
    """An evidence-backed query risk or improvement opportunity."""

    rule_id: str
    severity: Severity
    category: str
    title: str
    detail: str
    recommendation: str


@dataclass(frozen=True)
class PlanStep:
    """A physical-plan operator translated into plain English."""

    operator: str
    explanation: str


@dataclass(frozen=True)
class SqlAnalysis:
    """Complete output of the local SQL Coach."""

    formatted_sql: str
    suggested_sql: str
    score: int
    clauses: tuple[ClauseExplanation, ...]
    findings: tuple[OptimizationFinding, ...]
    physical_plan: str
    plan_steps: tuple[PlanStep, ...]
    referenced_tables: tuple[str, ...]


class SqlCoachError(ValueError):
    """Raised when a query is unsafe, invalid, or cannot be planned."""


PLAN_OPERATOR_EXPLANATIONS = {
    "EMPTY_RESULT": "Returns no rows because the optimizer proved the conditions cannot match.",
    "TOP_N": "Keeps only the requested top rows while sorting, which can avoid a full sort.",
    "ORDER_BY": "Sorts the result according to the ORDER BY expressions.",
    "HASH_GROUP_BY": "Builds hash groups to calculate grouped summaries.",
    "PERFECT_HASH_GROUP_BY": "Uses a compact hash table for grouped summaries with suitable keys.",
    "UNGROUPED_AGGREGATE": "Calculates a summary for the full input without grouping.",
    "HASH_JOIN": "Matches rows by building a hash table for the join keys.",
    "NESTED_LOOP_JOIN": "Compares join inputs with a nested-loop strategy.",
    "CROSS_PRODUCT": "Combines every row from one input with every row from another.",
    "FILTER": "Keeps only rows that satisfy the filter condition.",
    "PROJECTION": "Produces the selected output columns or expressions.",
    "PANDAS_SCAN": "Reads the uploaded in-memory pandas DataFrame.",
    "TABLE_SCAN": "Reads rows from a registered table or relation.",
    "SEQ_SCAN": "Reads table rows sequentially.",
    "STREAMING_LIMIT": "Stops reading after the requested number of rows is produced.",
    "LIMIT": "Restricts how many rows continue through the plan.",
    "WINDOW": "Calculates a window function across related rows.",
    "UNION": "Combines the outputs of multiple query branches.",
}


BLOCKED_EXPRESSION_TYPES = (
    exp.Alter,
    exp.Command,
    exp.Commit,
    exp.Copy,
    exp.Create,
    exp.Delete,
    exp.Drop,
    exp.Execute,
    exp.Grant,
    exp.Insert,
    exp.Merge,
    exp.Revoke,
    exp.Rollback,
    exp.Set,
    exp.Transaction,
    exp.TruncateTable,
    exp.Update,
    exp.Use,
)


def analyze_query(dataframe: pd.DataFrame, sql: str) -> SqlAnalysis:
    """Explain and assess one read-only DuckDB query without executing it."""
    normalized_frame = _normalize_dataframe(dataframe)
    expression = _parse_read_only(sql)
    cte_names = {
        cte.alias_or_name.casefold()
        for cte in expression.find_all(exp.CTE)
        if cte.alias_or_name
    }
    referenced_tables = _validate_sources(expression, cte_names)

    formatted_sql = _render_sql(expression)
    suggested_sql = _render_sql(_expand_safe_stars(expression, normalized_frame.columns))
    clauses = _explain_clauses(expression)
    findings = _find_optimization_opportunities(expression, normalized_frame)
    physical_plan = _build_physical_plan(normalized_frame, expression)
    plan_steps = _summarize_plan(physical_plan)
    score = max(0, 100 - sum({"high": 25, "medium": 12, "low": 5}[item.severity] for item in findings))

    return SqlAnalysis(
        formatted_sql=formatted_sql,
        suggested_sql=suggested_sql,
        score=score,
        clauses=clauses,
        findings=findings,
        physical_plan=physical_plan,
        plan_steps=plan_steps,
        referenced_tables=referenced_tables,
    )


def _normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [str(column) for column in dataframe.columns]
    if len(columns) != len(set(columns)):
        raise SqlCoachError("SQL Coach needs unique column names in the uploaded dataset.")
    normalized = dataframe.copy(deep=False)
    normalized.columns = columns
    return normalized


def _parse_read_only(sql: str) -> exp.Query:
    text = sql.strip()
    if not text:
        raise SqlCoachError("Enter a SQL query to analyze.")
    if len(text) > 20_000:
        raise SqlCoachError("Keep the query under 20,000 characters for this local coach.")
    try:
        statements = [statement for statement in sqlglot.parse(text, read="duckdb") if statement]
    except ParseError as exc:
        raise SqlCoachError(f"DuckDB SQL syntax error: {_parse_error_message(exc)}") from exc
    if len(statements) != 1:
        raise SqlCoachError("Analyze exactly one SQL statement at a time.")

    expression = statements[0]
    if not isinstance(expression, exp.Query):
        raise SqlCoachError(
            "Only read-only SELECT, WITH, or UNION queries are allowed. "
            "Data-changing and database-management statements are blocked."
        )
    if any(isinstance(node, BLOCKED_EXPRESSION_TYPES) for node in expression.walk()):
        raise SqlCoachError(
            "Only read-only SELECT, WITH, or UNION queries are allowed. "
            "Data-changing and database-management statements are blocked, including inside CTEs."
        )
    if expression.find(exp.Into):
        raise SqlCoachError("SELECT INTO is blocked because it creates or changes database state.")
    if expression.find(exp.Placeholder):
        raise SqlCoachError("Replace query placeholders with literal values before analysis.")
    if sum(1 for _ in expression.walk()) > 2_000:
        raise SqlCoachError("This query is too complex for the interactive SQL Coach.")
    return expression


def _parse_error_message(error: ParseError) -> str:
    if error.errors:
        first = error.errors[0]
        description = first.get("description", "Invalid SQL")
        line = first.get("line")
        column = first.get("col")
        if line and column:
            return f"{description} at line {line}, column {column}."
        return f"{description}."
    return str(error)


def _validate_sources(expression: exp.Query, cte_names: set[str]) -> tuple[str, ...]:
    tables: list[str] = []
    for table in expression.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SqlCoachError(
                "External table functions and file readers are blocked. "
                "Use only the uploaded_data table."
            )
        if table.args.get("catalog") or table.args.get("db"):
            raise SqlCoachError("Catalog and schema references are blocked in the local SQL Coach.")
        name = table.name
        folded = name.casefold()
        if folded not in {"uploaded_data", *cte_names}:
            raise SqlCoachError(
                f"Unknown table {name!r}. This milestone can analyze only uploaded_data and its CTEs."
            )
        if folded not in cte_names and name not in tables:
            tables.append(name)
    return tuple(tables)


def _render_sql(expression: exp.Expression) -> str:
    return expression.sql(dialect="duckdb", pretty=True).rstrip(";") + ";"


def _expand_safe_stars(expression: exp.Query, columns: pd.Index) -> exp.Query:
    """Expand a top-level single-table star without changing query semantics."""
    if len(columns) > 50 or not isinstance(expression, exp.Select):
        return expression
    if expression.args.get("with_") or expression.args.get("joins"):
        return expression
    from_clause = expression.args.get("from_")
    if not from_clause or not isinstance(from_clause.this, exp.Table):
        return expression
    if from_clause.this.name.casefold() != "uploaded_data":
        return expression

    rewritten = expression.copy()
    new_projections: list[exp.Expression] = []
    changed = False
    for projection in rewritten.expressions:
        if isinstance(projection, exp.Star) and not any(projection.args.values()):
            new_projections.extend(exp.column(str(column), quoted=True) for column in columns)
            changed = True
        elif (
            isinstance(projection, exp.Column)
            and isinstance(projection.this, exp.Star)
            and not any(projection.this.args.values())
        ):
            new_projections.extend(
                exp.column(str(column), table=projection.table or None, quoted=True)
                for column in columns
            )
            changed = True
        else:
            new_projections.append(projection)
    if changed:
        rewritten.set("expressions", new_projections)
    return rewritten


def _explain_clauses(expression: exp.Query) -> tuple[ClauseExplanation, ...]:
    select = _primary_select(expression)
    clauses: list[ClauseExplanation] = []
    ctes = list(expression.find_all(exp.CTE))
    if ctes:
        names = ", ".join(cte.alias_or_name for cte in ctes if cte.alias_or_name)
        clauses.append(
            ClauseExplanation(
                "WITH",
                f"Builds {len(ctes)} named intermediate result{'s' if len(ctes) != 1 else ''}: {names}.",
            )
        )

    if select is not None:
        projections = select.expressions
        direct_star = any(_is_projection_star(item) for item in projections)
        if direct_star:
            output = "Returns every column produced by the input relation."
        else:
            names = [item.alias_or_name or _short_sql(item) for item in projections]
            output = f"Returns {len(names)} output expression{'s' if len(names) != 1 else ''}: {_short_list(names)}."
        if select.args.get("distinct"):
            output += " Duplicate output rows are removed."
        clauses.append(ClauseExplanation("SELECT", output))

        from_clause = select.args.get("from_")
        if from_clause:
            clauses.append(
                ClauseExplanation("FROM", f"Reads from {_short_sql(from_clause.this)}.")
            )
        joins = select.args.get("joins") or []
        if joins:
            kinds = [((join.side or join.kind or "inner").strip().upper()) for join in joins]
            clauses.append(
                ClauseExplanation(
                    "JOIN",
                    f"Combines {len(joins)} additional relation{'s' if len(joins) != 1 else ''} using {_short_list(kinds)} join logic.",
                )
            )
        where = select.args.get("where")
        if where:
            clauses.append(
                ClauseExplanation("WHERE", f"Keeps rows where {_short_sql(where.this)}.")
            )
        group = select.args.get("group")
        if group:
            groups = [_short_sql(item) for item in group.expressions]
            clauses.append(
                ClauseExplanation("GROUP BY", f"Creates groups using {_short_list(groups)}.")
            )
        having = select.args.get("having")
        if having:
            clauses.append(
                ClauseExplanation("HAVING", f"Keeps grouped results where {_short_sql(having.this)}.")
            )
        order = select.args.get("order")
        if order:
            order_items = [_short_sql(item) for item in order.expressions]
            clauses.append(
                ClauseExplanation("ORDER BY", f"Sorts the result by {_short_list(order_items)}.")
            )
        limit = select.args.get("limit")
        if limit:
            clauses.append(
                ClauseExplanation("LIMIT", f"Returns at most {_short_sql(limit.expression)} rows.")
            )
    if isinstance(expression, exp.SetOperation):
        clauses.append(
            ClauseExplanation(
                expression.key.upper(),
                "Combines the outputs of two query branches before returning the final result.",
            )
        )
    return tuple(clauses)


def _find_optimization_opportunities(
    expression: exp.Query,
    dataframe: pd.DataFrame,
) -> tuple[OptimizationFinding, ...]:
    findings: list[OptimizationFinding] = []
    seen: set[str] = set()

    def add(
        rule_id: str,
        severity: Severity,
        category: str,
        title: str,
        detail: str,
        recommendation: str,
    ) -> None:
        if rule_id not in seen:
            seen.add(rule_id)
            findings.append(
                OptimizationFinding(
                    rule_id,
                    severity,
                    category,
                    title,
                    detail,
                    recommendation,
                )
            )

    select = _primary_select(expression)
    if select is not None:
        direct_stars = [item for item in select.expressions if _is_projection_star(item)]
        if direct_stars:
            add(
                "explicit-columns",
                "low",
                "Readability",
                "SELECT * hides the output contract",
                "The query requests every available output column, so schema changes can silently change its result.",
                "List the columns the result actually needs. The clean rewrite expands the current upload when that is unambiguous.",
            )

        group = select.args.get("group")
        has_aggregate = any(item.find(exp.AggFunc) for item in select.expressions)
        if not select.args.get("limit") and not group and not has_aggregate and not select.args.get("distinct"):
            severity: Severity = "medium" if len(dataframe) > 10_000 else "low"
            add(
                "unbounded-detail",
                severity,
                "Performance",
                "Detail query has no row limit",
                f"The uploaded dataset contains {len(dataframe):,} rows, and this detail query can return all of them.",
                "Add a LIMIT for interactive exploration, or add a selective WHERE condition when the full result is unnecessary.",
            )
        if select.args.get("distinct"):
            add(
                "distinct-cost",
                "medium" if len(dataframe) > 10_000 else "low",
                "Performance",
                "DISTINCT requires duplicate elimination",
                "DuckDB must compare or group the projected rows to remove duplicates.",
                "Keep DISTINCT only when duplicate output rows are possible and removing them is required.",
            )
        if select.args.get("order") and not select.args.get("limit"):
            add(
                "unbounded-sort",
                "medium",
                "Performance",
                "Full result is sorted without a limit",
                "ORDER BY may sort the entire result even when only a small preview is needed.",
                "Add a LIMIT when you need only the first or last rows; DuckDB can often use a TOP_N plan.",
            )
        if len(select.expressions) > 15:
            add(
                "wide-projection",
                "low",
                "Maintainability",
                "The result has many output expressions",
                f"The SELECT list contains {len(select.expressions)} expressions, which makes the result harder to review and reuse.",
                "Return only the fields needed by the next step, or split unrelated outputs into focused queries.",
            )

        order = select.args.get("order")
        if order and any(
            isinstance(item.this, exp.Literal) and not item.this.is_string
            for item in order.expressions
        ):
            add(
                "ordinal-order",
                "low",
                "Maintainability",
                "ORDER BY uses a column position",
                "A number such as ORDER BY 1 changes meaning when the SELECT list is reordered.",
                "Sort by the output column name or expression instead of its position.",
            )

        if group:
            for item in group.expressions:
                if isinstance(item, exp.Column) and item.name in dataframe.columns and len(dataframe) >= 20:
                    unique_ratio = dataframe[item.name].nunique(dropna=True) / max(len(dataframe), 1)
                    if unique_ratio >= 0.9:
                        add(
                            f"high-cardinality-group-{item.name}",
                            "medium",
                            "Performance",
                            "Grouping column is nearly unique",
                            f"{item.name!r} has a distinct-value ratio of {unique_ratio:.0%}, so most groups may contain only one row.",
                            "Confirm this is a meaningful grouping key; use a broader category or date grain when possible.",
                        )

    for like in [*expression.find_all(exp.Like), *expression.find_all(exp.ILike)]:
        pattern = like.expression
        if isinstance(pattern, exp.Literal) and pattern.is_string and pattern.this.startswith("%"):
            add(
                "leading-wildcard",
                "medium",
                "Performance",
                "Text search starts with a wildcard",
                f"The pattern {pattern.sql()} must check positions inside each candidate value.",
                "Use an exact match or a prefix pattern such as 'term%' when that matches the business question.",
            )

    for comparison in [*expression.find_all(exp.EQ), *expression.find_all(exp.NEQ)]:
        if isinstance(comparison.this, exp.Null) or isinstance(comparison.expression, exp.Null):
            add(
                "null-comparison",
                "high",
                "Correctness",
                "NULL is compared with = or <>",
                "SQL NULL represents an unknown value, so ordinary equality does not return the intended missing-value matches.",
                "Use IS NULL or IS NOT NULL.",
            )

    for not_expression in expression.find_all(exp.Not):
        if isinstance(not_expression.this, exp.In) and not_expression.this.args.get("query"):
            add(
                "not-in-subquery",
                "high",
                "Correctness",
                "NOT IN subquery can be defeated by NULL",
                "If the subquery returns one NULL, NOT IN can evaluate to unknown for every candidate row.",
                "Prefer NOT EXISTS with a correlated equality condition, or explicitly exclude NULL in the subquery.",
            )

    where_nodes = [select.args.get("where") for select in expression.find_all(exp.Select)]
    predicate_functions: set[str] = set()
    or_count = 0
    for where in [node for node in where_nodes if node]:
        or_count += sum(1 for _ in where.find_all(exp.Or))
        for function in where.find_all(exp.Func):
            if function.find(exp.Column):
                predicate_functions.add(function.key.upper())
    if predicate_functions:
        add(
            "function-on-filter-column",
            "medium",
            "Performance",
            "Filter transforms a column before comparison",
            f"The WHERE clause applies {_short_list(sorted(predicate_functions))} to one or more columns before filtering.",
            "When equivalent, compare the original column to a transformed constant or use a range predicate so filtering can happen earlier.",
        )
    if or_count >= 3:
        add(
            "many-or-branches",
            "low",
            "Maintainability",
            "WHERE contains many OR branches",
            f"The query contains {or_count + 1} OR alternatives, which are difficult to audit and can broaden the scan.",
            "For repeated equality checks on one column, use IN (...); otherwise verify each branch is necessary.",
        )

    joins = list(expression.find_all(exp.Join))
    for join in joins:
        kind = (join.kind or "").upper()
        if kind == "CROSS":
            add(
                "cross-join",
                "high",
                "Performance",
                "CROSS JOIN multiplies rows",
                "Every row from one side is paired with every row from the other side.",
                "Use an ON or USING condition unless the Cartesian product is explicitly required.",
            )
        elif not join.args.get("on") and not join.args.get("using"):
            add(
                "join-without-condition",
                "high",
                "Correctness",
                "JOIN has no matching condition",
                "Without ON or USING, the join can behave like a Cartesian product.",
                "Add the key relationship that determines which rows should match.",
            )
    uploaded_scans = sum(
        1
        for table in expression.find_all(exp.Table)
        if isinstance(table.this, exp.Identifier) and table.name.casefold() == "uploaded_data"
    )
    if joins and uploaded_scans > 1:
        add(
            "repeated-upload-scan",
            "medium",
            "Performance",
            "The uploaded dataset is scanned more than once",
            f"The query references uploaded_data {uploaded_scans} times across a join or subquery.",
            "Confirm the self-join is required and make the join condition as selective as possible.",
        )

    for union in expression.find_all(exp.Union):
        if union.args.get("distinct") is not False:
            add(
                "union-deduplication",
                "low",
                "Performance",
                "UNION removes duplicate rows",
                "UNION performs duplicate elimination after combining both branches.",
                "Use UNION ALL when retaining duplicates is acceptable or the branches cannot overlap.",
            )

    order = {"high": 0, "medium": 1, "low": 2}
    return tuple(sorted(findings, key=lambda item: (order[item.severity], item.rule_id)))


def _build_physical_plan(dataframe: pd.DataFrame, expression: exp.Query) -> str:
    connection = duckdb.connect(
        database=":memory:",
        config={"enable_external_access": False, "threads": 2},
    )
    try:
        connection.register("uploaded_data", dataframe)
        rows = connection.execute(
            "EXPLAIN " + expression.sql(dialect="duckdb")
        ).fetchall()
    except duckdb.Error as exc:
        raise SqlCoachError(f"DuckDB cannot plan this query against uploaded_data: {exc}") from exc
    finally:
        connection.close()
    return "\n\n".join(str(row[-1]) for row in rows)


def _summarize_plan(physical_plan: str) -> tuple[PlanStep, ...]:
    positions = {
        operator: physical_plan.find(operator)
        for operator in PLAN_OPERATOR_EXPLANATIONS
        if operator in physical_plan
    }
    return tuple(
        PlanStep(operator, PLAN_OPERATOR_EXPLANATIONS[operator])
        for operator, _ in sorted(positions.items(), key=lambda item: item[1])
    )


def _primary_select(expression: exp.Query) -> exp.Select | None:
    if isinstance(expression, exp.Select):
        return expression
    return expression.find(exp.Select)


def _is_projection_star(expression: exp.Expression) -> bool:
    return isinstance(expression, exp.Star) or (
        isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star)
    )


def _short_sql(expression: exp.Expression, limit: int = 140) -> str:
    text = expression.sql(dialect="duckdb")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _short_list(values: list[str], limit: int = 6) -> str:
    visible = values[:limit]
    text = ", ".join(visible)
    if len(values) > limit:
        text += f", and {len(values) - limit} more"
    return text or "none"
