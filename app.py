"""Streamlit interface for Open Data Scientist."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from ods import (
    AggregateRule,
    DEFAULT_LIMITS,
    DashboardCard,
    DashboardConfig,
    DashboardFilter,
    DashboardStudioError,
    ChartSuggestion,
    CleaningAction,
    CleaningError,
    DatasetLoadError,
    FilterRule,
    INTENTS,
    QueryBuilderError,
    QuerySpec,
    SqlCoachError,
    analyze_query,
    apply_cleaning_actions,
    apply_dashboard_filters,
    build_card_result,
    build_chart_data,
    build_cleaning_recipe,
    build_dashboard_html,
    build_markdown_report,
    build_query,
    categorical_filter_options,
    dashboard_config_from_json,
    dashboard_config_to_json,
    default_dashboard_config,
    default_filter_for_column,
    execute_query,
    format_metric_value,
    infer_column_semantics,
    load_dataset,
    move_dashboard_card,
    next_dashboard_id,
    profile_dataset,
    replay_cleaning_batches,
    remove_dashboard_card,
    replace_dashboard_card,
    replace_dashboard_filter,
    roles_from_mapping,
    suggest_dashboard,
    suggest_cleaning_actions,
    validate_dashboard_config,
    __release__,
)


ROLE_LABELS = {
    "identifier": "Identifier / ID",
    "numeric": "Numeric measure",
    "categorical": "Category",
    "datetime": "Date / time",
    "text": "Free text",
    "ignore": "Ignore",
}
ROLE_VALUES = {label: role for role, label in ROLE_LABELS.items()}
AGGREGATIONS = {
    "Average": "mean",
    "Median": "median",
    "Total": "sum",
    "Count": "count",
    "Minimum": "min",
    "Maximum": "max",
}
FILTER_OPERATORS = {
    "Equals": "eq",
    "Does not equal": "ne",
    "Greater than": "gt",
    "At least": "gte",
    "Less than": "lt",
    "At most": "lte",
    "Contains": "contains",
    "Starts with": "starts_with",
    "Is missing": "is_null",
    "Is not missing": "is_not_null",
}
SUMMARY_CALCULATIONS = {
    "Count rows": "count_rows",
    "Count non-missing": "count_values",
    "Average": "mean",
    "Median": "median",
    "Total": "sum",
    "Minimum": "min",
    "Maximum": "max",
}
STUDIO_CARD_LABELS = {
    "kpi": "KPI card",
    "bar": "Bar chart",
    "line": "Line chart",
    "scatter": "Scatter plot",
    "distribution": "Distribution",
}
STUDIO_CARD_VALUES = {label: value for value, label in STUDIO_CARD_LABELS.items()}
STUDIO_METRIC_LABELS = {
    "row_count": "Row count",
    "sum": "Total",
    "mean": "Average",
    "median": "Median",
    "distinct_count": "Distinct count",
}
STUDIO_METRIC_VALUES = {
    label: value for value, label in STUDIO_METRIC_LABELS.items()
}
STUDIO_DATE_GRAINS = {
    "Day": "day",
    "Week": "week",
    "Month": "month",
    "Quarter": "quarter",
    "Year": "year",
}
SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "examples" / "sample_customers.csv"


def get_cleaning_state(dataset_key: str) -> dict[str, object]:
    """Return the action history and preview for one uploaded file."""
    state_key = f"cleaning-state-{dataset_key}"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or "batches" not in state:
        state = {"batches": [], "preview": None}
        st.session_state[state_key] = state
    return state


def cleaning_history_fingerprint(batches: list[tuple[CleaningAction, ...]]) -> str:
    """Create a stable downstream cache key for the active cleaned dataset."""
    serialized = tuple(
        tuple((action.action_id, action.parameters) for action in batch)
        for batch in batches
    )
    return sha256(repr(serialized).encode()).hexdigest()[:10]


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def render_chart(dataframe, suggestion: ChartSuggestion) -> None:
    chart_data = build_chart_data(dataframe, suggestion)
    if chart_data.empty:
        st.info("There is not enough usable data to draw this chart.")
        return

    if suggestion.kind == "time_series":
        value_column = suggestion.y or "count"
        st.line_chart(chart_data.set_index(suggestion.x)[value_column], height=300)
    elif suggestion.kind == "scatter" and suggestion.y:
        st.scatter_chart(chart_data, x=suggestion.x, y=suggestion.y, height=300)
    else:
        value_column = suggestion.y or "count"
        st.bar_chart(chart_data.set_index(suggestion.x)[value_column], height=300)

    st.caption(f"Recommendation confidence: {suggestion.confidence:.0%}")
    with st.expander("Verify calculation"):
        st.dataframe(chart_data, width="stretch", hide_index=True)
        st.caption(
            f"This is the exact {len(chart_data):,}-row summary used by the chart. "
            "No hidden model or paid API is involved."
        )


def build_role_review(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the editable semantic layer shown above the dashboard."""
    return pd.DataFrame(
        [
            {
                "Column": semantic.column,
                "Role": ROLE_LABELS[semantic.role],
                "Format": semantic.display_format,
                "Confidence": f"{semantic.confidence:.0%}",
                "Why ODS chose it": semantic.reason,
            }
            for semantic in infer_column_semantics(dataframe)
        ]
    )


def filter_operator_labels(series: pd.Series) -> list[str]:
    """Return sensible visual filter choices for a column's data type."""
    if is_bool_dtype(series):
        return ["Equals", "Does not equal", "Is missing", "Is not missing"]
    if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
        return [
            "Equals",
            "Does not equal",
            "Greater than",
            "At least",
            "Less than",
            "At most",
            "Is missing",
            "Is not missing",
        ]
    return [
        "Equals",
        "Does not equal",
        "Contains",
        "Starts with",
        "Is missing",
        "Is not missing",
    ]


def default_aggregate_alias(function: str, column: str | None) -> str:
    """Create a short, readable output name for a summary calculation."""
    labels = {
        "count_rows": "row_count",
        "count_values": "count",
        "mean": "average",
        "median": "median",
        "sum": "total",
        "min": "minimum",
        "max": "maximum",
    }
    return labels[function] if column is None else f"{labels[function]}_{column}"


def render_visual_sql_builder(
    dataframe: pd.DataFrame,
    dataset_key: str,
    filename: str,
) -> None:
    """Render a button-driven query builder backed by in-memory DuckDB."""
    st.subheader("Visual SQL query builder")
    st.caption(
        "Choose columns, filters, summaries, and sorting with controls. ODS generates readable SQL "
        "and runs it only against this uploaded dataset in memory."
    )

    column_names = [str(column) for column in dataframe.columns]
    if not column_names:
        st.info("This dataset has no columns to query.")
        return
    if len(column_names) != len(set(column_names)):
        st.error("Visual SQL needs unique column names. Rename duplicate columns and upload the file again.")
        return

    key_prefix = f"visual-sql-{dataset_key}"
    mode = st.radio(
        "What do you want to create?",
        ["View and filter rows", "Create a summary"],
        horizontal=True,
        key=f"{key_prefix}-mode",
    )

    selected_columns: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    aggregates: tuple[AggregateRule, ...] = ()
    output_names: list[str]

    if mode == "View and filter rows":
        chosen_columns = st.multiselect(
            "Columns to show",
            column_names,
            default=column_names,
            help="The result keeps the same row-level detail as the upload.",
            key=f"{key_prefix}-columns",
        )
        selected_columns = tuple(chosen_columns)
        output_names = list(chosen_columns)
    else:
        semantics = infer_column_semantics(dataframe)
        suggested_group = next(
            (
                semantic.column
                for semantic in semantics
                if semantic.role in {"categorical", "datetime"}
            ),
            None,
        )
        chosen_groups = st.multiselect(
            "Group results by",
            column_names,
            default=[suggested_group] if suggested_group else [],
            help="Leave this empty to calculate one total for the full dataset.",
            key=f"{key_prefix}-groups",
        )
        group_by = tuple(chosen_groups)

        numeric_columns = [
            name for name in column_names if is_numeric_dtype(dataframe[name])
        ]
        calculation_labels = ["Count rows", "Count non-missing"]
        if numeric_columns:
            calculation_labels.extend(
                ["Average", "Median", "Total", "Minimum", "Maximum"]
            )
        calculation_label = st.selectbox(
            "Summary calculation",
            calculation_labels,
            key=f"{key_prefix}-calculation",
        )
        function = SUMMARY_CALCULATIONS[calculation_label]
        source_column: str | None = None
        if function != "count_rows":
            candidates = column_names if function == "count_values" else numeric_columns
            source_column = st.selectbox(
                "Column to summarize",
                candidates,
                key=f"{key_prefix}-summary-column",
            )
        default_alias = default_aggregate_alias(function, source_column)
        alias = st.text_input(
            "Result column name",
            value=default_alias,
            key=f"{key_prefix}-alias-{function}-{source_column or 'rows'}",
        ).strip()
        aggregates = (
            AggregateRule(function=function, column=source_column, alias=alias),
        )
        output_names = [*chosen_groups, alias]

    st.markdown("#### Optional filters")
    filter_count = st.slider(
        "Number of filters",
        min_value=0,
        max_value=3,
        value=0,
        help="Filters are combined with AND.",
        key=f"{key_prefix}-filter-count",
    )
    filters: list[FilterRule] = []
    for index in range(filter_count):
        column_control, operator_control, value_control = st.columns([1.2, 1.2, 1.6])
        filter_column = column_control.selectbox(
            f"Filter {index + 1} column",
            column_names,
            key=f"{key_prefix}-filter-column-{index}",
        )
        operator_label = operator_control.selectbox(
            f"Filter {index + 1} rule",
            filter_operator_labels(dataframe[filter_column]),
            key=f"{key_prefix}-filter-operator-{index}",
        )
        operator = FILTER_OPERATORS[operator_label]
        filter_value = None
        if operator in {"is_null", "is_not_null"}:
            value_control.caption("No value is needed for this rule.")
        elif is_bool_dtype(dataframe[filter_column]):
            filter_value = value_control.selectbox(
                f"Filter {index + 1} value",
                [True, False],
                key=f"{key_prefix}-filter-value-{index}",
            )
        elif is_datetime64_any_dtype(dataframe[filter_column]):
            non_null_dates = pd.to_datetime(
                dataframe[filter_column], errors="coerce"
            ).dropna()
            default_date = (
                non_null_dates.iloc[0].date()
                if not non_null_dates.empty
                else pd.Timestamp.today().date()
            )
            filter_value = value_control.date_input(
                f"Filter {index + 1} value",
                value=default_date,
                key=f"{key_prefix}-filter-value-{index}",
            )
        else:
            filter_value = value_control.text_input(
                f"Filter {index + 1} value",
                help="Numeric values may include commas, such as 1,000.",
                key=f"{key_prefix}-filter-value-{index}",
            )
        filters.append(FilterRule(filter_column, operator, filter_value))

    st.markdown("#### Sort and limit")
    sort_control, direction_control, limit_control = st.columns([1.4, 1, 1])
    sort_label = sort_control.selectbox(
        "Sort results by",
        ["No sorting", *output_names],
        key=f"{key_prefix}-sort-column",
    )
    direction_label = direction_control.selectbox(
        "Direction",
        ["Ascending", "Descending"],
        disabled=sort_label == "No sorting",
        key=f"{key_prefix}-sort-direction",
    )
    limit = int(
        limit_control.number_input(
            "Maximum rows",
            min_value=1,
            max_value=5000,
            value=100,
            step=25,
            key=f"{key_prefix}-limit",
        )
    )

    spec = QuerySpec(
        selected_columns=selected_columns,
        filters=tuple(filters),
        group_by=group_by,
        aggregates=aggregates,
        sort_by=None if sort_label == "No sorting" else sort_label,
        sort_descending=direction_label == "Descending",
        limit=limit,
    )
    signature = sha256(f"{dataset_key}:{spec!r}".encode()).hexdigest()
    state_key = f"{key_prefix}-result"

    try:
        query = build_query(dataframe, spec)
    except QueryBuilderError as exc:
        st.error(str(exc))
        query = None

    if query is not None:
        st.session_state[f"sql-coach-latest-{dataset_key}"] = query.display_sql
        with st.expander("Generated SQL", expanded=False):
            st.code(query.display_sql, language="sql")
            st.caption(
                "ODS quotes column names and binds filter values separately before execution. "
                "The displayed SQL is a readable copy for learning and reuse."
            )

    if st.button(
        "Run query",
        type="primary",
        disabled=query is None,
        key=f"{key_prefix}-run",
    ) and query is not None:
        try:
            result = execute_query(dataframe, spec)
            st.session_state[state_key] = {
                "signature": signature,
                "dataframe": result.dataframe,
            }
        except QueryBuilderError as exc:
            st.error(str(exc))

    saved_result = st.session_state.get(state_key)
    if saved_result and saved_result["signature"] == signature:
        result_frame = saved_result["dataframe"]
        result_metrics = st.columns(2)
        result_metrics[0].metric("Result rows", f"{len(result_frame):,}")
        result_metrics[1].metric("Result columns", f"{len(result_frame.columns):,}")
        st.dataframe(result_frame.head(500), width="stretch", hide_index=True)
        if len(result_frame) > 500:
            st.caption("Previewing the first 500 rows; the download includes the full result.")
        download_name = filename.rsplit(".", 1)[0] + "-query-result.csv"
        st.download_button(
            "Download query result",
            data=result_frame.to_csv(index=False).encode("utf-8"),
            file_name=download_name,
            mime="text/csv",
            key=f"{key_prefix}-download",
        )
    elif saved_result:
        st.info("The query controls changed. Run the query again to refresh the result.")


def render_sql_coach(dataframe: pd.DataFrame, dataset_key: str) -> None:
    """Render local query explanation, safety, and optimization guidance."""
    st.subheader("SQL Coach")
    st.caption(
        "Understand a DuckDB query before running it. The coach parses the SQL locally, checks "
        "correctness and performance risks, and asks DuckDB for the real physical plan."
    )

    latest_sql = st.session_state.get(f"sql-coach-latest-{dataset_key}")
    source_options = ["Paste or edit SQL"]
    if latest_sql:
        source_options.insert(0, "Latest Visual SQL query")
    source = st.radio(
        "Query source",
        source_options,
        horizontal=True,
        key=f"sql-coach-source-{dataset_key}",
    )

    if source == "Latest Visual SQL query":
        sql = latest_sql
        st.code(sql, language="sql")
        st.caption("Change the Visual SQL controls to update this query automatically.")
    else:
        sql = st.text_area(
            "DuckDB SQL",
            value="SELECT *\nFROM uploaded_data\nLIMIT 100;",
            height=220,
            help="Use uploaded_data as the table name. Only one read-only query is accepted.",
            key=f"sql-coach-input-{dataset_key}",
        )

    st.info(
        "Safety boundary: only one read-only query against `uploaded_data` is analyzed. "
        "The coach blocks writes, database commands, other tables, and external file readers."
    )
    signature = sha256(f"{dataset_key}:{sql}".encode()).hexdigest()
    state_key = f"sql-coach-result-{dataset_key}"

    if st.button(
        "Analyze query",
        type="primary",
        key=f"sql-coach-run-{dataset_key}",
    ):
        try:
            analysis = analyze_query(dataframe, sql)
            st.session_state[state_key] = {
                "signature": signature,
                "analysis": analysis,
            }
        except SqlCoachError as exc:
            st.session_state.pop(state_key, None)
            st.error(str(exc))

    saved = st.session_state.get(state_key)
    if not saved:
        return
    if saved["signature"] != signature:
        st.info("The SQL changed. Analyze it again to refresh the explanation and plan.")
        return

    analysis = saved["analysis"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Query score", f"{analysis.score}/100")
    metric_columns[1].metric("Findings", f"{len(analysis.findings):,}")
    metric_columns[2].metric("Plan operators", f"{len(analysis.plan_steps):,}")
    st.success(
        "Read-only validation passed. DuckDB planned this query without executing its result."
    )

    st.markdown("#### What the query does")
    clause_columns = st.columns(2)
    for index, clause in enumerate(analysis.clauses):
        with clause_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{clause.clause}**")
                st.write(clause.explanation)

    st.markdown("#### Optimization and correctness review")
    if not analysis.findings:
        st.success("No rule-based risks were detected for this dataset and query shape.")
    for finding in analysis.findings:
        message = (
            f"**{finding.title}** · {finding.category}\n\n"
            f"{finding.detail}\n\n**Recommendation:** {finding.recommendation}"
        )
        if finding.severity == "high":
            st.error(message)
        elif finding.severity == "medium":
            st.warning(message)
        else:
            st.info(message)

    st.markdown("#### Clean DuckDB rewrite")
    st.code(analysis.suggested_sql, language="sql")
    if analysis.suggested_sql != analysis.formatted_sql:
        st.caption(
            "This rewrite expands an unambiguous top-level `SELECT *` using the current upload. "
            "Other recommendations are not applied automatically when they could change results."
        )
    else:
        st.caption(
            "The query is formatted consistently. Recommendations that could change results remain "
            "advice instead of being applied silently."
        )
    st.download_button(
        "Download clean SQL",
        data=analysis.suggested_sql.encode("utf-8"),
        file_name="ods-clean-query.sql",
        mime="text/plain",
        key=f"sql-coach-download-{dataset_key}",
    )

    st.markdown("#### DuckDB physical plan")
    if analysis.plan_steps:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Operator": step.operator,
                        "What it does": step.explanation,
                    }
                    for step in analysis.plan_steps
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    with st.expander("Raw EXPLAIN plan"):
        st.code(analysis.physical_plan, language="text")
        st.caption(
            "The plan is generated by DuckDB. Optimization findings are transparent local rules, "
            "not guarantees from an AI model."
        )


def render_cleaning_studio(
    original: pd.DataFrame,
    current: pd.DataFrame,
    state: dict[str, object],
    dataset_key: str,
    source_sha256: str,
    filename: str,
) -> None:
    """Render review, preview, apply, undo, reset, and export controls."""
    st.subheader("Data Cleaning Studio")
    st.caption(
        "Review deterministic fixes before they touch the working dataset. Every operation shows its "
        "evidence, requires an explicit preview and Apply, and remains reproducible from the source upload."
    )

    batches = state.get("batches", [])
    if not isinstance(batches, list):
        batches = []
        state["batches"] = batches
    applied_actions = [action for batch in batches for action in batch]
    original_profile = profile_dataset(original)
    current_profile = profile_dataset(current)
    history_key = cleaning_history_fingerprint(batches)

    metrics = st.columns(4)
    score_delta = current_profile.health_score - original_profile.health_score
    metrics[0].metric(
        "Current quality score",
        f"{current_profile.health_score}/100",
        delta=f"{score_delta:+d} from upload" if score_delta else "No score change",
    )
    metrics[1].metric(
        "Current rows",
        f"{current_profile.rows:,}",
        delta=f"{current_profile.rows - original_profile.rows:+,}",
    )
    metrics[2].metric(
        "Current columns",
        f"{current_profile.columns:,}",
        delta=f"{current_profile.columns - original_profile.columns:+,}",
    )
    metrics[3].metric("Applied fixes", f"{len(applied_actions):,}")
    st.caption(
        "The score measures missingness, exact duplicates, and constant columns. A truthful type or "
        "format correction may improve trust without changing the score."
    )

    controls = st.columns([1, 1, 4])
    if controls[0].button(
        "Undo last batch",
        disabled=not batches,
        key=f"cleaning-undo-{dataset_key}-{history_key}",
    ):
        batches.pop()
        state["preview"] = None
        st.rerun()
    if controls[1].button(
        "Reset all",
        disabled=not batches,
        key=f"cleaning-reset-{dataset_key}-{history_key}",
    ):
        batches.clear()
        state["preview"] = None
        st.rerun()

    if applied_actions:
        with st.expander(f"Applied history · {len(applied_actions)} fixes"):
            for index, action in enumerate(applied_actions, start=1):
                location = f" · `{action.column}`" if action.column else ""
                st.markdown(
                    f"{index}. **{action.title}**{location} — {action.evidence}"
                )

    try:
        suggestions = suggest_cleaning_actions(current)
    except CleaningError as exc:
        st.error(str(exc))
        suggestions = ()

    st.markdown("#### 1. Review suggested fixes")
    if not suggestions:
        st.success(
            "No conservative cleaning fixes remain. Domain-specific rules may still be needed."
        )
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Confidence": action.confidence.title(),
                        "Suggested fix": action.title,
                        "Estimated impact": (
                            f"{action.affected_rows:,} rows "
                            f"({action.affected_percent:.1f}%)"
                        ),
                        "Evidence": action.evidence,
                    }
                    for action in suggestions
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    action_lookup = {action.action_id: action for action in suggestions}
    selected_ids = st.multiselect(
        "Choose fixes to preview",
        list(action_lookup),
        default=[],
        format_func=lambda action_id: (
            f"{action_lookup[action_id].confidence.title()} confidence · "
            f"{action_lookup[action_id].title}"
        ),
        help="Nothing is selected automatically. Drop-column fixes cannot be combined with other fixes for the same column.",
        key=f"cleaning-selection-{dataset_key}-{history_key}",
    )
    selected_actions = tuple(action_lookup[action_id] for action_id in selected_ids)
    preview_signature = sha256(
        repr((history_key, tuple(selected_ids))).encode()
    ).hexdigest()

    if selected_actions:
        with st.expander("Selected assumptions", expanded=True):
            for action in selected_actions:
                st.markdown(
                    f"- **{action.title}** — {action.recommendation} "
                    f"({action.affected_rows:,} estimated rows)"
                )

    if st.button(
        "Preview selected fixes",
        type="primary",
        disabled=not selected_actions,
        key=f"cleaning-preview-{dataset_key}-{history_key}",
    ):
        try:
            preview_frame = apply_cleaning_actions(current, selected_actions)
            state["preview"] = {
                "signature": preview_signature,
                "dataframe": preview_frame,
                "actions": selected_actions,
            }
        except CleaningError as exc:
            state["preview"] = None
            st.error(str(exc))

    preview = state.get("preview")
    if isinstance(preview, dict) and preview.get("signature") != preview_signature:
        st.info("The selected fixes changed. Preview them again before applying.")
        preview = None

    if isinstance(preview, dict):
        preview_frame = preview["dataframe"]
        preview_profile = profile_dataset(preview_frame)
        st.markdown("#### 2. Verify the before/after preview")
        impact = st.columns(4)
        impact[0].metric(
            "Quality score",
            f"{preview_profile.health_score}/100",
            delta=f"{preview_profile.health_score - current_profile.health_score:+d}",
        )
        impact[1].metric(
            "Rows",
            f"{preview_profile.rows:,}",
            delta=f"{preview_profile.rows - current_profile.rows:+,}",
        )
        impact[2].metric(
            "Columns",
            f"{preview_profile.columns:,}",
            delta=f"{preview_profile.columns - current_profile.columns:+,}",
        )
        changed_types = sum(
            str(current.dtypes[column]) != str(preview_frame.dtypes[column])
            for column in current.columns.intersection(preview_frame.columns)
        )
        impact[3].metric("Changed data types", f"{changed_types:,}")

        sample_columns = st.columns(2)
        with sample_columns[0]:
            st.markdown("**Before · first 12 rows**")
            st.dataframe(current.head(12), width="stretch", hide_index=True)
        with sample_columns[1]:
            st.markdown("**After · first 12 rows**")
            st.dataframe(preview_frame.head(12), width="stretch", hide_index=True)

        st.warning(
            "Apply updates the working dataset used by the dashboard, Visual SQL, SQL Coach, "
            "profiles, statistics, and downloads. The original upload remains available through Reset."
        )
        if st.button(
            "Apply verified fixes",
            type="primary",
            key=f"cleaning-apply-{dataset_key}-{history_key}",
        ):
            batches.append(tuple(preview["actions"]))
            state["preview"] = None
            st.rerun()

    st.markdown("#### 3. Export the current result")
    export_columns = st.columns(2)
    base_name = filename.rsplit(".", 1)[0]
    export_columns[0].download_button(
        "Download current cleaned CSV",
        data=current.to_csv(index=False).encode("utf-8"),
        file_name=f"{base_name}-cleaned.csv",
        mime="text/csv",
        key=f"cleaning-download-data-{dataset_key}-{history_key}",
    )
    recipe = build_cleaning_recipe(
        filename,
        source_sha256,
        original,
        current,
        batches,
    )
    export_columns[1].download_button(
        "Download cleaning recipe",
        data=recipe.encode("utf-8"),
        file_name=f"{base_name}-cleaning-recipe.json",
        mime="application/json",
        key=f"cleaning-download-recipe-{dataset_key}-{history_key}",
    )
    st.caption(
        "The recipe records the source hash, ordered operations, parameters, evidence, and before/after schema."
    )


def option_index(options: list[str], value: str) -> int:
    """Return a safe selectbox index for a possibly refreshed configuration."""
    return options.index(value) if value in options else 0


def safe_download_stem(value: str) -> str:
    """Create a short cross-platform filename without changing the visible title."""
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return (normalized or "ods-dashboard")[:60]


def get_dashboard_studio_state(
    dataframe: pd.DataFrame,
    dataset_key: str,
) -> dict[str, object]:
    """Return a validated dashboard state scoped to the active cleaned dataset."""
    state_key = f"dashboard-studio-state-{dataset_key}"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or not isinstance(
        state.get("config"), DashboardConfig
    ):
        state = {
            "config": default_dashboard_config(dataframe),
            "nonce": 0,
            "html_export": None,
        }
        st.session_state[state_key] = state
        return state
    try:
        validate_dashboard_config(dataframe, state["config"])
    except DashboardStudioError:
        state = {
            "config": default_dashboard_config(dataframe),
            "nonce": int(state.get("nonce", 0)) + 1,
            "html_export": None,
        }
        st.session_state[state_key] = state
    return state


def commit_dashboard_config(
    dataframe: pd.DataFrame,
    state: dict[str, object],
    config: DashboardConfig,
    *,
    reset_widgets: bool = False,
) -> None:
    """Validate and save one local layout, invalidating stale HTML exports."""
    validate_dashboard_config(dataframe, config)
    state["config"] = config
    state["html_export"] = None
    if reset_widgets:
        state["nonce"] = int(state.get("nonce", 0)) + 1


def studio_column_groups(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Return all, numeric-compatible, and inferred date columns for controls."""
    columns = [str(column) for column in dataframe.columns]
    numeric = [
        column
        for column in columns
        if not is_bool_dtype(dataframe[column])
        and pd.to_numeric(dataframe[column], errors="coerce").notna().any()
    ]
    date_columns = [
        semantic.column
        for semantic in infer_column_semantics(dataframe)
        if semantic.role == "datetime"
    ]
    return columns, numeric, date_columns


def render_dashboard_card_editor(
    dataframe: pd.DataFrame,
    card: DashboardCard,
    key_prefix: str,
) -> DashboardCard:
    """Render type-aware card controls and return the proposed definition."""
    columns, numeric, date_columns = studio_column_groups(dataframe)
    available_kinds = ["kpi"]
    if columns:
        available_kinds.append("bar")
    if date_columns:
        available_kinds.append("line")
    if len(numeric) >= 2:
        available_kinds.append("scatter")
    if numeric:
        available_kinds.append("distribution")
    if card.kind not in available_kinds:
        available_kinds.append(card.kind)
    kind_labels = [STUDIO_CARD_LABELS[kind] for kind in available_kinds]
    kind_label = st.selectbox(
        "Card type",
        kind_labels,
        index=option_index(kind_labels, STUDIO_CARD_LABELS[card.kind]),
        key=f"{key_prefix}-kind",
    )
    kind = STUDIO_CARD_VALUES[kind_label]
    title = st.text_input(
        "Card title",
        value=card.title,
        max_chars=100,
        key=f"{key_prefix}-title",
    ).strip()

    if kind == "kpi":
        metric_values = ["row_count"]
        if numeric:
            metric_values.extend(["sum", "mean", "median"])
        if columns:
            metric_values.append("distinct_count")
        metric_labels = [STUDIO_METRIC_LABELS[value] for value in metric_values]
        current_metric = card.metric if card.metric in metric_values else "row_count"
        metric_label = st.selectbox(
            "Calculation",
            metric_labels,
            index=option_index(
                metric_labels, STUDIO_METRIC_LABELS[current_metric]
            ),
            key=f"{key_prefix}-metric",
        )
        metric = STUDIO_METRIC_VALUES[metric_label]
        column = None
        if metric != "row_count":
            candidates = columns if metric == "distinct_count" else numeric
            current_column = card.column if card.column in candidates else candidates[0]
            column = st.selectbox(
                "Source column",
                candidates,
                index=option_index(candidates, current_column),
                key=f"{key_prefix}-column-{metric}",
            )
        return DashboardCard(
            card.card_id,
            "kpi",
            title,
            metric=metric,
            column=column,
        )

    if kind == "bar":
        current_x = card.x if card.x in columns else columns[0]
        x = st.selectbox(
            "Category column",
            columns,
            index=option_index(columns, current_x),
            key=f"{key_prefix}-bar-x",
        )
        aggregation_values = ["row_count", "distinct_count"]
        if numeric:
            aggregation_values[1:1] = ["sum", "mean", "median"]
        aggregation_labels = [
            STUDIO_METRIC_LABELS[value] for value in aggregation_values
        ]
        current_aggregation = (
            card.aggregation
            if card.aggregation in aggregation_values
            else "row_count"
        )
        aggregation_label = st.selectbox(
            "Calculation",
            aggregation_labels,
            index=option_index(
                aggregation_labels,
                STUDIO_METRIC_LABELS[current_aggregation],
            ),
            key=f"{key_prefix}-bar-aggregation",
        )
        aggregation = STUDIO_METRIC_VALUES[aggregation_label]
        y = None
        if aggregation != "row_count":
            candidates = columns if aggregation == "distinct_count" else numeric
            current_y = card.y if card.y in candidates else candidates[0]
            y = st.selectbox(
                "Value column",
                candidates,
                index=option_index(candidates, current_y),
                key=f"{key_prefix}-bar-y-{aggregation}",
            )
        top_n = int(
            st.number_input(
                "Maximum groups",
                min_value=3,
                max_value=50,
                value=int(card.top_n),
                step=1,
                key=f"{key_prefix}-bar-top-n",
            )
        )
        return DashboardCard(
            card.card_id,
            "bar",
            title,
            x=x,
            y=y,
            aggregation=aggregation,
            top_n=top_n,
        )

    if kind == "line":
        current_x = card.x if card.x in date_columns else date_columns[0]
        x = st.selectbox(
            "Date column",
            date_columns,
            index=option_index(date_columns, current_x),
            key=f"{key_prefix}-line-x",
        )
        aggregation_values = ["row_count", "distinct_count"]
        if numeric:
            aggregation_values[1:1] = ["sum", "mean", "median"]
        aggregation_labels = [
            STUDIO_METRIC_LABELS[value] for value in aggregation_values
        ]
        current_aggregation = (
            card.aggregation
            if card.aggregation in aggregation_values
            else "row_count"
        )
        aggregation_label = st.selectbox(
            "Calculation",
            aggregation_labels,
            index=option_index(
                aggregation_labels,
                STUDIO_METRIC_LABELS[current_aggregation],
            ),
            key=f"{key_prefix}-line-aggregation",
        )
        aggregation = STUDIO_METRIC_VALUES[aggregation_label]
        y = None
        if aggregation != "row_count":
            candidates = columns if aggregation == "distinct_count" else numeric
            current_y = card.y if card.y in candidates else candidates[0]
            y = st.selectbox(
                "Value column",
                candidates,
                index=option_index(candidates, current_y),
                key=f"{key_prefix}-line-y-{aggregation}",
            )
        grain_labels = list(STUDIO_DATE_GRAINS)
        current_grain = next(
            label
            for label, value in STUDIO_DATE_GRAINS.items()
            if value == card.date_grain
        )
        grain_label = st.selectbox(
            "Date grain",
            grain_labels,
            index=option_index(grain_labels, current_grain),
            key=f"{key_prefix}-line-grain",
        )
        return DashboardCard(
            card.card_id,
            "line",
            title,
            x=x,
            y=y,
            aggregation=aggregation,
            date_grain=STUDIO_DATE_GRAINS[grain_label],
        )

    if kind == "scatter":
        current_x = card.x if card.x in numeric else numeric[0]
        x = st.selectbox(
            "X-axis column",
            numeric,
            index=option_index(numeric, current_x),
            key=f"{key_prefix}-scatter-x",
        )
        y_options = [column for column in numeric if column != x]
        current_y = card.y if card.y in y_options else y_options[0]
        y = st.selectbox(
            "Y-axis column",
            y_options,
            index=option_index(y_options, current_y),
            key=f"{key_prefix}-scatter-y-{x}",
        )
        return DashboardCard(card.card_id, "scatter", title, x=x, y=y)

    current_column = card.column if card.column in numeric else numeric[0]
    column = st.selectbox(
        "Numeric column",
        numeric,
        index=option_index(numeric, current_column),
        key=f"{key_prefix}-distribution-column",
    )
    bins = int(
        st.slider(
            "Bins",
            min_value=3,
            max_value=50,
            value=int(card.bins),
            key=f"{key_prefix}-distribution-bins",
        )
    )
    return DashboardCard(
        card.card_id,
        "distribution",
        title,
        column=column,
        bins=bins,
    )


def render_dashboard_filters(
    dataframe: pd.DataFrame,
    config: DashboardConfig,
    state: dict[str, object],
    key_prefix: str,
) -> DashboardConfig:
    """Render global filter controls and persist each validated change."""
    nonce = int(state.get("nonce", 0))
    with st.expander(
        f"Global filters · {len(config.filters)} active",
        expanded=bool(config.filters),
    ):
        st.caption(
            "Filters use AND logic and update every card. An empty value selection means all values."
        )
        for index, item in enumerate(config.filters):
            with st.container(border=True):
                heading, remove_control = st.columns([5, 1])
                heading.markdown(f"**{item.column}** · {item.kind.replace('_', ' ')}")
                if remove_control.button(
                    "Remove",
                    key=f"{key_prefix}-filter-remove-{item.filter_id}-{nonce}",
                ):
                    updated = replace(
                        config,
                        filters=tuple(
                            existing
                            for existing in config.filters
                            if existing.filter_id != item.filter_id
                        ),
                    )
                    commit_dashboard_config(
                        dataframe, state, updated, reset_widgets=True
                    )
                    st.rerun()

                if item.kind == "values":
                    options = list(categorical_filter_options(dataframe, item.column))
                    for saved in item.values:
                        if saved not in options:
                            options.append(saved)
                    selected = tuple(
                        st.multiselect(
                            "Values",
                            options,
                            default=list(item.values),
                            placeholder="All values",
                            key=f"{key_prefix}-filter-values-{item.filter_id}-{nonce}",
                        )
                    )
                    proposed = replace_dashboard_filter(
                        config, replace(item, values=selected)
                    )
                elif item.kind == "range":
                    range_columns = st.columns(2)
                    minimum = float(
                        range_columns[0].number_input(
                            "Minimum",
                            value=float(item.minimum),
                            key=f"{key_prefix}-filter-min-{item.filter_id}-{nonce}",
                        )
                    )
                    maximum = float(
                        range_columns[1].number_input(
                            "Maximum",
                            value=float(item.maximum),
                            key=f"{key_prefix}-filter-max-{item.filter_id}-{nonce}",
                        )
                    )
                    if minimum > maximum:
                        st.error("Minimum must not be greater than maximum.")
                        proposed = config
                    else:
                        proposed = replace_dashboard_filter(
                            config,
                            replace(item, minimum=minimum, maximum=maximum),
                        )
                else:
                    date_columns = st.columns(2)
                    start = date_columns[0].date_input(
                        "Start date",
                        value=pd.Timestamp(item.start).date(),
                        key=f"{key_prefix}-filter-start-{item.filter_id}-{nonce}",
                    )
                    end = date_columns[1].date_input(
                        "End date",
                        value=pd.Timestamp(item.end).date(),
                        key=f"{key_prefix}-filter-end-{item.filter_id}-{nonce}",
                    )
                    if start > end:
                        st.error("Start date must not be after end date.")
                        proposed = config
                    else:
                        proposed = replace_dashboard_filter(
                            config,
                            replace(
                                item,
                                start=start.isoformat(),
                                end=end.isoformat(),
                            ),
                        )

                if proposed != config:
                    commit_dashboard_config(dataframe, state, proposed)
                    st.rerun()

        used_columns = {item.column for item in config.filters}
        available_columns = [
            str(column)
            for column in dataframe.columns
            if str(column) not in used_columns
        ]
        if len(config.filters) >= 5:
            st.info("The five-filter limit keeps dashboards understandable and fast.")
        elif available_columns:
            add_columns = st.columns([3, 1])
            selected_column = add_columns[0].selectbox(
                "Add a filter",
                available_columns,
                key=f"{key_prefix}-filter-new-column-{nonce}",
            )
            if add_columns[1].button(
                "Add filter",
                type="primary",
                key=f"{key_prefix}-filter-add-{nonce}",
            ):
                new_filter = default_filter_for_column(
                    dataframe,
                    selected_column,
                    next_dashboard_id(config, "filter"),
                )
                updated = replace(
                    config, filters=(*config.filters, new_filter)
                )
                commit_dashboard_config(
                    dataframe, state, updated, reset_widgets=True
                )
                st.rerun()
    return config


def render_dashboard_result_card(
    dataframe: pd.DataFrame,
    config: DashboardConfig,
    card: DashboardCard,
    index: int,
    state: dict[str, object],
    key_prefix: str,
) -> None:
    """Render one card, its audit evidence, and compact editing controls."""
    result = build_card_result(dataframe, card)
    if card.kind == "kpi":
        st.metric(card.title, format_metric_value(result.value))
    else:
        st.markdown(f"#### {card.title}")
        if result.audit_table.empty:
            st.info("No usable rows match this card and the active filters.")
        elif result.figure is not None:
            st.plotly_chart(
                result.figure,
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )

    with st.expander("Calculation details"):
        st.caption(result.calculation)
        st.dataframe(
            result.audit_table.head(500),
            width="stretch",
            hide_index=True,
        )
        if len(result.audit_table) > 500:
            st.caption(
                f"Showing 500 of {len(result.audit_table):,} audit rows. The chart uses the full calculated result."
            )

    nonce = int(state.get("nonce", 0))
    with st.expander("Edit card"):
        proposed = render_dashboard_card_editor(
            dataframe,
            card,
            f"{key_prefix}-edit-{card.card_id}-{nonce}",
        )
        if proposed != card:
            try:
                updated = replace_dashboard_card(config, proposed)
                commit_dashboard_config(dataframe, state, updated)
                st.rerun()
            except DashboardStudioError as exc:
                st.error(str(exc))

    controls = st.columns(3)
    if controls[0].button(
        "← Earlier",
        disabled=index == 0,
        key=f"{key_prefix}-earlier-{card.card_id}-{nonce}",
    ):
        updated = move_dashboard_card(config, card.card_id, -1)
        commit_dashboard_config(dataframe, state, updated, reset_widgets=True)
        st.rerun()
    if controls[1].button(
        "Later →",
        disabled=index == len(config.cards) - 1,
        key=f"{key_prefix}-later-{card.card_id}-{nonce}",
    ):
        updated = move_dashboard_card(config, card.card_id, 1)
        commit_dashboard_config(dataframe, state, updated, reset_widgets=True)
        st.rerun()
    if controls[2].button(
        "Remove",
        disabled=len(config.cards) == 1,
        key=f"{key_prefix}-remove-{card.card_id}-{nonce}",
    ):
        updated = remove_dashboard_card(config, card.card_id)
        commit_dashboard_config(dataframe, state, updated, reset_widgets=True)
        st.rerun()


def render_dashboard_studio(
    dataframe: pd.DataFrame,
    dataset_key: str,
    filename: str,
) -> None:
    """Render the local visual dashboard composer and portable exports."""
    st.subheader("Dashboard Studio")
    st.caption(
        "Compose a focused dashboard with visual controls. Every filter and calculation runs locally, "
        "and every chart includes the exact table behind it."
    )
    state = get_dashboard_studio_state(dataframe, dataset_key)
    config = state["config"]
    assert isinstance(config, DashboardConfig)
    nonce = int(state.get("nonce", 0))
    key_prefix = f"dashboard-studio-{dataset_key}"

    dashboard_name = st.text_input(
        "Dashboard name",
        value=config.name,
        max_chars=80,
        key=f"{key_prefix}-name-{nonce}",
    ).strip()
    if dashboard_name and dashboard_name != config.name:
        proposed = replace(config, name=dashboard_name)
        try:
            commit_dashboard_config(dataframe, state, proposed)
            config = proposed
        except DashboardStudioError as exc:
            st.error(str(exc))

    config = render_dashboard_filters(
        dataframe, config, state, key_prefix
    )
    config = state["config"]
    assert isinstance(config, DashboardConfig)
    try:
        filtered = apply_dashboard_filters(dataframe, config.filters)
    except DashboardStudioError as exc:
        st.error(str(exc))
        return

    summary = st.columns(4)
    summary[0].metric("Rows in view", f"{len(filtered):,}", delta=f"of {len(dataframe):,}")
    summary[1].metric("Cards", f"{len(config.cards):,}")
    summary[2].metric("Global filters", f"{len(config.filters):,}")
    summary[3].metric("Processing", "Local")

    if filtered.empty:
        st.warning(
            "The active filters return no rows. KPI row counts remain accurate; loosen a filter to restore charts."
        )

    st.markdown("#### Dashboard canvas")
    canvas_columns = st.columns(2)
    for index, card in enumerate(config.cards):
        with canvas_columns[index % 2]:
            with st.container(border=True):
                render_dashboard_result_card(
                    filtered,
                    config,
                    card,
                    index,
                    state,
                    key_prefix,
                )

    with st.expander(
        f"Add a card · {len(config.cards)}/12",
        expanded=False,
    ):
        if len(config.cards) >= 12:
            st.info("This dashboard has reached the 12-card clarity limit.")
        else:
            draft = DashboardCard("draft", "kpi", "New KPI")
            proposed_draft = render_dashboard_card_editor(
                dataframe,
                draft,
                f"{key_prefix}-new-card-{nonce}",
            )
            if st.button(
                "Add card",
                type="primary",
                key=f"{key_prefix}-add-card-{nonce}",
            ):
                new_card = replace(
                    proposed_draft,
                    card_id=next_dashboard_id(config, "card"),
                )
                updated = replace(config, cards=(*config.cards, new_card))
                try:
                    commit_dashboard_config(
                        dataframe, state, updated, reset_widgets=True
                    )
                    st.rerun()
                except DashboardStudioError as exc:
                    st.error(str(exc))

    st.markdown("#### Save and share")
    st.caption(
        "Configuration JSON stores the layout and filters, not the dataset. The HTML export contains "
        "the filtered dashboard results so it can open without Python or an internet connection."
    )
    stem = safe_download_stem(config.name)
    action_columns = st.columns(3)
    action_columns[0].download_button(
        "Download layout JSON",
        data=dashboard_config_to_json(config).encode("utf-8"),
        file_name=f"{stem}.json",
        mime="application/json",
        key=f"{key_prefix}-download-json",
    )
    if action_columns[1].button(
        "Reset starter layout",
        key=f"{key_prefix}-reset-{nonce}",
    ):
        commit_dashboard_config(
            dataframe,
            state,
            default_dashboard_config(dataframe),
            reset_widgets=True,
        )
        st.rerun()

    signature = sha256(
        f"{dataset_key}:{dashboard_config_to_json(config)}".encode()
    ).hexdigest()
    if action_columns[2].button(
        "Prepare offline HTML",
        type="primary",
        key=f"{key_prefix}-prepare-html-{nonce}",
    ):
        with st.spinner("Building the standalone dashboard…"):
            state["html_export"] = {
                "signature": signature,
                "data": build_dashboard_html(dataframe, config).encode("utf-8"),
            }

    html_export = state.get("html_export")
    if isinstance(html_export, dict) and html_export.get("signature") == signature:
        st.download_button(
            "Download offline dashboard HTML",
            data=html_export["data"],
            file_name=f"{stem}.html",
            mime="text/html",
            type="primary",
            key=f"{key_prefix}-download-html",
        )

    uploaded_config = st.file_uploader(
        "Load a saved dashboard layout",
        type=["json"],
        help="ODS validates file size, format, calculations, filters, and every referenced column before loading.",
        key=f"{key_prefix}-upload-config-{nonce}",
    )
    if st.button(
        "Load configuration",
        disabled=uploaded_config is None,
        key=f"{key_prefix}-load-config-{nonce}",
    ) and uploaded_config is not None:
        try:
            loaded = dashboard_config_from_json(
                uploaded_config.getvalue().decode("utf-8"), dataframe
            )
            commit_dashboard_config(
                dataframe, state, loaded, reset_widgets=True
            )
            st.rerun()
        except (DashboardStudioError, UnicodeDecodeError) as exc:
            st.error(str(exc))


def render_guided_dashboard(dataframe: pd.DataFrame, dataset_key: str) -> None:
    """Render the preserved role-reviewed recommendation workflow."""
    st.subheader("Smart guided dashboard")
    st.caption(
        "ODS recommends a small dashboard, but you stay in control. Review the inferred roles, "
        "choose the question, and verify the exact calculation behind every chart."
    )

    role_review = build_role_review(dataframe)
    low_confidence = any(
        semantic.confidence < 0.9
        for semantic in infer_column_semantics(dataframe)
    )
    with st.expander("1. Review column roles", expanded=low_confidence):
        st.caption(
            "Correct anything ODS misunderstood. IDs are excluded from measures, free text is not charted, "
            "and ignored columns are left out entirely."
        )
        reviewed_table = st.data_editor(
            role_review,
            width="stretch",
            hide_index=True,
            disabled=["Column", "Format", "Confidence", "Why ODS chose it"],
            column_config={
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    options=list(ROLE_LABELS.values()),
                    required=True,
                ),
                "Why ODS chose it": st.column_config.TextColumn(width="large"),
            },
            key=f"role-review-{dataset_key}",
        )

    st.markdown("#### 2. Choose the question")
    controls = st.columns([2, 1, 1, 1])
    intent = controls[0].selectbox(
        "Dashboard goal",
        INTENTS,
        help="This determines which chart families ODS recommends.",
        key=f"guided-intent-{dataset_key}",
    )
    aggregation_label = controls[1].selectbox(
        "Aggregation",
        list(AGGREGATIONS),
        help="Used for category comparisons and time trends.",
        key=f"guided-aggregation-{dataset_key}",
    )
    date_grain = controls[2].selectbox(
        "Date grain",
        ["Day", "Week", "Month", "Quarter", "Year"],
        index=2,
        help="Used when a chart groups records over time.",
        key=f"guided-grain-{dataset_key}",
    )
    chart_count = controls[3].slider(
        "Charts",
        min_value=1,
        max_value=4,
        value=4,
        key=f"guided-count-{dataset_key}",
    )

    role_mapping = {
        str(row["Column"]): ROLE_VALUES[str(row["Role"])]
        for row in reviewed_table.to_dict(orient="records")
    }
    reviewed_roles = roles_from_mapping(dataframe, role_mapping)
    suggestions = suggest_dashboard(
        dataframe,
        max_charts=chart_count,
        roles=reviewed_roles,
        intent=intent,
        aggregation=AGGREGATIONS[aggregation_label],
        date_grain=date_grain.lower(),
    )

    st.markdown("#### 3. Recommended dashboard")
    if not suggestions:
        st.info(
            "ODS could not make a safe recommendation for this goal with the reviewed roles. "
            "Try another goal or correct a column role above."
        )
    dashboard_columns = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        with dashboard_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {suggestion.title}")
                st.caption(suggestion.explanation)
                render_chart(dataframe, suggestion)


st.set_page_config(
    page_title="Open Data Scientist",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #07111f; color: #edf7ff; }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .92); }
    .ods-kicker { color: #65dfff; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
    .ods-title { font-size: clamp(2.6rem, 6vw, 5.2rem); line-height: .96; letter-spacing: -.06em; margin: .7rem 0 1rem; }
    .ods-title span { color: #56c9ff; }
    .ods-subtitle { color: #91a9bd; max-width: 760px; font-size: 1.05rem; line-height: 1.7; }
    [data-testid="stFileUploader"] { border: 1px solid rgba(101, 209, 255, .22); border-radius: 16px; padding: 1rem; background: #0b1b2d; }
    [data-testid="stMetric"] { border: 1px solid rgba(101, 209, 255, .17); border-radius: 14px; padding: 1rem; background: #0b1b2d; }
    .ods-score { border: 1px solid rgba(101, 209, 255, .22); border-radius: 16px; padding: 1.15rem; background: #0b1b2d; }
    .ods-score strong { color: #6ee7ff; font-size: 2rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: rgba(101, 209, 255, .17); background: #0b1b2d; }
    </style>
    <div class="ods-kicker">OPEN-SOURCE · LOCAL-FIRST · NO PAID API · {release}</div>
    <h1 class="ods-title">Turn raw files into a <span>clear data story.</span></h1>
    <p class="ods-subtitle">Upload a CSV or Excel file. ODS profiles, cleans, visualizes, and queries the dataset with transparent local rules and downloadable evidence.</p>
    """.replace("{release}", __release__),
    unsafe_allow_html=True,
)

st.info(
    "**Privacy before you upload:** on the public demo, your file is transferred to "
    "the hosted Streamlit session and processed in memory. ODS does not send it to an "
    "AI model or paid API. Run ODS locally for confidential or regulated data.",
    icon="🔒",
)

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx", "xls"],
    help=(
        "ODS processes the upload in memory and does not require a paid API. "
        f"Release-candidate limit: {DEFAULT_LIMITS.max_upload_bytes // (1024 * 1024)} MB, "
        f"{DEFAULT_LIMITS.max_rows:,} rows, and {DEFAULT_LIMITS.max_columns:,} columns."
    ),
)

sample_controls = st.columns([1, 4])
sample_requested = sample_controls[0].button(
    "Try sample dataset",
    type="primary",
    width="stretch",
    disabled=uploaded_file is not None,
    key="try-sample-dataset",
)
sample_controls[1].caption(
    "No file ready? Load the included customer dataset in one click, then follow the guided walkthrough."
)
if sample_requested:
    st.session_state["ods-use-sample-dataset"] = True

source_is_sample = False
if uploaded_file is not None:
    source_name = Path(uploaded_file.name).name
    source_bytes = uploaded_file.getvalue()
    st.session_state["ods-use-sample-dataset"] = False
elif st.session_state.get("ods-use-sample-dataset", False):
    source_is_sample = True
    source_name = SAMPLE_DATA_PATH.name
    try:
        source_bytes = SAMPLE_DATA_PATH.read_bytes()
    except OSError:
        st.error("The bundled sample dataset is unavailable in this installation.")
        st.stop()
else:
    st.info("Upload a CSV or Excel file, or try the included sample, to start the walkthrough.")
    st.stop()

try:
    original_dataframe = load_dataset(source_name, source_bytes)
except DatasetLoadError as exc:
    st.error(str(exc))
    st.stop()

source_sha256 = sha256(source_bytes).hexdigest()
source_key = source_sha256[:12]
cleaning_state = get_cleaning_state(source_key)
cleaning_batches = cleaning_state.get("batches", [])
if not isinstance(cleaning_batches, list):
    cleaning_batches = []
    cleaning_state["batches"] = cleaning_batches
try:
    dataframe = replay_cleaning_batches(original_dataframe, cleaning_batches)
except CleaningError as exc:
    st.error(f"The saved cleaning history could not be replayed: {exc}")
    cleaning_batches.clear()
    cleaning_state["preview"] = None
    dataframe = original_dataframe.copy(deep=True)

try:
    profile = profile_dataset(dataframe)
except MemoryError:
    st.error(
        "The dataset exceeded available memory during profiling. Reduce its size or run ODS locally with more memory."
    )
    st.stop()
active_dataset_key = (
    f"{source_key}-{cleaning_history_fingerprint(cleaning_batches)}"
)

st.success(f"Loaded {source_name} successfully.")
if source_is_sample:
    with st.expander("Three-minute sample walkthrough", expanded=True):
        st.markdown(
            "1. Open **Data → Quality** to see explainable checks.  \n"
            "2. Open **Dashboard** to explore the starter cards and calculation evidence.  \n"
            "3. Open **Visual SQL** to build a query with controls, then review it in **SQL Coach**.  \n"
            "Every result can be reproduced locally without a paid API."
        )
if cleaning_batches:
    applied_count = sum(len(batch) for batch in cleaning_batches)
    st.caption(
        f"The working dataset includes {applied_count:,} applied cleaning "
        f"fix{'es' if applied_count != 1 else ''}. Reset remains available in Clean data."
    )
metric_columns = st.columns(6)
metrics = [
    ("Rows", f"{profile.rows:,}"),
    ("Columns", f"{profile.columns:,}"),
    ("Memory", format_bytes(profile.memory_bytes)),
    ("Missing cells", f"{profile.missing_cells:,}"),
    ("Duplicates", f"{profile.duplicate_rows:,}"),
    ("Quality score", f"{profile.health_score}/100"),
]
for container, (label, value) in zip(metric_columns, metrics, strict=True):
    container.metric(label, value)

data_tab, cleaning_tab, dashboard_tab, query_tab, coach_tab = st.tabs(
    ["Data", "Clean", "Dashboard", "Visual SQL", "SQL Coach"]
)

with data_tab:
    preview_tab, quality_tab, columns_tab, statistics_tab = st.tabs(
        ["Preview", "Quality", "Columns", "Statistics"]
    )
    with preview_tab:
        st.subheader("Data preview")
        st.dataframe(dataframe.head(100), width="stretch", hide_index=True)
        st.caption(
            "Showing up to the first 100 rows from the current working dataset."
        )
    with quality_tab:
        st.subheader("Automatic quality findings")
        if not profile.issues:
            st.success("No automatic quality warnings were detected.")
        for issue in profile.issues:
            message = f"**{issue.title}**"
            if issue.column:
                message += f" · `{issue.column}`"
            message += f" — {issue.detail}"
            if issue.severity == "critical":
                st.error(message)
            elif issue.severity == "warning":
                st.warning(message)
            else:
                st.info(message)
    with columns_tab:
        st.subheader("Column profile")
        st.dataframe(
            profile.column_profile, width="stretch", hide_index=True
        )
    with statistics_tab:
        st.subheader("Numeric statistics")
        if profile.numeric_summary.empty:
            st.info("No numeric columns were detected.")
        else:
            st.dataframe(profile.numeric_summary, width="stretch")

    report = build_markdown_report(source_name, profile)
    st.download_button(
        "Download quality report",
        data=report,
        file_name=f"{source_name.rsplit('.', 1)[0]}-quality-report.md",
        mime="text/markdown",
        type="primary",
        key=f"quality-report-{active_dataset_key}",
    )

with cleaning_tab:
    render_cleaning_studio(
        original_dataframe,
        dataframe,
        cleaning_state,
        source_key,
        source_sha256,
        source_name,
    )

with dashboard_tab:
    dashboard_mode = st.radio(
        "Dashboard mode",
        ["Dashboard Studio", "Guided recommendations"],
        horizontal=True,
        key=f"dashboard-mode-{active_dataset_key}",
    )
    if dashboard_mode == "Dashboard Studio":
        render_dashboard_studio(
            dataframe,
            active_dataset_key,
            source_name,
        )
    else:
        render_guided_dashboard(dataframe, active_dataset_key)

with query_tab:
    render_visual_sql_builder(
        dataframe,
        active_dataset_key,
        source_name,
    )

with coach_tab:
    render_sql_coach(
        dataframe,
        active_dataset_key,
    )
