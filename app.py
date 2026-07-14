"""Streamlit interface for Open Data Scientist."""

from __future__ import annotations

from hashlib import sha256

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from ods import (
    AggregateRule,
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
    build_chart_data,
    build_cleaning_recipe,
    build_markdown_report,
    build_query,
    execute_query,
    infer_column_semantics,
    load_dataset,
    profile_dataset,
    replay_cleaning_batches,
    roles_from_mapping,
    suggest_dashboard,
    suggest_cleaning_actions,
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
        st.dataframe(chart_data, use_container_width=True, hide_index=True)
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
        st.dataframe(result_frame.head(500), use_container_width=True, hide_index=True)
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
            use_container_width=True,
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
            use_container_width=True,
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
            st.dataframe(current.head(12), use_container_width=True, hide_index=True)
        with sample_columns[1]:
            st.markdown("**After · first 12 rows**")
            st.dataframe(preview_frame.head(12), use_container_width=True, hide_index=True)

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
    <div class="ods-kicker">OPEN-SOURCE · LOCAL-FIRST · NO PAID API</div>
    <h1 class="ods-title">Turn raw files into a <span>clear data story.</span></h1>
    <p class="ods-subtitle">Upload a CSV or Excel file. ODS profiles, cleans, visualizes, and queries the dataset with transparent local rules and downloadable evidence.</p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx", "xls"],
    help="ODS processes the upload in memory and does not require a paid API.",
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to generate your first automated profile.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.getvalue()
    original_dataframe = load_dataset(uploaded_file.name, uploaded_bytes)
except DatasetLoadError as exc:
    st.error(str(exc))
    st.stop()

source_sha256 = sha256(uploaded_bytes).hexdigest()
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

profile = profile_dataset(dataframe)
active_dataset_key = (
    f"{source_key}-{cleaning_history_fingerprint(cleaning_batches)}"
)

st.success(f"Loaded {uploaded_file.name} successfully.")
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

overview_tab, cleaning_tab, dashboard_tab, query_tab, coach_tab, columns_tab, quality_tab, statistics_tab = st.tabs(
    [
        "Data preview",
        "Clean data",
        "Smart dashboard",
        "Visual SQL",
        "SQL Coach",
        "Column profile",
        "Quality findings",
        "Statistics",
    ]
)

with overview_tab:
    st.subheader("Data preview")
    st.dataframe(dataframe.head(100), use_container_width=True, hide_index=True)
    st.caption("Showing up to the first 100 rows from the current working dataset.")

with cleaning_tab:
    render_cleaning_studio(
        original_dataframe,
        dataframe,
        cleaning_state,
        source_key,
        source_sha256,
        uploaded_file.name,
    )

with dashboard_tab:
    st.subheader("Smart guided dashboard")
    st.caption(
        "ODS recommends a small dashboard, but you stay in control. Review the inferred roles, "
        "choose the question, and verify the exact calculation behind every chart."
    )

    role_review = build_role_review(dataframe)
    low_confidence = any(semantic.confidence < 0.9 for semantic in infer_column_semantics(dataframe))
    with st.expander("1. Review column roles", expanded=low_confidence):
        st.caption(
            "Correct anything ODS misunderstood. IDs are excluded from measures, free text is not charted, "
            "and ignored columns are left out entirely."
        )
        reviewed_table = st.data_editor(
            role_review,
            use_container_width=True,
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
            key=f"role-review-{active_dataset_key}",
        )

    st.markdown("#### 2. Choose the question")
    controls = st.columns([2, 1, 1, 1])
    intent = controls[0].selectbox(
        "Dashboard goal",
        INTENTS,
        help="This determines which chart families ODS recommends.",
    )
    aggregation_label = controls[1].selectbox(
        "Aggregation",
        list(AGGREGATIONS),
        help="Used for category comparisons and time trends.",
    )
    date_grain = controls[2].selectbox(
        "Date grain",
        ["Day", "Week", "Month", "Quarter", "Year"],
        index=2,
        help="Used when a chart groups records over time.",
    )
    chart_count = controls[3].slider("Charts", min_value=1, max_value=4, value=4)

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

with query_tab:
    render_visual_sql_builder(
        dataframe,
        active_dataset_key,
        uploaded_file.name,
    )

with coach_tab:
    render_sql_coach(
        dataframe,
        active_dataset_key,
    )

with columns_tab:
    st.subheader("Column profile")
    st.dataframe(profile.column_profile, use_container_width=True, hide_index=True)

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

with statistics_tab:
    st.subheader("Numeric statistics")
    if profile.numeric_summary.empty:
        st.info("No numeric columns were detected.")
    else:
        st.dataframe(profile.numeric_summary, use_container_width=True)

report = build_markdown_report(uploaded_file.name, profile)
st.download_button(
    "Download quality report",
    data=report,
    file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}-quality-report.md",
    mime="text/markdown",
    type="primary",
)
