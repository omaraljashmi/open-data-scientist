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
    DatasetLoadError,
    FilterRule,
    INTENTS,
    QueryBuilderError,
    QuerySpec,
    build_chart_data,
    build_markdown_report,
    build_query,
    execute_query,
    infer_column_semantics,
    load_dataset,
    profile_dataset,
    roles_from_mapping,
    suggest_dashboard,
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
    <p class="ods-subtitle">Upload a CSV or Excel file. ODS profiles the dataset, identifies quality risks, summarizes its structure, and creates a downloadable report—all locally.</p>
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
    dataframe = load_dataset(uploaded_file.name, uploaded_bytes)
    profile = profile_dataset(dataframe)
except DatasetLoadError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Loaded {uploaded_file.name} successfully.")
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

overview_tab, dashboard_tab, query_tab, columns_tab, quality_tab, statistics_tab = st.tabs(
    [
        "Data preview",
        "Smart dashboard",
        "Visual SQL",
        "Column profile",
        "Quality findings",
        "Statistics",
    ]
)

with overview_tab:
    st.subheader("Data preview")
    st.dataframe(dataframe.head(100), use_container_width=True, hide_index=True)
    st.caption("Showing up to the first 100 rows.")

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
            key=f"role-review-{sha256(uploaded_bytes).hexdigest()[:12]}",
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
        sha256(uploaded_bytes).hexdigest()[:12],
        uploaded_file.name,
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
