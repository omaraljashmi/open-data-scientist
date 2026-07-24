"""Visual SQL query builder page.

Session-state contract (written by app.py):
    original_df     pd.DataFrame
    filename        str
    dataset_key     str
"""

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
    FilterRule,
    QueryBuilderError,
    QuerySpec,
    build_query,
    execute_query,
    infer_column_semantics,
    replay_cleaning_batches,
)
from app_shared import (
    FILTER_OPERATORS,
    SUMMARY_CALCULATIONS,
    cleaning_history_fingerprint,
    default_aggregate_alias,
    filter_operator_labels,
    get_cleaning_state,
    render_page_header,
)

st.set_page_config(page_title="ODS · Visual SQL", layout="wide")
render_page_header(
    "Visual SQL query builder",
    "Choose columns, filters, summaries, and sorting with controls. ODS generates readable SQL "
    "and runs it only against this uploaded dataset in memory.",
)

# ── Guard ─────────────────────────────────────────────────────────────────────
original_df = st.session_state.get("original_df")
filename    = st.session_state.get("filename", "upload")
dataset_key = st.session_state.get("dataset_key", "")

if original_df is None:
    st.info("Upload a file on the home page to get started.")
    st.stop()

# ── Resolve current dataframe ─────────────────────────────────────────────────
state       = get_cleaning_state(dataset_key)
batches     = state.get("batches", [])
history_key = cleaning_history_fingerprint(batches)
df          = (
    replay_cleaning_batches(original_df, batches)
    if batches
    else original_df.copy()
)
scoped_key  = f"{dataset_key}-{history_key}"
key_prefix  = f"visual-sql-{scoped_key}"

column_names = [str(c) for c in df.columns]
if not column_names:
    st.info("This dataset has no columns to query.")
    st.stop()

if len(column_names) != len(set(column_names)):
    st.error(
        "Visual SQL needs unique column names. "
        "Rename duplicate columns and upload the file again."
    )
    st.stop()

# ── Mode ──────────────────────────────────────────────────────────────────────
mode = st.radio(
    "What do you want to create?",
    ["View and filter rows", "Create a summary"],
    horizontal=True,
    key=f"{key_prefix}-mode",
)

selected_columns: tuple[str, ...] = ()
group_by:         tuple[str, ...] = ()
aggregates:       tuple[AggregateRule, ...] = ()
output_names:     list[str]

if mode == "View and filter rows":
    chosen = st.multiselect(
        "Columns to show",
        column_names,
        default=column_names,
        help="The result keeps the same row-level detail as the upload.",
        key=f"{key_prefix}-columns",
    )
    selected_columns = tuple(chosen)
    output_names     = list(chosen)

else:
    semantics = infer_column_semantics(df)
    suggested_group = next(
        (s.column for s in semantics if s.role in {"categorical", "datetime"}),
        None,
    )
    chosen_groups = st.multiselect(
        "Group results by",
        column_names,
        default=[suggested_group] if suggested_group else [],
        help="Leave empty to calculate one total for the full dataset.",
        key=f"{key_prefix}-groups",
    )
    group_by = tuple(chosen_groups)

    numeric_cols = [c for c in column_names if is_numeric_dtype(df[c])]
    calc_labels  = ["Count rows", "Count non-missing"]
    if numeric_cols:
        calc_labels.extend(["Average", "Median", "Total", "Minimum", "Maximum"])

    calc_label = st.selectbox(
        "Summary calculation", calc_labels, key=f"{key_prefix}-calculation"
    )
    function = SUMMARY_CALCULATIONS[calc_label]

    source_column: str | None = None
    if function != "count_rows":
        candidates   = column_names if function == "count_values" else numeric_cols
        source_column = st.selectbox(
            "Column to summarize", candidates,
            key=f"{key_prefix}-summary-column",
        )

    default_alias = default_aggregate_alias(function, source_column)
    alias = st.text_input(
        "Result column name",
        value=default_alias,
        key=f"{key_prefix}-alias-{function}-{source_column or 'rows'}",
    ).strip()

    aggregates   = (AggregateRule(function=function, column=source_column, alias=alias),)
    output_names = [*chosen_groups, alias]

# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("#### Optional filters")
filter_count = st.slider(
    "Number of filters", min_value=0, max_value=3, value=0,
    help="Filters are combined with AND.",
    key=f"{key_prefix}-filter-count",
)

filters: list[FilterRule] = []
for idx in range(filter_count):
    col_c, op_c, val_c = st.columns([1.2, 1.2, 1.6])
    filter_column = col_c.selectbox(
        f"Filter {idx+1} column", column_names,
        key=f"{key_prefix}-filter-column-{idx}",
    )
    op_label = op_c.selectbox(
        f"Filter {idx+1} rule",
        filter_operator_labels(df[filter_column]),
        key=f"{key_prefix}-filter-operator-{idx}",
    )
    operator     = FILTER_OPERATORS[op_label]
    filter_value = None

    if operator in {"is_null", "is_not_null"}:
        val_c.caption("No value needed for this rule.")
    elif is_bool_dtype(df[filter_column]):
        filter_value = val_c.selectbox(
            f"Filter {idx+1} value", [True, False],
            key=f"{key_prefix}-filter-value-{idx}",
        )
    elif is_datetime64_any_dtype(df[filter_column]):
        non_null = pd.to_datetime(df[filter_column], errors="coerce").dropna()
        default_date = (
            non_null.iloc[0].date() if not non_null.empty
            else pd.Timestamp.today().date()
        )
        filter_value = val_c.date_input(
            f"Filter {idx+1} value", value=default_date,
            key=f"{key_prefix}-filter-value-{idx}",
        )
    else:
        filter_value = val_c.text_input(
            f"Filter {idx+1} value",
            help="Numeric values may include commas, such as 1,000.",
            key=f"{key_prefix}-filter-value-{idx}",
        )

    filters.append(FilterRule(filter_column, operator, filter_value))

# ── Sort + limit ──────────────────────────────────────────────────────────────
st.markdown("#### Sort and limit")
sort_c, dir_c, lim_c = st.columns([1.4, 1, 1])

sort_label = sort_c.selectbox(
    "Sort results by", ["No sorting", *output_names],
    key=f"{key_prefix}-sort-column",
)
direction_label = dir_c.selectbox(
    "Direction", ["Ascending", "Descending"],
    disabled=(sort_label == "No sorting"),
    key=f"{key_prefix}-sort-direction",
)
limit = int(lim_c.number_input(
    "Maximum rows", min_value=1, max_value=5000, value=100, step=25,
    key=f"{key_prefix}-limit",
))

# ── Build QuerySpec ───────────────────────────────────────────────────────────
spec = QuerySpec(
    selected_columns=selected_columns,
    filters=tuple(filters),
    group_by=group_by,
    aggregates=aggregates,
    sort_by=None if sort_label == "No sorting" else sort_label,
    sort_descending=(direction_label == "Descending"),
    limit=limit,
)

# ── Generated SQL preview ─────────────────────────────────────────────────────
query = None
try:
    query = build_query(df, spec)
except QueryBuilderError as exc:
    st.error(str(exc))

if query is not None:
    # Store latest SQL so SQL Coach page can pick it up
    st.session_state[f"sql-coach-latest-{scoped_key}"] = query.display_sql

    with st.expander("Generated SQL", expanded=False):
        st.code(query.display_sql, language="sql")
        st.caption(
            "ODS quotes column names and binds filter values separately before "
            "execution. The displayed SQL is a readable copy for learning and reuse."
        )

# ── Run query ─────────────────────────────────────────────────────────────────
signature  = sha256(f"{scoped_key}:{spec!r}".encode()).hexdigest()
state_key  = f"{key_prefix}-result"

if st.button(
    "Run query",
    type="primary",
    disabled=(query is None),
    key=f"{key_prefix}-run",
) and query is not None:
    try:
        result = execute_query(df, spec)
        st.session_state[state_key] = {
            "signature": signature,
            "dataframe": result.dataframe,
        }
    except QueryBuilderError as exc:
        st.error(str(exc))

saved = st.session_state.get(state_key)
if saved and saved["signature"] == signature:
    result_df = saved["dataframe"]
    m1, m2    = st.columns(2)
    m1.metric("Result rows",    f"{len(result_df):,}")
    m2.metric("Result columns", f"{len(result_df.columns):,}")
    st.dataframe(result_df.head(500), width="stretch", hide_index=True)
    if len(result_df) > 500:
        st.caption("Previewing the first 500 rows; the download includes the full result.")
    download_name = filename.rsplit(".", 1)[0] + "-query-result.csv"
    st.download_button(
        "Download query result",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
        key=f"{key_prefix}-download",
    )
elif saved:
    st.info("The query controls changed. Run the query again to refresh the result.")
