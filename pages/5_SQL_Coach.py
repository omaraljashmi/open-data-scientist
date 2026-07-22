"""SQL Coach — local query explanation, linting, and EXPLAIN plan page.

Session-state contract (written by app.py):
    original_df                     pd.DataFrame
    dataset_key                     str
    sql-coach-latest-{scoped_key}   str   (written by 4_Visual_SQL.py, optional)
"""

from __future__ import annotations

from hashlib import sha256

import pandas as pd
import streamlit as st

from ods import (
    SqlCoachError,
    analyze_query,
    replay_cleaning_batches,
)
from app_shared import cleaning_history_fingerprint, get_cleaning_state

st.set_page_config(page_title="ODS · SQL Coach", layout="wide")

# ── Guard ─────────────────────────────────────────────────────────────────────
original_df = st.session_state.get("original_df")
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
scoped_key = f"{dataset_key}-{history_key}"

# ── Page header ───────────────────────────────────────────────────────────────
st.subheader("SQL Coach")
st.caption(
    "Understand a DuckDB query before running it. The coach parses the SQL locally, "
    "checks correctness and performance risks, and asks DuckDB for the real physical plan."
)

# ── Query source ──────────────────────────────────────────────────────────────
latest_sql   = st.session_state.get(f"sql-coach-latest-{scoped_key}")
source_opts  = ["Paste or edit SQL"]
if latest_sql:
    source_opts.insert(0, "Latest Visual SQL query")

source = st.radio(
    "Query source", source_opts, horizontal=True,
    key=f"sql-coach-source-{scoped_key}",
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
        key=f"sql-coach-input-{scoped_key}",
    )

st.info(
    "Safety boundary: only one read-only query against `uploaded_data` is analyzed. "
    "The coach blocks writes, database commands, other tables, and external file readers."
)

# ── Analyze ───────────────────────────────────────────────────────────────────
# analyze_query(df, sql) -> SqlAnalysis
# SqlAnalysis fields:
#   .score          int
#   .findings       list[Finding]   each: .title / .category / .detail / .recommendation / .severity
#   .plan_steps     list[PlanStep]  each: .operator / .explanation
#   .clauses        list[Clause]    each: .clause / .explanation
#   .suggested_sql  str
#   .formatted_sql  str
#   .physical_plan  str             raw EXPLAIN output

signature = sha256(f"{scoped_key}:{sql}".encode()).hexdigest()
state_key = f"sql-coach-result-{scoped_key}"

if st.button("Analyze query", type="primary", key=f"sql-coach-run-{scoped_key}"):
    try:
        analysis = analyze_query(df, sql)
        st.session_state[state_key] = {"signature": signature, "analysis": analysis}
    except SqlCoachError as exc:
        st.session_state.pop(state_key, None)
        st.error(str(exc))

saved = st.session_state.get(state_key)
if not saved:
    st.stop()

if saved["signature"] != signature:
    st.info("The SQL changed. Analyze it again to refresh the explanation and plan.")
    st.stop()

analysis = saved["analysis"]

# ── Score tiles ───────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Query score",    f"{analysis.score}/100")
m2.metric("Findings",       f"{len(analysis.findings):,}")
m3.metric("Plan operators", f"{len(analysis.plan_steps):,}")

st.success("Read-only validation passed. DuckDB planned this query without executing its result.")

# ── Clause explanations ───────────────────────────────────────────────────────
st.markdown("#### What the query does")
clause_cols = st.columns(2)
for i, clause in enumerate(analysis.clauses):
    with clause_cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"**{clause.clause}**")
            st.write(clause.explanation)

# ── Findings ──────────────────────────────────────────────────────────────────
st.markdown("#### Optimization and correctness review")

if not analysis.findings:
    st.success("No rule-based risks detected for this dataset and query shape.")

for finding in analysis.findings:
    msg = (
        f"**{finding.title}** · {finding.category}\n\n"
        f"{finding.detail}\n\n"
        f"**Recommendation:** {finding.recommendation}"
    )
    if finding.severity == "high":
        st.error(msg)
    elif finding.severity == "medium":
        st.warning(msg)
    else:
        st.info(msg)

# ── Clean rewrite ─────────────────────────────────────────────────────────────
st.markdown("#### Clean DuckDB rewrite")
st.code(analysis.suggested_sql, language="sql")

if analysis.suggested_sql != analysis.formatted_sql:
    st.caption(
        "This rewrite expands an unambiguous top-level `SELECT *` using the current "
        "upload. Other recommendations are not applied automatically when they could "
        "change results."
    )
else:
    st.caption(
        "The query is formatted consistently. Recommendations that could change "
        "results remain advice instead of being applied silently."
    )

st.download_button(
    "Download clean SQL",
    data=analysis.suggested_sql.encode("utf-8"),
    file_name="ods-clean-query.sql",
    mime="text/plain",
    key=f"sql-coach-download-{scoped_key}",
)

# ── Physical plan ─────────────────────────────────────────────────────────────
st.markdown("#### DuckDB physical plan")

if analysis.plan_steps:
    st.dataframe(
        [{"Operator": s.operator, "What it does": s.explanation}
         for s in analysis.plan_steps],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Raw EXPLAIN plan"):
        st.code(analysis.physical_plan, language="text")
    st.caption(
        "The plan is generated by DuckDB. Optimization findings are transparent "
        "local rules, not guarantees from an AI model."
    )
