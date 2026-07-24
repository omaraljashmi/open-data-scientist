"""Data profile, quality report, and guided dashboard page.

Session-state contract (written by app.py):
    current_df          pd.DataFrame   cleaned working copy
    filename            str            original upload name
    dataset_key         str            first 16 chars of source SHA-256
    profile-{key}       DatasetProfile profiler output for current_df
"""

from __future__ import annotations

import streamlit as st

from ods import (
    ADVISOR_PRESETS,
    INTENTS,                   # tuple[str] of intent names
    AdvisorError,
    build_markdown_report,     # (file_name: str, profile) -> str
    infer_column_semantics,    # (df) -> tuple[ColumnSemantic, ...]
    request_chart_advice,
    roles_from_mapping,        # (df, mapping: dict[str,str]) -> ColumnRoles
    suggest_dashboard,         # (df, *, roles, intent) -> tuple[ChartSuggestion, ...]
)
from app_shared import (
    ROLE_LABELS,
    ROLE_VALUES,
    cleaning_history_fingerprint,
    format_bytes,
    get_cleaning_state,
    render_chart,
    render_page_header,
)

st.set_page_config(page_title="ODS · Profile", layout="wide")
render_page_header(
    "Data profile",
    "Explainable quality checks, column semantics, and a guided dashboard — computed locally.",
)

# ── Guard: require a loaded dataset ──────────────────────────────────────────
df          = st.session_state.get("current_df")
filename    = st.session_state.get("filename", "upload")
dataset_key = st.session_state.get("dataset_key", "")
profile     = st.session_state.get(f"profile-{dataset_key}")

if df is None or profile is None:
    st.info("Upload a file on the home page to get started.")
    st.stop()

# Scope widget state to the file AND the cleaning history so edits do not
# leak across cleaning changes (e.g. after dropping a column).
history_key = cleaning_history_fingerprint(get_cleaning_state(dataset_key).get("batches", []))
scoped_key  = f"{dataset_key}-{history_key}"

# ── Summary tiles ─────────────────────────────────────────────────────────────
# DatasetProfile fields (profiler.py):
#   rows, columns, missing_cells, duplicate_rows, memory_bytes, health_score
cols = st.columns(6)
cols[0].metric("Rows",            f"{profile.rows:,}")
cols[1].metric("Columns",         f"{profile.columns:,}")
cols[2].metric("Missing cells",   f"{profile.missing_cells:,}")
cols[3].metric("Duplicate rows",  f"{profile.duplicate_rows:,}")
cols[4].metric("Memory",          format_bytes(profile.memory_bytes))
cols[5].metric("Quality score",   f"{profile.health_score}/100")

# ── Column detail ─────────────────────────────────────────────────────────────
# profile.column_profile is a ready-made DataFrame with columns:
#   column / dtype / missing_count / missing_percent / unique_count / examples
st.subheader("Column summary")
st.dataframe(profile.column_profile, width="stretch", hide_index=True)

# ── Quality issues ────────────────────────────────────────────────────────────
# profile.issues is a tuple[QualityIssue]; severities are critical/warning/info
if profile.issues:
    with st.expander(f"{len(profile.issues)} quality issue(s)"):
        for issue in profile.issues:
            msg = f"**{issue.title}**"
            if issue.column:
                msg += f" · `{issue.column}`"
            msg += f"\n\n{issue.detail}"
            if issue.severity == "critical":
                st.error(msg)
            elif issue.severity == "warning":
                st.warning(msg)
            else:
                st.info(msg)
else:
    st.success("No quality issues detected.")

# ── Numeric statistics ────────────────────────────────────────────────────────
# profile.numeric_summary is a ready-made DataFrame (may be empty for non-numeric uploads)
if profile.numeric_summary is not None and not profile.numeric_summary.empty:
    with st.expander("Numeric statistics"):
        st.dataframe(profile.numeric_summary, width="stretch")

# ── Download quality report ───────────────────────────────────────────────────
# build_markdown_report(file_name: str, profile: DatasetProfile) -> str
report_md = build_markdown_report(filename, profile)
st.download_button(
    "Download quality report (.md)",
    data=report_md.encode("utf-8"),
    file_name=f"{filename.rsplit('.', 1)[0]}-quality-report.md",
    mime="text/markdown",
    key=f"profile-report-download-{dataset_key}",
)

# ── Guided dashboard ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Smart guided dashboard")
st.caption(
    "ODS infers a semantic role for every column. Correct any wrong assignments, "
    "then pick a goal to get chart recommendations."
)

# Build editable role table from infer_column_semantics
# Each ColumnSemantic has: .column / .role / .display_format / .confidence / .reason
semantics = infer_column_semantics(df)
role_df = st.data_editor(
    [
        {
            "Column":          s.column,
            "Role":            ROLE_LABELS[s.role],
            "Format":          s.display_format,
            "Confidence":      f"{s.confidence:.0%}",
            "Why ODS chose it": s.reason,
        }
        for s in semantics
    ],
    column_config={
        "Role": st.column_config.SelectboxColumn(
            "Role", options=list(ROLE_LABELS.values())
        )
    },
    width="stretch",
    hide_index=True,
    key=f"role-editor-{scoped_key}",
)

# Map edited labels back to role keys, then get updated roles
role_mapping = {row["Column"]: ROLE_VALUES[row["Role"]] for row in role_df}
updated_roles = roles_from_mapping(df, role_mapping)

# INTENTS is a tuple[str] — selectbox directly, no .keys()
intent = st.selectbox(
    "What do you want to explore?",
    INTENTS,
    key=f"intent-{scoped_key}",
)

# suggest_dashboard(df, *, roles, intent) — roles and intent are keyword-only
suggestions = suggest_dashboard(df, roles=updated_roles, intent=intent)

if not suggestions:
    st.info("No suitable charts found for this goal and column combination.")
else:
    for suggestion in suggestions:
        # ChartSuggestion has .title and .explanation (not .reason)
        st.markdown(f"**{suggestion.title}** — {suggestion.explanation}")
        render_chart(df, suggestion)

# ── Optional AI chart advisor (zero-cost, bring your own endpoint) ────────────
st.divider()
with st.expander("AI chart advisor (optional)"):
    st.caption(
        "Off by default and never required. Bring your own free endpoint — a free-tier "
        "Google Gemini or Groq key, or a local Ollama server. ODS sends **only dataset "
        "metadata** (column names, roles, formats, unique and missing counts) — never any "
        "cell values — and validates every suggestion against your columns before charting."
    )
    preset_name = st.selectbox(
        "Provider preset",
        list(ADVISOR_PRESETS),
        key=f"advisor-preset-{scoped_key}",
    )
    preset = ADVISOR_PRESETS[preset_name]
    advisor_columns = st.columns(2)
    advisor_base = advisor_columns[0].text_input(
        "Base URL (OpenAI-compatible)",
        value=preset["base_url"],
        key=f"advisor-base-{preset_name}-{scoped_key}",
    ).strip()
    advisor_model = advisor_columns[1].text_input(
        "Model",
        value=preset["model"],
        key=f"advisor-model-{preset_name}-{scoped_key}",
    ).strip()
    advisor_key = st.text_input(
        "API key" + ("" if preset["needs_key"] else " (not needed for a local server)"),
        type="password",
        key=f"advisor-key-{scoped_key}",
    ).strip()
    st.caption("The key lives only in this session's memory — never stored or logged.")

    # A keyed preset with no key would only bounce with a 401 — disable the
    # button instead, and say exactly what to do about it.
    missing_key = bool(preset["needs_key"]) and not advisor_key
    if missing_key:
        st.caption(
            f"**{preset_name}** needs your free API key (press Enter after pasting it), "
            "or switch to the Local Ollama preset. Everything else in ODS works without a key."
        )

    advisor_state_key = f"advisor-result-{scoped_key}"
    if st.button("Ask the advisor", key=f"advisor-run-{scoped_key}", disabled=missing_key):
        try:
            with st.spinner("Asking the advisor…"):
                advice = request_chart_advice(
                    df,
                    intent=intent,
                    base_url=advisor_base,
                    model=advisor_model,
                    api_key=advisor_key or None,
                )
            st.session_state[advisor_state_key] = advice
        except AdvisorError as exc:
            st.session_state.pop(advisor_state_key, None)
            st.error(str(exc))

    advice = st.session_state.get(advisor_state_key)
    if advice:
        st.caption(
            "Advisor picks are validated against your columns and rendered with the same "
            "local, auditable calculations as every other ODS chart."
        )
        for suggestion in advice:
            st.markdown(f"**{suggestion.title}** — {suggestion.explanation}")
            render_chart(df, suggestion)
