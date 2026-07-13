"""Streamlit interface for Open Data Scientist."""

from __future__ import annotations

from hashlib import sha256

import pandas as pd
import streamlit as st

from ods import (
    ChartSuggestion,
    DatasetLoadError,
    INTENTS,
    build_chart_data,
    build_markdown_report,
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

overview_tab, dashboard_tab, columns_tab, quality_tab, statistics_tab = st.tabs(
    ["Data preview", "Smart dashboard", "Column profile", "Quality findings", "Statistics"]
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
