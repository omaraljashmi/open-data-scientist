"""Streamlit interface for Open Data Scientist Milestone 1."""

from __future__ import annotations

import streamlit as st

from ods import DatasetLoadError, build_markdown_report, load_dataset, profile_dataset


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


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
    dataframe = load_dataset(uploaded_file.name, uploaded_file.getvalue())
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

overview_tab, columns_tab, quality_tab, statistics_tab = st.tabs(
    ["Data preview", "Column profile", "Quality findings", "Statistics"]
)

with overview_tab:
    st.subheader("Data preview")
    st.dataframe(dataframe.head(100), use_container_width=True, hide_index=True)
    st.caption("Showing up to the first 100 rows.")

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
