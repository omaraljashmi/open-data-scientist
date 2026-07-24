"""Data Insight Studio — landing page and session-state loader.

Runs inside the ``st.navigation`` router defined in ``app.py``, which owns
``st.set_page_config`` for every page.
"""

from __future__ import annotations

from hashlib import sha256

import streamlit as st

# ---------------------------------------------------------------------------
# NOTE: all ods imports at the top so a missing dependency surfaces immediately
# ---------------------------------------------------------------------------
from ods import (
    DEFAULT_LIMITS,
    DatasetLoadError,
    __release__,
    load_dataset,
    profile_dataset,
    replay_cleaning_batches,
)
from app_shared import SAMPLE_DATA_PATH, apply_theme, format_bytes, get_cleaning_state

apply_theme()
st.markdown(
    f"""
    <div class="ods-hero">
      <div class="ods-kicker">Open-source · Local-first · No paid API · {__release__}</div>
      <h1>Turn raw files into a <span class="ods-accent">clear data story.</span></h1>
      <p>Upload a CSV or Excel file. Data Insight Studio profiles, cleans, visualizes, queries, and exports
      the dataset with transparent local rules and downloadable evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Privacy notice ────────────────────────────────────────────────────────────
with st.expander("Privacy notice", expanded=False):
    st.markdown(
        "Data Insight Studio never sends uploaded data to a paid API, and never sends it to any AI model "
        "by default. The optional AI chart advisor on the Profile page is off until you "
        "configure it, uses only endpoints you bring (a free tier or a local server), and "
        "sends **dataset metadata only** — column names, roles, and counts — never cell values. "
        "A local installation processes your data in your local Streamlit process. "
        "The public demo transfers it to a hosted Streamlit session — "
        "do not upload confidential, regulated, personal, or sensitive data there."
    )

# ── File upload ───────────────────────────────────────────────────────────────
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a CSV, XLSX, or XLS file",
        type=["csv", "xlsx", "xls"],
        help=(
            f"Max {DEFAULT_LIMITS.max_upload_bytes // (1024 * 1024)} MB · "
            f"{DEFAULT_LIMITS.max_rows:,} rows · "
            f"{DEFAULT_LIMITS.max_columns:,} columns"
        ),
    )

# One adaptive action slot: with no file it offers the sample; once a file
# is staged it becomes the green submit button. Nothing is processed until
# the user presses it (remove the staged file to get the sample back).
use_sample = False
submit_upload = False
with col_sample:
    st.write("")
    st.write("")
    if uploaded is None:
        use_sample = st.button("Try sample dataset", key="try-sample-dataset")
    else:
        submit_upload = st.button(
            "Analyze this file", type="primary", key="submit-upload"
        )
        st.caption(f"{uploaded.name} · {format_bytes(uploaded.size)}")

# ── Resolve source ────────────────────────────────────────────────────────────
raw_bytes: bytes | None = None
filename: str | None = None

if use_sample and SAMPLE_DATA_PATH.exists():
    raw_bytes = SAMPLE_DATA_PATH.read_bytes()
    filename = SAMPLE_DATA_PATH.name
elif submit_upload and uploaded is not None:
    raw_bytes = uploaded.read()
    filename = uploaded.name

# ── Load + profile (only when file changes) ───────────────────────────────────
if raw_bytes is not None and filename is not None:
    source_sha256 = sha256(raw_bytes).hexdigest()
    dataset_key = source_sha256[:16]

    if st.session_state.get("dataset_key") != dataset_key:
        try:
            # loader.py signature: load_dataset(name: str, data: bytes)
            original_df = load_dataset(filename, raw_bytes)
        except DatasetLoadError as exc:
            st.error(str(exc))
            st.stop()

        # Replay any cleaning already applied to this file
        cleaning_state = get_cleaning_state(dataset_key)
        batches = cleaning_state.get("batches", [])
        current_df = (
            replay_cleaning_batches(original_df, batches)
            if batches
            else original_df.copy()
        )

        # profile_dataset(df) — profiler.py signature
        profile = profile_dataset(current_df)

        # ── Write all shared session state ────────────────────────────────────
        # Pages read these keys; only this page and 2_Clean.py write them.
        st.session_state["original_df"]               = original_df
        st.session_state["current_df"]                = current_df
        st.session_state["filename"]                  = filename
        st.session_state["dataset_key"]               = dataset_key
        st.session_state["source_sha256"]             = source_sha256
        st.session_state[f"profile-{dataset_key}"]   = profile

# ── Summary banner (shown whenever a dataset is loaded in this session) ───────
_dk = st.session_state.get("dataset_key")
_profile = st.session_state.get(f"profile-{_dk}") if _dk else None

if _profile is not None:
    _fn = st.session_state.get("filename", "upload")
    st.success(
        f"Loaded **{_fn}** — "
        f"{_profile.rows:,} rows · "
        f"{_profile.columns:,} columns · "
        f"quality score **{_profile.health_score}/100**"
    )
    st.info("Use the sidebar to navigate: Profile · Clean · Dashboard · Visual SQL · SQL Coach")
else:
    st.info("Upload a file and press **Analyze this file**, or click **Try sample dataset** to begin.")
