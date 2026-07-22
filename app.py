"""Open Data Scientist — landing page and session-state loader."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

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
from app_shared import SAMPLE_DATA_PATH, format_bytes, get_cleaning_state

st.set_page_config(
    page_title="Open Data Scientist",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal dark-theme CSS (kept from original) ───────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #1e2327; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Open Data Scientist")
st.caption(f"v{__release__} · local-first · no paid API")

# ── Privacy notice ────────────────────────────────────────────────────────────
with st.expander("Privacy notice", expanded=False):
    st.markdown(
        "ODS does not send uploaded data to a paid API or external AI model. "
        "A local installation processes it in your local Streamlit process. "
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

with col_sample:
    st.write("")
    st.write("")
    use_sample = st.button("Try sample dataset", key="try-sample-dataset")

# ── Resolve source ────────────────────────────────────────────────────────────
raw_bytes: bytes | None = None
filename: str | None = None

if use_sample and SAMPLE_DATA_PATH.exists():
    raw_bytes = SAMPLE_DATA_PATH.read_bytes()
    filename = SAMPLE_DATA_PATH.name
elif uploaded is not None:
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
    st.info("Upload a file above or click **Try sample dataset** to begin.")
