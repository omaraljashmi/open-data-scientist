"""Data Cleaning Studio page.

Session-state contract (written by app.py):
    original_df     pd.DataFrame   raw upload, never mutated
    filename        str
    dataset_key     str
    source_sha256   str            hex digest of raw upload bytes
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from ods import (
    CleaningError,
    apply_cleaning_actions,
    build_cleaning_recipe,
    profile_dataset,
    replay_cleaning_batches,
    suggest_cleaning_actions,
)
from app_shared import cleaning_history_fingerprint, get_cleaning_state
from hashlib import sha256

st.set_page_config(page_title="ODS · Clean", layout="wide")

# ── Guard ─────────────────────────────────────────────────────────────────────
original_df  = st.session_state.get("original_df")
filename     = st.session_state.get("filename", "upload")
dataset_key  = st.session_state.get("dataset_key", "")
source_sha256 = st.session_state.get("source_sha256", "")

if original_df is None:
    st.info("Upload a file on the home page to get started.")
    st.stop()

# ── Resolve current (post-cleaning) dataframe ─────────────────────────────────
state   = get_cleaning_state(dataset_key)
batches = state.get("batches", [])

current_df = (
    replay_cleaning_batches(original_df, batches)
    if batches
    else original_df.copy()
)

# ── Header metrics ────────────────────────────────────────────────────────────
st.subheader("Data Cleaning Studio")
st.caption(
    "Review deterministic fixes before they touch the working dataset. "
    "Every operation shows its evidence, requires an explicit preview and Apply, "
    "and remains reproducible from the source upload."
)

history_key     = cleaning_history_fingerprint(batches)
original_profile = profile_dataset(original_df)
current_profile  = profile_dataset(current_df)
applied_actions  = [action for batch in batches for action in batch]

# Keep the shared working copy in sync for the other pages. This also covers
# Undo/Reset, which mutate the batches and rerun this page from the top.
st.session_state["current_df"] = current_df
st.session_state[f"profile-{dataset_key}"] = current_profile

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
    "The score measures missingness, exact duplicates, and constant columns. "
    "A truthful type or format correction may improve trust without changing the score."
)

# ── Undo / Reset ──────────────────────────────────────────────────────────────
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
            st.markdown(f"{index}. **{action.title}**{location} — {action.evidence}")

# ── 1. Suggested fixes ────────────────────────────────────────────────────────
try:
    suggestions = suggest_cleaning_actions(current_df)
except CleaningError as exc:
    st.error(str(exc))
    suggestions = ()

st.markdown("#### 1. Review suggested fixes")

if not suggestions:
    st.success(
        "No conservative cleaning fixes remain. "
        "Domain-specific rules may still be needed."
    )
else:
    st.dataframe(
        [
            {
                "Confidence":       action.confidence.title(),
                "Suggested fix":    action.title,
                "Estimated impact": (
                    f"{action.affected_rows:,} rows "
                    f"({action.affected_percent:.1f}%)"
                ),
                "Evidence":         action.evidence,
            }
            for action in suggestions
        ],
        width="stretch",
        hide_index=True,
    )

    action_lookup = {action.action_id: action for action in suggestions}

    selected_ids = st.multiselect(
        "Choose fixes to preview",
        list(action_lookup),
        default=[],
        format_func=lambda aid: (
            f"{action_lookup[aid].confidence.title()} confidence · "
            f"{action_lookup[aid].title}"
        ),
        help=(
            "Nothing is selected automatically. "
            "Drop-column fixes cannot be combined with other fixes for the same column."
        ),
        key=f"cleaning-selection-{dataset_key}-{history_key}",
    )

    selected_actions = tuple(action_lookup[aid] for aid in selected_ids)
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
            key=f"cleaning-preview-{dataset_key}-{history_key}",
        ):
            try:
                preview_frame = apply_cleaning_actions(current_df, selected_actions)
                state["preview"] = {
                    "signature": preview_signature,
                    "dataframe": preview_frame,
                    "actions":   selected_actions,
                }
            except CleaningError as exc:
                state["preview"] = None
                st.error(str(exc))

    # ── 2. Before/after preview ───────────────────────────────────────────────
    preview = state.get("preview")

    if isinstance(preview, dict) and preview.get("signature") != preview_signature:
        st.info("The selected fixes changed. Preview them again before applying.")
        preview = None

    if isinstance(preview, dict):
        preview_frame   = preview["dataframe"]
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
            str(current_df.dtypes[col]) != str(preview_frame.dtypes[col])
            for col in current_df.columns.intersection(preview_frame.columns)
        )
        impact[3].metric("Changed data types", f"{changed_types:,}")

        sample_cols = st.columns(2)
        with sample_cols[0]:
            st.markdown("**Before · first 12 rows**")
            st.dataframe(current_df.head(12), width="stretch", hide_index=True)
        with sample_cols[1]:
            st.markdown("**After · first 12 rows**")
            st.dataframe(preview_frame.head(12), width="stretch", hide_index=True)

        st.warning(
            "Apply updates the working dataset used by the dashboard, Visual SQL, "
            "SQL Coach, profiles, statistics, and downloads. "
            "The original upload remains available through Reset."
        )

        if st.button(
            "Apply verified fixes",
            type="primary",
            key=f"cleaning-apply-{dataset_key}-{history_key}",
        ):
            batches.append(tuple(preview["actions"]))
            state["preview"] = None
            # Update shared current_df in session state
            new_current = replay_cleaning_batches(original_df, batches)
            st.session_state["current_df"] = new_current
            st.session_state[f"profile-{dataset_key}"] = profile_dataset(new_current)
            st.rerun()

# ── 3. Export ─────────────────────────────────────────────────────────────────
st.markdown("#### 3. Export the current result")
base_name    = filename.rsplit(".", 1)[0]
export_cols  = st.columns(2)

export_cols[0].download_button(
    "Download current cleaned CSV",
    data=current_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{base_name}-cleaned.csv",
    mime="text/csv",
    key=f"cleaning-download-data-{dataset_key}-{history_key}",
)

recipe = build_cleaning_recipe(
    filename,
    source_sha256,
    original_df,
    current_df,
    batches,
)
export_cols[1].download_button(
    "Download cleaning recipe",
    data=recipe.encode("utf-8"),
    file_name=f"{base_name}-cleaning-recipe.json",
    mime="application/json",
    key=f"cleaning-download-recipe-{dataset_key}-{history_key}",
)
st.caption(
    "The recipe records the source hash, ordered operations, parameters, "
    "evidence, and before/after schema."
)
