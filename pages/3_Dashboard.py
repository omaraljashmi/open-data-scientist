"""Dashboard Studio page.

Session-state contract (written by app.py):
    original_df     pd.DataFrame
    dataset_key     str
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st
from pandas.api.types import is_bool_dtype

from ods import (
    MAX_CARDS,
    MAX_FILTERS,
    DashboardCard,
    DashboardConfig,
    DashboardStudioError,
    apply_dashboard_filters,
    build_card_result,
    build_dashboard_html,
    categorical_filter_options,
    dashboard_config_from_json,
    dashboard_config_to_json,
    default_dashboard_config,
    default_filter_for_column,
    format_metric_value,
    infer_column_semantics,
    move_dashboard_card,
    next_dashboard_id,
    remove_dashboard_card,
    replace_dashboard_filter,
    replay_cleaning_batches,
    validate_dashboard_config,
)
from app_shared import (
    STUDIO_CARD_LABELS,
    STUDIO_CARD_VALUES,
    cleaning_history_fingerprint,
    get_cleaning_state,
    render_page_header,
    safe_download_stem,
)

st.set_page_config(page_title="ODS · Dashboard", layout="wide")
render_page_header(
    "Dashboard Studio",
    "Compose KPI and chart cards with global filters — every result carries the exact audit table behind it.",
)

# ── Guard ─────────────────────────────────────────────────────────────────────
original_df = st.session_state.get("original_df")
dataset_key = st.session_state.get("dataset_key", "")

if original_df is None:
    st.info("Upload a file on the home page to get started.")
    st.stop()

# ── Resolve current (post-cleaning) dataframe ─────────────────────────────────
state       = get_cleaning_state(dataset_key)
batches     = state.get("batches", [])
history_key = cleaning_history_fingerprint(batches)
current_df  = (
    replay_cleaning_batches(original_df, batches)
    if batches
    else original_df.copy()
)

# Scope all dashboard state to both the file AND the cleaning history
scoped_key = f"{dataset_key}-{history_key}"

# ── Dashboard Studio state ────────────────────────────────────────────────────
def _get_studio_state(df: pd.DataFrame, key: str) -> dict:
    sk    = f"dashboard-studio-state-{key}"
    saved = st.session_state.get(sk)
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), DashboardConfig):
        saved = {"config": default_dashboard_config(df), "nonce": 0, "html_export": None}
        st.session_state[sk] = saved
        return saved
    try:
        validate_dashboard_config(df, saved["config"])
    except DashboardStudioError:
        saved = {
            "config":      default_dashboard_config(df),
            "nonce":       int(saved.get("nonce", 0)) + 1,
            "html_export": None,
        }
        st.session_state[sk] = saved
    return saved


def _commit(df: pd.DataFrame, studio_state: dict, config: DashboardConfig, *, reset_widgets: bool = False) -> None:
    validate_dashboard_config(df, config)
    studio_state["config"]      = config
    studio_state["html_export"] = None
    if reset_widgets:
        studio_state["nonce"] = int(studio_state.get("nonce", 0)) + 1


studio_state = _get_studio_state(current_df, scoped_key)
config       = studio_state["config"]
nonce        = studio_state["nonce"]

# ── Column groups ─────────────────────────────────────────────────────────────
all_cols = [str(c) for c in current_df.columns]
numeric_cols = [
    c
    for c in all_cols
    if not is_bool_dtype(current_df[c])
    and pd.to_numeric(current_df[c], errors="coerce").notna().any()
]
date_cols = [
    s.column for s in infer_column_semantics(current_df) if s.role == "datetime"
]

# ── Global filter sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dashboard filters")
    used_filter_columns = {flt.column for flt in config.filters}
    addable_columns = [c for c in all_cols if c not in used_filter_columns]

    if len(config.filters) >= MAX_FILTERS:
        st.info(f"Maximum {MAX_FILTERS} filters reached.")
    elif addable_columns:
        filter_col = st.selectbox(
            "Add filter on column",
            addable_columns,
            key=f"ds-add-filter-col-{scoped_key}-{nonce}",
        )
        if st.button("Add filter", key=f"ds-add-filter-btn-{scoped_key}-{nonce}"):
            # default_filter_for_column(df, column, filter_id) — 3 args
            new_filter = default_filter_for_column(
                current_df, filter_col, next_dashboard_id(config, "filter")
            )
            try:
                _commit(
                    current_df,
                    studio_state,
                    replace(config, filters=(*config.filters, new_filter)),
                    reset_widgets=True,
                )
                st.rerun()
            except DashboardStudioError as exc:
                st.error(str(exc))

    for flt in config.filters:
        with st.expander(f"Filter: {flt.column}", expanded=True):
            # Render kind-specific controls; DashboardFilter kinds are
            # values / range / date_range (dashboard_studio.py).
            proposed = flt
            if flt.kind == "values":
                options = list(categorical_filter_options(current_df, flt.column))
                for saved_value in flt.values:
                    if saved_value not in options:
                        options.append(saved_value)
                selected = tuple(
                    st.multiselect(
                        "Values",
                        options,
                        default=list(flt.values),
                        placeholder="All values",
                        key=f"ds-flt-values-{flt.filter_id}-{nonce}",
                    )
                )
                proposed = replace(flt, values=selected)
            elif flt.kind == "range":
                minimum = float(
                    st.number_input(
                        "Minimum",
                        value=float(flt.minimum),
                        key=f"ds-flt-min-{flt.filter_id}-{nonce}",
                    )
                )
                maximum = float(
                    st.number_input(
                        "Maximum",
                        value=float(flt.maximum),
                        key=f"ds-flt-max-{flt.filter_id}-{nonce}",
                    )
                )
                if minimum > maximum:
                    st.error("Minimum must not be greater than maximum.")
                else:
                    proposed = replace(flt, minimum=minimum, maximum=maximum)
            else:
                start = st.date_input(
                    "Start date",
                    value=pd.Timestamp(flt.start).date(),
                    key=f"ds-flt-start-{flt.filter_id}-{nonce}",
                )
                end = st.date_input(
                    "End date",
                    value=pd.Timestamp(flt.end).date(),
                    key=f"ds-flt-end-{flt.filter_id}-{nonce}",
                )
                if start > end:
                    st.error("Start date must not be after end date.")
                else:
                    proposed = replace(flt, start=start.isoformat(), end=end.isoformat())

            if proposed != flt:
                try:
                    # replace_dashboard_filter(config, updated) -> DashboardConfig
                    _commit(current_df, studio_state, replace_dashboard_filter(config, proposed))
                    st.rerun()
                except DashboardStudioError as exc:
                    st.error(str(exc))

            if st.button("Remove filter", key=f"ds-flt-remove-{flt.filter_id}-{nonce}"):
                _commit(
                    current_df,
                    studio_state,
                    replace(
                        config,
                        filters=tuple(
                            existing
                            for existing in config.filters
                            if existing.filter_id != flt.filter_id
                        ),
                    ),
                    reset_widgets=True,
                )
                st.rerun()

# ── Apply filters ─────────────────────────────────────────────────────────────
try:
    filtered_df = apply_dashboard_filters(current_df, config.filters)
except DashboardStudioError as exc:
    st.error(str(exc))
    filtered_df = current_df

# ── Canvas header ─────────────────────────────────────────────────────────────
new_name = st.text_input(
    "Dashboard name",
    value=config.name,
    max_chars=80,
    key=f"ds-name-{scoped_key}-{nonce}",
).strip()
if new_name and new_name != config.name:
    try:
        _commit(current_df, studio_state, replace(config, name=new_name))
        config = studio_state["config"]
    except DashboardStudioError as exc:
        st.error(str(exc))

st.caption(f"{len(filtered_df):,} of {len(current_df):,} rows in view · {len(config.filters)} global filter(s)")

# ── Render cards ──────────────────────────────────────────────────────────────
card_pairs = [config.cards[i : i + 2] for i in range(0, len(config.cards), 2)]
for pair_index, pair in enumerate(card_pairs):
    columns_ui = st.columns(2)
    for offset, card in enumerate(pair):
        card_index = pair_index * 2 + offset
        with columns_ui[offset]:
            with st.container(border=True):
                try:
                    result = build_card_result(filtered_df, card)
                except DashboardStudioError as exc:
                    st.warning(str(exc))
                    result = None

                if result is not None:
                    # CardResult fields: value / figure / audit_table / calculation.
                    # KPI cards have figure=None; charts have value=None.
                    if card.kind == "kpi":
                        st.metric(card.title, format_metric_value(result.value))
                    else:
                        st.markdown(f"**{card.title}**")
                        if result.audit_table.empty:
                            st.info("No usable rows match this card and the active filters.")
                        elif result.figure is not None:
                            st.plotly_chart(
                                result.figure,
                                width="stretch",
                                config={"displaylogo": False, "responsive": True},
                            )
                    with st.expander("Calculation details"):
                        st.caption(result.calculation)
                        st.dataframe(
                            result.audit_table.head(500),
                            width="stretch",
                            hide_index=True,
                        )
                        if len(result.audit_table) > 500:
                            st.caption(
                                f"Showing 500 of {len(result.audit_table):,} audit rows."
                            )

                # Card controls
                ctrl_cols = st.columns(3)

                if ctrl_cols[0].button(
                    "↑",
                    key=f"ds-up-{card.card_id}-{nonce}",
                    help="Move earlier",
                    disabled=card_index == 0,
                ):
                    # move_dashboard_card(config, card_id, direction: int)
                    _commit(
                        current_df,
                        studio_state,
                        move_dashboard_card(config, card.card_id, -1),
                        reset_widgets=True,
                    )
                    st.rerun()

                if ctrl_cols[1].button(
                    "↓",
                    key=f"ds-down-{card.card_id}-{nonce}",
                    help="Move later",
                    disabled=card_index == len(config.cards) - 1,
                ):
                    _commit(
                        current_df,
                        studio_state,
                        move_dashboard_card(config, card.card_id, 1),
                        reset_widgets=True,
                    )
                    st.rerun()

                if ctrl_cols[2].button(
                    "✕",
                    key=f"ds-remove-{card.card_id}-{nonce}",
                    help="Remove card",
                    disabled=len(config.cards) == 1,
                ):
                    try:
                        _commit(
                            current_df,
                            studio_state,
                            remove_dashboard_card(config, card.card_id),
                            reset_widgets=True,
                        )
                        st.rerun()
                    except DashboardStudioError as exc:
                        st.error(str(exc))

# ── Add card ──────────────────────────────────────────────────────────────────
if len(config.cards) < MAX_CARDS:
    st.divider()
    with st.expander("Add card", expanded=False):
        # Offer only card kinds the current columns can support, and prefill
        # each new card with the first suitable column so it validates.
        available_kinds = ["kpi"]
        if all_cols:
            available_kinds.append("bar")
        if date_cols:
            available_kinds.append("line")
        if len(numeric_cols) >= 2:
            available_kinds.append("scatter")
        if numeric_cols:
            available_kinds.append("distribution")

        new_kind_label = st.selectbox(
            "Card type",
            [STUDIO_CARD_LABELS[kind] for kind in available_kinds],
            key=f"ds-new-kind-{scoped_key}-{nonce}",
        )
        new_kind = STUDIO_CARD_VALUES[new_kind_label]
        new_card_title = st.text_input(
            "Card title",
            value=new_kind_label,
            max_chars=100,
            key=f"ds-new-title-{scoped_key}-{nonce}",
        ).strip()

        if st.button("Add card", type="primary", key=f"ds-add-card-{scoped_key}-{nonce}"):
            card_kwargs: dict[str, object] = {}
            if new_kind == "bar":
                card_kwargs["x"] = all_cols[0]
            elif new_kind == "line":
                card_kwargs["x"] = date_cols[0]
            elif new_kind == "scatter":
                card_kwargs["x"] = numeric_cols[0]
                card_kwargs["y"] = numeric_cols[1]
            elif new_kind == "distribution":
                card_kwargs["column"] = numeric_cols[0]

            new_card = DashboardCard(
                next_dashboard_id(config, "card"),
                new_kind,
                new_card_title or new_kind_label,
                **card_kwargs,
            )
            try:
                _commit(
                    current_df,
                    studio_state,
                    replace(config, cards=(*config.cards, new_card)),
                    reset_widgets=True,
                )
                st.rerun()
            except DashboardStudioError as exc:
                st.error(str(exc))
        st.caption(
            "New charts start from the first suitable column. For precise control "
            "over columns and calculations, save the layout JSON, edit it, and load it back."
        )

# ── Save / Load layout JSON ───────────────────────────────────────────────────
st.divider()
save_col, load_col = st.columns(2)

with save_col:
    layout_json = dashboard_config_to_json(config)
    st.download_button(
        "Save layout JSON",
        data=layout_json.encode("utf-8"),
        file_name=f"{safe_download_stem(config.name)}-layout.json",
        mime="application/json",
        key=f"ds-save-{scoped_key}-{nonce}",
    )

with load_col:
    uploaded_layout = st.file_uploader(
        "Load layout JSON",
        type=["json"],
        key=f"ds-load-{scoped_key}-{nonce}",
    )
    if uploaded_layout is not None:
        try:
            # dashboard_config_from_json(text, df) validates against the dataset
            loaded_config = dashboard_config_from_json(
                uploaded_layout.getvalue().decode("utf-8"), current_df
            )
            _commit(current_df, studio_state, loaded_config, reset_widgets=True)
            st.rerun()
        except (DashboardStudioError, UnicodeDecodeError) as exc:
            st.error(f"Could not load layout: {exc}")

# ── Offline HTML export ───────────────────────────────────────────────────────
if st.button("Export standalone HTML dashboard", key=f"ds-html-{scoped_key}-{nonce}"):
    try:
        # build_dashboard_html(df, config) applies config.filters itself
        studio_state["html_export"] = build_dashboard_html(current_df, config)
    except DashboardStudioError as exc:
        st.error(str(exc))

if studio_state.get("html_export"):
    st.download_button(
        "Download HTML",
        data=studio_state["html_export"].encode("utf-8"),
        file_name=f"{safe_download_stem(config.name)}-dashboard.html",
        mime="text/html",
        key=f"ds-html-download-{scoped_key}-{nonce}",
    )
