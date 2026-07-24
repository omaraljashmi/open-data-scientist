"""Zero-cost export pipeline page.

Session-state contract (written by app.py):
    original_df     pd.DataFrame
    filename        str
    dataset_key     str

Credentials entered on this page live only in this Streamlit session's
memory. They are never written to disk, logs, or the pipeline JSON.
"""

from __future__ import annotations

import streamlit as st

from ods import (
    AIRTABLE_FREE_RECORD_LIMIT,
    AirtableDestination,
    PipelineConfig,
    PipelineError,
    WebhookDestination,
    check_airtable_connection,
    dataframe_to_records,
    pipeline_config_to_json,
    preview_payload,
    push_to_airtable,
    push_to_webhook,
    replay_cleaning_batches,
)
from app_shared import cleaning_history_fingerprint, get_cleaning_state, safe_download_stem

st.set_page_config(page_title="ODS · Pipeline", layout="wide")

# ── Guard ─────────────────────────────────────────────────────────────────────
original_df = st.session_state.get("original_df")
filename    = st.session_state.get("filename", "upload")
dataset_key = st.session_state.get("dataset_key", "")

if original_df is None:
    st.info("Upload a file on the home page to get started.")
    st.stop()

# ── Resolve current (post-cleaning) dataframe ─────────────────────────────────
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
st.subheader("Export pipeline")
st.caption(
    "Send the current working dataset to a destination you own. ODS stays zero-cost: "
    "destinations use your own free-tier account, credentials live only in this session, "
    "and no AI model or paid API is involved."
)

kind = st.radio(
    "Destination",
    ["Airtable", "Webhook (any endpoint you control)"],
    horizontal=True,
    key=f"pipeline-kind-{scoped_key}",
)

pipeline_name = st.text_input(
    "Pipeline name",
    value=f"{filename.rsplit('.', 1)[0]} export",
    max_chars=80,
    key=f"pipeline-name-{scoped_key}",
).strip()

row_limit = int(
    st.number_input(
        "Rows to send",
        min_value=1,
        max_value=max(len(df), 1),
        value=len(df),
        key=f"pipeline-rows-{scoped_key}",
    )
)

config: PipelineConfig | None = None
token = ""
secret_value: str | None = None

if kind == "Airtable":
    columns = st.columns(2)
    base_id = columns[0].text_input(
        "Base ID (app…)",
        help="Copy it from the base URL: airtable.com/appXXXXXXXXXXXXXX/…",
        key=f"pipeline-base-{scoped_key}",
    ).strip()
    table = columns[1].text_input(
        "Table name or table ID",
        help="The table must already exist in the base; fields are matched by column name.",
        key=f"pipeline-table-{scoped_key}",
    ).strip()
    token = st.text_input(
        "Personal access token",
        type="password",
        help="Create a free token at airtable.com/create/tokens with data.records:write scope.",
        key=f"pipeline-token-{scoped_key}",
    ).strip()
    typecast = st.checkbox(
        "Let Airtable convert value types (typecast)",
        value=True,
        key=f"pipeline-typecast-{scoped_key}",
    )
    st.caption(
        "Kept only in this session's memory — never stored, logged, or written into the pipeline JSON. "
        f"Airtable's free plan currently allows about {AIRTABLE_FREE_RECORD_LIMIT:,} records per base "
        "and has a monthly API call allowance; batches of 10 records keep calls low."
    )
    if row_limit > AIRTABLE_FREE_RECORD_LIMIT:
        st.warning(
            f"You are sending {row_limit:,} rows; Airtable's free plan caps a base at "
            f"about {AIRTABLE_FREE_RECORD_LIMIT:,} records. Reduce the rows or use a paid base you already have."
        )
    if base_id and table:
        config = PipelineConfig(
            name=pipeline_name or "ods-pipeline",
            kind="airtable",
            airtable=AirtableDestination(base_id=base_id, table=table, typecast=typecast),
            row_limit=row_limit,
        )
        if st.button("Check connection", key=f"pipeline-check-{scoped_key}"):
            try:
                st.success(check_airtable_connection(config.airtable, token))
            except PipelineError as exc:
                st.error(str(exc))
else:
    url = st.text_input(
        "Webhook URL (https)",
        help="Any endpoint you control: your own service, or a free-tier automation URL (n8n, Make, Zapier…).",
        key=f"pipeline-url-{scoped_key}",
    ).strip()
    columns = st.columns(3)
    batch_size = int(
        columns[0].number_input(
            "Records per request",
            min_value=1,
            max_value=1000,
            value=500,
            key=f"pipeline-batch-{scoped_key}",
        )
    )
    secret_header = columns[1].text_input(
        "Secret header name (optional)",
        placeholder="X-ODS-Secret",
        key=f"pipeline-secret-header-{scoped_key}",
    ).strip()
    secret_input = columns[2].text_input(
        "Secret value (optional)",
        type="password",
        key=f"pipeline-secret-value-{scoped_key}",
    )
    secret_value = secret_input or None
    st.caption(
        "Kept only in this session's memory — the secret value is never stored or written into the pipeline JSON."
    )
    if url:
        config = PipelineConfig(
            name=pipeline_name or "ods-pipeline",
            kind="webhook",
            webhook=WebhookDestination(
                url=url,
                batch_size=batch_size,
                secret_header=secret_header or None,
            ),
            row_limit=row_limit,
        )

if config is None:
    st.info("Fill in the destination details above to continue.")
    st.stop()

# ── Audit before sending ──────────────────────────────────────────────────────
try:
    records = dataframe_to_records(df, row_limit=row_limit)
except PipelineError as exc:
    st.error(str(exc))
    st.stop()

st.markdown("#### 1. Review exactly what will be sent")
metrics = st.columns(3)
metrics[0].metric("Records", f"{len(records):,}")
metrics[1].metric("Columns", f"{len(df.columns):,}")
metrics[2].metric("Cleaning fixes applied", f"{sum(len(batch) for batch in batches):,}")
with st.expander("First request payload (verbatim)"):
    try:
        st.code(preview_payload(records, config), language="json")
    except PipelineError as exc:
        st.error(str(exc))
        st.stop()
    st.caption("This is byte-for-byte the JSON body of the first request. No hidden fields are added.")

# ── Run ───────────────────────────────────────────────────────────────────────
st.markdown("#### 2. Run the pipeline")
if st.button("Send to destination", type="primary", key=f"pipeline-run-{scoped_key}"):
    progress_bar = st.progress(0, text="Sending…")

    def update_progress(done: int, total: int) -> None:
        progress_bar.progress(
            min(done / max(total, 1), 1.0),
            text=f"Sending… {min(done, total):,} of {total:,} records",
        )

    try:
        if config.kind == "airtable":
            assert config.airtable is not None
            report = push_to_airtable(
                records, config.airtable, token, progress=update_progress
            )
        else:
            assert config.webhook is not None
            report = push_to_webhook(
                records,
                config.webhook,
                secret_value=secret_value,
                progress=update_progress,
                pipeline_name=config.name,
            )
    except PipelineError as exc:
        progress_bar.empty()
        st.error(str(exc))
    else:
        progress_bar.empty()
        result_columns = st.columns(3)
        result_columns[0].metric("Sent", f"{report.sent_records:,}/{report.total_records:,}")
        result_columns[1].metric("Requests", f"{report.batches_sent:,}")
        result_columns[2].metric("Failures", f"{len(report.failures):,}")
        if report.ok:
            st.success(f"All records were delivered to {report.destination}.")
        else:
            for failure in report.failures:
                st.error(failure)

# ── Save the pipeline for headless runs ───────────────────────────────────────
st.markdown("#### 3. Reuse this pipeline")
st.caption(
    "The pipeline JSON stores the destination shape only — no tokens or secrets. "
    "Run it headlessly (cron, CI, a free GitHub Actions schedule) with: "
    "`python -m scripts.run_pipeline --input data.csv --pipeline pipeline.json` "
    "plus your token in the AIRTABLE_TOKEN or ODS_WEBHOOK_SECRET environment variable. "
    "Add `--cleaning-recipe recipe.json` to replay saved cleaning fixes first."
)
st.download_button(
    "Download pipeline JSON",
    data=pipeline_config_to_json(config).encode("utf-8"),
    file_name=f"{safe_download_stem(config.name)}-pipeline.json",
    mime="application/json",
    key=f"pipeline-download-{scoped_key}",
)
