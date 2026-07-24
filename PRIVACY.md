# Privacy Notice

Effective for Open Data Scientist `v1.0.0`.

Open Data Scientist (ODS) is designed to analyze tabular data without sending it to a paid API, and without sending it to any AI model by default. Data leaves the application only through actions you explicitly configure and run: the export pipeline (which sends data rows to a destination you own) and the optional AI chart advisor (which sends dataset metadata only — never cell values — to an endpoint you bring). The privacy boundary depends on where you run it.

## Local installation

When ODS runs on your own computer:

- uploaded CSV and Excel bytes are read into the local Streamlit process;
- profiling, cleaning, dashboards, DuckDB queries, and SQL analysis run in local memory;
- the application does not include an application database or upload-storage service;
- Streamlit usage telemetry is disabled in the included configuration; and
- ODS does not call an external AI model or paid API unless you explicitly configure and run one of the opt-in outbound features below.

## Opt-in outbound features

Two features send anything over the network, and both are off until you configure them:

- **Export pipeline** — sends the current working dataset's rows to a destination you own (your Airtable base via your own personal access token, or a webhook endpoint you control). The exact first request payload is shown before anything is sent. Credentials live in session memory or environment variables only; pipeline JSON never contains them.
- **AI chart advisor** — sends **dataset metadata only** (column names, inferred roles, display formats, unique and missing counts, row count) to an OpenAI-compatible endpoint you bring: a free-tier key or a local server. Cell values, example values, and statistics of values never leave the machine, and every suggestion the model returns is validated locally before it is used. The API key lives in session memory only.

Use a local installation for confidential, regulated, personal, or otherwise sensitive data. Your operating system, browser, container platform, and network configuration remain outside ODS's control.

## Public Streamlit demo

When you use the public demo, your browser transfers the selected file to the hosted Streamlit application. ODS processes it in that hosted session rather than on your computer. The application code does not intentionally persist the source file, but the hosting provider controls the surrounding infrastructure, session lifecycle, network handling, and operational logs.

Do not upload confidential, regulated, personal, or sensitive data to the public demo. Use the included sample dataset or run ODS locally instead.

## Downloads and exports

- Quality reports contain profile summaries and column-level evidence.
- Cleaned CSV and Visual SQL downloads contain data rows selected by the user.
- Cleaning recipes contain a source hash, ordered operations, parameters, and evidence, but not the complete source rows.
- Dashboard layout JSON contains controls and referenced column names, but not source rows.
- Standalone dashboard HTML intentionally embeds the filtered results required to render its cards offline.

Review every download before sharing it. ODS cannot control files after they leave the application.

## SQL boundary

Visual SQL and SQL Coach use isolated in-memory DuckDB connections. External database access is disabled, and SQL Coach accepts only one read-only query against the uploaded dataset. This boundary reduces accidental access; it is not a substitute for running sensitive workloads in an approved environment.

## Questions

For a suspected security problem, follow [SECURITY.md](SECURITY.md). For general questions, open a GitHub issue without attaching private data.
