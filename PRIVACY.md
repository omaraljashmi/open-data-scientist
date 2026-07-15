# Privacy Notice

Effective for Open Data Scientist `v1.0.0`.

Open Data Scientist (ODS) is designed to analyze tabular data without sending it to a paid API or external AI model. The privacy boundary depends on where you run it.

## Local installation

When ODS runs on your own computer:

- uploaded CSV and Excel bytes are read into the local Streamlit process;
- profiling, cleaning, dashboards, DuckDB queries, and SQL analysis run in local memory;
- the application does not include an application database or upload-storage service;
- Streamlit usage telemetry is disabled in the included configuration; and
- ODS does not call an external AI model or paid API.

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
