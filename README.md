# Open Data Scientist (ODS)

[![Launch Live App](https://img.shields.io/badge/Launch_Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://open-data-scientist-omar.streamlit.app)
[![GitHub License](https://img.shields.io/github/license/omaraljashmi/open-data-scientist?style=for-the-badge)](LICENSE)
[![CI](https://github.com/omaraljashmi/open-data-scientist/actions/workflows/ci.yml/badge.svg)](https://github.com/omaraljashmi/open-data-scientist/actions/workflows/ci.yml)
[![Stable release](https://img.shields.io/badge/release-v1.0.0-56c9ff)](docs/releases/v1.0.0.md)

Open Data Scientist is a local-first, open-source assistant that turns CSV and Excel files into understandable profiles, reviewable cleaning steps, guided and custom dashboards, visual queries, and SQL optimization guidance—without a paid API.

**[Try the live demo →](https://open-data-scientist-omar.streamlit.app)** — click **Try sample dataset** for a no-upload walkthrough.

> The public demo processes uploads in a hosted Streamlit session. Do not upload confidential, regulated, personal, or sensitive data; [run ODS locally](#quick-start) instead.

## Current capabilities

### Milestone 1: Data Profile + Quality Report

- CSV and Excel uploads
- Dataset dimensions, memory use, missing cells, and duplicate counts
- Column types, missingness, unique values, and examples
- Automatic warnings for missing data, duplicates, constant columns, and possible IDs
- Numeric descriptive statistics
- Explainable 0–100 quality score
- Downloadable Markdown quality report
- A polished Streamlit interface

### Milestone 2: Automatic Dashboard Generator

- Infers numeric, categorical, date, and identifier columns
- Generates up to four useful charts instead of overwhelming the user
- Creates category counts, numeric distributions, time trends, missingness charts, averages by category, and scatterplots when appropriate
- Excludes likely identifier columns from numeric measures
- Explains why every chart was selected

### Milestone 2.1: Smart Guided Dashboard

- Shows the inferred analytical role, display format, confidence, and reason for every column
- Lets the user correct identifiers, measures, categories, dates, free text, and ignored columns
- Recommends charts for a selected goal: overview, trends, comparisons, distributions, relationships, or data quality
- Makes aggregation and date grain explicit instead of guessing silently
- Uses data-driven histogram bins and safer identifier/date detection
- Shows the exact summary table behind every chart for verification
- Runs entirely with local Python libraries and no paid API

### Milestone 3: Visual SQL Query Builder

- Builds row-level and grouped-summary queries with buttons instead of handwritten SQL
- Lets the user choose output columns, up to three filters, grouping, a calculation, sorting, and a row limit
- Supports counts, averages, medians, totals, minimums, and maximums
- Shows the generated SQL so every result is understandable and reusable
- Executes against the current upload in an isolated, in-memory DuckDB database
- Quotes column names and binds filter values separately to prevent SQL injection
- Previews results and downloads the complete result as CSV
- Runs locally with open-source libraries and no paid API

### Milestone 4: Local SQL Coach

- Explains Visual SQL or pasted DuckDB queries clause by clause in plain English
- Uses SQLGlot for DuckDB-aware parsing, formatting, and syntax-tree inspection
- Uses DuckDB `EXPLAIN` to show the real physical plan without executing the query result
- Translates physical-plan operators such as scans, filters, joins, grouping, sorting, and limits
- Detects correctness risks including unsafe NULL comparisons and `NOT IN` subqueries
- Detects performance and maintainability risks including broad projections, unbounded results, leading wildcards, function-wrapped filters, expensive sorts, and Cartesian joins
- Uses the uploaded dataset to identify nearly unique grouping keys
- Produces a conservative clean rewrite and never silently applies changes that could alter results
- Blocks writes, database commands, multiple statements, unknown tables, and external file readers
- Runs locally with transparent rules and no paid API

### Milestone 5: Data Cleaning Studio

- Detects exact duplicates, surrounding whitespace, blank text, inconsistent category casing, suspicious data types, missing values, constant columns, and robust IQR outliers
- Separates evidence from assumptions and gives every recommendation a visible confidence level
- Selects nothing automatically and requires a before/after preview before Apply
- Converts only unambiguous numeric and date text while protecting identifier-like and leading-zero codes
- Uses explicit median or missing-category fills only for limited missingness and leaves ambiguous cases for domain review
- Flags possible outliers in a new Boolean column instead of deleting or capping source values
- Applies verified fixes to dashboards, Visual SQL, SQL Coach, profiles, statistics, reports, and downloads
- Supports multi-batch undo and a full reset without overwriting the uploaded source
- Downloads the current cleaned CSV and a reproducible JSON recipe with source hash, ordered operations, parameters, and evidence
- Runs locally with deterministic rules and no paid API

### Milestone 6: Dashboard Studio

- Starts with a useful four-card layout while keeping the earlier guided recommendation mode
- Builds editable KPI cards, category bars, time trends, scatter plots, and numeric distributions
- Supports row counts, totals, averages, medians, and distinct counts with visual column controls
- Applies up to five categorical, numeric-range, or date-range filters to every card
- Renames, changes, reorders, adds, and removes cards in a responsive two-column canvas
- Shows the exact audit table and plain-language calculation behind every result
- Saves and reloads validated layout JSON without embedding the source dataset
- Exports a standalone responsive HTML dashboard with Plotly included for offline viewing
- Runs locally with open-source libraries and no paid API

### Milestone 6.1: Release Candidate Hardening

- Adds an in-app privacy boundary and one-click sample walkthrough
- Enforces clear limits for file size, rows, columns, and expanded XLSX content
- Returns understandable errors for empty, malformed, binary, encrypted, and oversized inputs
- Locks the verified dependency graph while retaining supported ranges for development
- Tests Python 3.11/3.12, Streamlit rendering, representative performance, and Docker health in CI
- Prepares versioned release notes, changelog, security guidance, contributor documentation, and a release checklist

### Milestone 6.2: Stable Release Validation

- Runs seven synthetic CSV, XLSX, and XLS files through every product area
- Verifies UTF-8 BOM, semicolon, pipe, quoted multiline, sparse, mixed-type, and single-column inputs
- Requires nine unsafe or malformed input families to fail with understandable messages
- Fixes single-column CSV detection and rejects blank or duplicate source headers before silent renaming
- Adds privacy-safe bug reports, feature requests, and pull-request verification templates
- Records the verified Python 3.11/3.12 CI, Docker health, and hosted sample-walkthrough evidence

### Milestone 7: Zero-Cost Export Pipeline

- Sends the current working dataset (with all applied cleaning fixes) to a destination the user owns
- Appends to an existing Airtable table on the free plan using the user's own personal access token, in API-sized batches with rate limiting and typecast
- Posts JSON batches to any webhook endpoint the user controls, with an optional secret header
- Shows the exact first request payload before anything is sent, and reports sent, failed, and retried batches afterwards
- Keeps credentials in session memory or environment variables only; pipeline JSON stores the destination shape and never a token
- Replays saved cleaning recipes headlessly, so `python -m scripts.run_pipeline` turns an upload plus two JSON files into a schedulable pipeline (cron or a free GitHub Actions schedule)
- Uses no AI model and no paid API: destination mapping is deterministic and every service involved has a free tier

### Milestone 7.1: Structured Visual Theme + Optional AI Chart Advisor

- Applies one shared visual theme across every page: a pitch-green and cream palette with a centered content column, consistent page headers with a kicker and muted subtitle, and card-styled metrics and expanders
- Adds an **opt-in** AI chart advisor to the Profile page that asks an LLM to pick charts for the selected analysis goal — off by default and never required
- Stays zero-cost by using only endpoints the user brings: a free-tier Google Gemini or Groq key, a local Ollama server, or any OpenAI-compatible URL
- Sends **dataset metadata only** — column names, inferred roles, formats, unique and missing counts — never cell values, examples, or statistics
- Treats the model as untrusted: every suggestion is validated against the real columns and roles, invalid picks are dropped, and surviving picks render through the same local, auditable chart calculations as the deterministic recommendations
- Keeps the API key in session memory only; it is never stored, logged, or written to disk

## Quick start

The stable release is verified on Python 3.11 and 3.12.

```bash
git clone https://github.com/omaraljashmi/open-data-scientist.git
cd open-data-scientist
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.lock
python -m streamlit run app.py
```

Then click **Try sample dataset** or upload a supported file. Use `requirements.txt` instead when you intentionally want the newest compatible dependencies rather than the exact stable-release environment.

### Docker

```bash
docker build -t open-data-scientist .
docker run --rm -p 8501:8501 open-data-scientist
```

Open `http://localhost:8501`. The image includes a health check at `/_stcore/health`.

### Input envelope

| File boundary | Default |
|---|---:|
| CSV, XLSX, or XLS upload | 25 MB |
| Rows | 250,000 |
| Columns | 500 |
| Expanded XLSX content | 250 MB |

These are deliberate stable-release guardrails, not claims that every file at the limit will use the same memory or runtime. See [performance and resource details](docs/PERFORMANCE.md).

## Project structure

```text
open-data-scientist/
├── app.py                     # Entrypoint: page config + st.navigation router
├── Home.py                    # Landing page and session-state loader
├── app_shared.py              # Shared theme, label maps, and page helpers
├── pages/
│   ├── 1_Profile.py           # Profile, quality report, guided dashboard, AI advisor
│   ├── 2_Clean.py             # Data Cleaning Studio
│   ├── 3_Dashboard.py         # Dashboard Studio
│   ├── 4_Visual_SQL.py        # Visual SQL query builder
│   ├── 5_SQL_Coach.py         # SQL explanation and optimization
│   └── 6_Pipeline.py          # Zero-cost export pipeline
├── ods/
│   ├── loader.py              # CSV and Excel ingestion
│   ├── cleaning.py            # Review-first cleaning recommendations and replay
│   ├── profiler.py            # Profiling and quality rules
│   ├── dashboard.py           # Chart inference and preparation
│   ├── dashboard_studio.py    # Custom cards, global filters, validation, and HTML export
│   ├── query_builder.py       # Safe visual SQL generation and execution
│   ├── sql_coach.py           # Read-only SQL explanation and plan review
│   ├── advisor.py             # Optional bring-your-own-endpoint AI chart advisor
│   ├── pipeline.py            # Airtable and webhook export destinations
│   ├── reporting.py           # Downloadable report generation
│   └── version.py             # Release version constants
├── tests/
│   ├── test_app_smoke.py      # Landing page and sample walkthrough
│   ├── test_cleaning.py       # Cleaning safety, replay, and recipe tests
│   ├── test_profiler.py       # Profiling and dashboard tests
│   ├── test_dashboard_studio.py # Dashboard calculations, filters, config, and export tests
│   ├── test_query_builder.py  # Query generation and security tests
│   ├── test_sql_coach.py      # SQL safety, explanation, and plan tests
│   ├── test_advisor.py        # Advisor privacy, validation, and endpoint-failure tests
│   ├── test_pipeline.py       # Export batching, retries, and credential-safety tests
│   ├── test_performance.py    # Representative workflow performance gate
│   └── test_validation_matrix.py # Multi-format end-to-end regression gate
├── scripts/                   # Benchmark, validation, and headless pipeline runners
├── docs/                      # Performance, validation, checklists, and release notes
├── examples/                  # Safe sample data
├── requirements.txt           # Supported dependency ranges
├── requirements.lock          # Exact stable-release environment
├── Dockerfile
└── LICENSE
```

## Run the tests

The test suite uses Python's standard library test runner:

```bash
python -m compileall -q app.py ods tests scripts
python -m unittest discover -s tests -v
python -m scripts.validation_matrix
python -m scripts.benchmark --rows 100000 --max-seconds 30
```

GitHub Actions runs the suite and validation matrix on Python 3.11 and 3.12 and separately builds and health-checks the Docker image. See the [validation matrix](docs/VALIDATION.md) and [stable-release checklist](docs/STABLE_RELEASE_CHECKLIST.md).

## Roadmap

- [x] Milestone 1 — Upload, profile, and quality report
- [x] Milestone 2 — Automatic dashboard generator
- [x] Milestone 2.1 — Guided semantic roles and auditable chart calculations
- [x] Milestone 3 — Visual SQL query builder with DuckDB
- [ ] Milestone 3.1 — OR filter groups and multi-file joins
- [x] Milestone 4 — SQL explanation and optimization assistant
- [x] Milestone 5 — Review-first Data Cleaning Studio with undo and recipes
- [x] Milestone 6 — Custom dashboard composer and shareable dashboard configuration
- [x] Milestone 6.1 — Release candidate hardening, CI, privacy, guardrails, and Docker
- [x] Milestone 6.2 — Multi-format validation and stable `v1.0.0` promotion
- [x] Milestone 7 — Zero-cost export pipeline (Airtable, webhook, headless runner)
- [x] Milestone 7.1 — Structured visual theme and optional free AI chart advisor (bring-your-own endpoint, metadata only)
- [ ] Milestone 8 — Agent activity log, CLI distribution, richer demo assets, and contributor expansion

## Privacy

ODS never sends uploaded data to a paid API, and never sends it to any AI model by default. The optional AI chart advisor is off until configured, talks only to an endpoint the user brings (a free tier or a local server), and receives dataset metadata only — column names, roles, and counts, never cell values. A local installation processes your data in your local Streamlit process; the public demo first transfers it to a hosted Streamlit session. Cleaning operations are deterministic, previewed, and replayed without overwriting the source. Dashboard layout JSON contains controls but no source rows; standalone HTML intentionally embeds the filtered results needed to render offline. SQL Coach uses an isolated in-memory DuckDB connection with external access disabled and generates an `EXPLAIN` plan without executing the result.

Read the full [Privacy Notice](PRIVACY.md) and [Security Policy](SECURITY.md) before using your own data.

## License

Copyright © 2026 Omar Al Jashmi. Released under the [MIT License](LICENSE).
