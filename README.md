# Open Data Scientist (ODS)

[![Launch Live App](https://img.shields.io/badge/Launch_Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://open-data-scientist-omar.streamlit.app)
[![GitHub License](https://img.shields.io/github/license/omaraljashmi/open-data-scientist?style=for-the-badge)](LICENSE)

Open Data Scientist is a local-first, open-source assistant that turns CSV and Excel files into understandable profiles, guided dashboards, visual queries, and SQL optimization guidance—without a paid API.

**[Try the live demo →](https://open-data-scientist-omar.streamlit.app)**

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

## Quick start

```bash
git clone https://github.com/omaraljashmi/open-data-scientist.git
cd open-data-scientist
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then upload your own dataset or try `examples/sample_customers.csv`.

## Project structure

```text
open-data-scientist/
├── app.py                     # Streamlit interface
├── ods/
│   ├── loader.py              # CSV and Excel ingestion
│   ├── profiler.py            # Profiling and quality rules
│   ├── dashboard.py           # Chart inference and preparation
│   ├── query_builder.py       # Safe visual SQL generation and execution
│   ├── sql_coach.py           # Read-only SQL explanation and plan review
│   └── reporting.py           # Downloadable report generation
├── tests/
│   ├── test_profiler.py       # Profiling and dashboard tests
│   ├── test_query_builder.py  # Query generation and security tests
│   └── test_sql_coach.py      # SQL safety, explanation, and plan tests
├── examples/                  # Safe sample data
├── requirements.txt
└── LICENSE
```

## Run the tests

The test suite uses Python's standard library test runner:

```bash
python -m unittest discover -s tests -v
```

## Roadmap

- [x] Milestone 1 — Upload, profile, and quality report
- [x] Milestone 2 — Automatic dashboard generator
- [x] Milestone 2.1 — Guided semantic roles and auditable chart calculations
- [x] Milestone 3 — Visual SQL query builder with DuckDB
- [ ] Milestone 3.1 — OR filter groups and multi-file joins
- [x] Milestone 4 — SQL explanation and optimization assistant
- [ ] Milestone 5 — Optional local LLM and agent activity log
- [ ] Milestone 6 — CLI, Docker image, demo, and contributor documentation

## Privacy

ODS processes files inside the Streamlit session and does not send uploaded data to a paid API or external model. SQL Coach uses an isolated in-memory DuckDB connection with external access disabled and generates an `EXPLAIN` plan without executing the query result.

## License

Copyright © 2026 Omar Al Jashmi. Released under the [MIT License](LICENSE).
