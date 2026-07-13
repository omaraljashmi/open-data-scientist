# Open Data Scientist (ODS)

[![Launch Live App](https://img.shields.io/badge/Launch_Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://open-data-scientist-omar.streamlit.app)
[![GitHub License](https://img.shields.io/github/license/omaraljashmi/open-data-scientist?style=for-the-badge)](LICENSE)

Open Data Scientist is a local-first, open-source assistant that turns CSV and Excel files into understandable data profiles, quality findings, statistics, and downloadable reports—without a paid API.

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
│   └── reporting.py           # Downloadable report generation
├── tests/test_profiler.py     # Core unit tests
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
- [ ] Milestone 3 — Visual SQL query builder with DuckDB
- [ ] Milestone 4 — SQL explanation and optimization assistant
- [ ] Milestone 5 — Optional local LLM and agent activity log
- [ ] Milestone 6 — CLI, Docker image, demo, and contributor documentation

## Privacy

ODS processes files inside the local Streamlit session. Milestone 1 does not send data to a paid API or external model.

## License

Copyright © 2026 Omar Al Jashmi. Released under the [MIT License](LICENSE).
