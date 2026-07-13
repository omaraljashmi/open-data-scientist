# Open Data Scientist (ODS)

Open Data Scientist is a local-first, open-source assistant that turns CSV and Excel files into understandable data profiles, quality findings, statistics, and downloadable reports—without a paid API.

## Milestone 1: Data Profile + Quality Report

The first working milestone includes:

- CSV and Excel uploads
- Dataset dimensions, memory use, missing cells, and duplicate counts
- Column types, missingness, unique values, and examples
- Automatic warnings for missing data, duplicates, constant columns, and possible IDs
- Numeric descriptive statistics
- Explainable 0–100 quality score
- Downloadable Markdown quality report
- A polished Streamlit interface

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
- [ ] Milestone 2 — Automated analysis-question planner
- [ ] Milestone 3 — Pandas/DuckDB analysis engine and charts
- [ ] Milestone 4 — Optional local LLM and agent activity log
- [ ] Milestone 5 — CLI, Docker image, demo, and contributor documentation

## Privacy

ODS processes files inside the local Streamlit session. Milestone 1 does not send data to a paid API or external model.

## License

Copyright © 2026 Omar Al Jashmi. Released under the [MIT License](LICENSE).

