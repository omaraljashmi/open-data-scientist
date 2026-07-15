# Contributing

Contributions that make ODS safer, clearer, faster, or easier to verify are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.lock
```

`requirements.txt` contains supported dependency ranges. `requirements.lock` records the exact release-candidate environment.

## Before opening a pull request

```bash
python -m compileall -q app.py ods tests scripts
python -m unittest discover -s tests -v
python -m scripts.benchmark --rows 25000 --max-seconds 20
```

If Docker is available:

```bash
docker build -t open-data-scientist .
docker run --rm -p 8501:8501 open-data-scientist
```

Keep analytical rules deterministic and explainable. Add tests for calculation changes, malformed input, safety boundaries, and user-visible failure modes. Never include real private data in tests, screenshots, issues, or commits.

## Pull requests

Describe the problem, the user-visible behavior, how it was tested, and any privacy or compatibility impact. Keep unrelated changes separate so reviewers can verify the result.
