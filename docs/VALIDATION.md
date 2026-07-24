# Stable-release validation matrix

Milestone 6.2 validates the complete Data Insight Studio workflow with deterministic synthetic files. The matrix is safe to run locally and in CI because it contains no real customer, employer, financial, or personal data.

## What the matrix exercises

Every accepted file passes through:

1. bounded CSV or Excel loading;
2. profiling and quality scoring;
3. conservative high-confidence cleaning and recipe generation;
4. Dashboard Studio calculation, configuration round-trip, and standalone HTML export;
5. Visual SQL construction and DuckDB execution;
6. SQL Coach parsing, safety validation, and physical-plan evidence; and
7. Markdown quality-report generation.

## Accepted-file cases

| Case | Format | Boundary covered |
|---|---|---|
| Standard customer records | CSV | Mixed identifiers, categories, amounts, dates, missing values, and whitespace |
| International records | CSV | UTF-8 BOM, semicolon delimiter, and non-ASCII text |
| Quoted notes | CSV | Quoted commas and multiline values |
| Status list | CSV | A genuine single-column file with no delimiter |
| Sparse records | CSV | Pipe delimiter, missing numeric values, and blank text |
| Mixed orders | XLSX | Numeric, text, date, and Boolean values plus multiple worksheets |
| Legacy customers | XLS | The documented legacy Excel reader path |

## Expected rejections

The matrix also requires clear failures for unsupported extensions, empty files, header-only files, binary content disguised as CSV, malformed CSV quoting, blank headers, duplicate headers, damaged XLSX content, and oversized uploads.

## Run it

Install the exact release environment, then run:

```bash
python -m scripts.validation_matrix
```

Success prints JSON with `"status": "passed"`, seven accepted files, nine rejected files, and evidence from every product area. The full unit suite calls the same matrix so the standalone command and CI cannot silently diverge.

## Stable-promotion gate

The matrix is one part of the release decision. Stable `v1.0.0` also requires:

- the complete locked test suite on Python 3.11 and 3.12;
- the deterministic performance check;
- a successful Docker build and container health check;
- a successful hosted Streamlit sample walkthrough; and
- no unresolved release-blocking issue.

Synthetic coverage cannot prove that every real-world schema is correct for its business meaning. Users must still verify inferred roles, cleaning decisions, aggregations, filters, and SQL results against their domain rules.
