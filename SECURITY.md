# Security Policy

## Supported version

Security fixes are prepared for the current release candidate and the latest code on `main`. Earlier milestone snapshots are not maintained as separate supported releases.

## Reporting a vulnerability

Do not place secrets, private datasets, exploit details, or personal information in a public issue.

Use GitHub's private vulnerability-reporting flow for this repository when it is available. If it is unavailable, email `omaraljashmi.uni@gmail.com` with the subject `ODS security report`. Include the affected version, impact, reproduction steps, and the smallest safe proof of concept. Do not attach real sensitive data.

## Security boundaries

- Treat the public demo as a hosted service and use a local installation for sensitive data.
- The default upload guardrails are 25 MB, 250,000 rows, and 500 columns.
- XLSX archives are checked for malformed, encrypted, or unexpectedly expanded content before parsing.
- Visual SQL parameterizes values and runs only against an in-memory relation.
- SQL Coach blocks writes, management statements, multiple statements, external readers, and unknown tables.
- Dependency versions used for release verification are recorded in `requirements.lock`.

These controls reduce risk but do not turn ODS into a hardened multi-tenant data platform. Deploy it only in an environment appropriate for the data being processed.
