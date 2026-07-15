# Changelog

All notable changes to Open Data Scientist are documented here.

## [Unreleased]

## [1.0.0-rc.1] - 2026-07-14

### Added

- One-click bundled sample dataset and three-minute product walkthrough.
- Explicit public-demo privacy notice and detailed privacy/security documentation.
- Input limits for bytes, rows, columns, and expanded XLSX archives.
- Friendly errors for empty, binary, malformed, encrypted, oversized, and header-only files.
- Exact dependency lockfile and missing legacy `.xls` reader dependency.
- Python 3.11/3.12 CI, Streamlit smoke testing, deterministic performance checks, and Docker smoke testing.
- Production-oriented Dockerfile, health check, contributor guide, release checklist, and release notes.

### Changed

- Streamlit minimum version is 1.59 and deprecated width parameters are replaced.
- Excel and numeric headers are normalized to unique text before analysis.
- Public-demo upload size is reduced from 200 MB to a deliberate 25 MB processing envelope.

### Security

- XLSX archive expansion and member-count checks reduce decompression-risk exposure.
- Hosted-session privacy wording now distinguishes the public demo from a local installation.

## [0.6.0] - 2026-07-13

- Completed the explainable profiler, smart dashboards, Visual SQL, SQL Coach, Data Cleaning Studio, and Dashboard Studio milestones.

[Unreleased]: https://github.com/omaraljashmi/open-data-scientist/compare/v1.0.0-rc.1...HEAD
[1.0.0-rc.1]: https://github.com/omaraljashmi/open-data-scientist/releases/tag/v1.0.0-rc.1
