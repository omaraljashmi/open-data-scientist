# v1.0.0 Stable Release Checklist

This checklist promotes `v1.0.0-rc.1` only after Milestone 6.2 validation. Do not create the stable tag while any required item remains open.

## Validation gate

- [x] Preserve the published release-candidate behavior and version metadata during validation.
- [x] Validate standard, international, quoted multiline, single-column, and sparse CSV inputs.
- [x] Validate mixed-type XLSX and legacy XLS inputs.
- [x] Reject unsupported, empty, header-only, binary, malformed, blank-header, duplicate-header, damaged, and oversized inputs clearly.
- [x] Exercise profiling, cleaning, dashboards, Visual SQL, SQL Coach, and exports for every accepted case.
- [x] Add privacy-safe bug, feature, and pull-request templates.
- [x] Pass the complete locked 86-test suite locally on Python 3.11 and Python 3.12 after every Milestone 6.2 change.
- [x] Pass the deterministic 100,000-row benchmark in 0.618 seconds in the clean Python 3.12 stable-preparation environment.

## CI and deployment gate

- [x] Python 3.11 test job passes on GitHub Actions.
- [x] Python 3.12 test job passes on GitHub Actions.
- [x] Docker image builds on GitHub Actions.
- [x] Docker container reaches the Streamlit health endpoint.
- [x] Hosted Streamlit redeploys from the approved commit.
- [x] Hosted sample walkthrough completes without an exception.

Release-candidate run `29387684032` built the Docker image successfully, but the first health probe reached the starting container after roughly 0.7 seconds and exited with curl error 56 (`Connection reset by peer`). The focused CI fix retries all temporary startup errors, prints container logs on failure, and always removes the test container. Verification run `29389780191` passed the Python 3.11, Python 3.12, Docker build, and Streamlit container-health jobs. The public Streamlit app then completed the bundled sample walkthrough with six rows, five columns, a 73/100 quality score, and no rendered exception.

## Stable promotion

- [x] Resolve every release-blocking issue and record the verification evidence.
- [x] Change version constants from `1.0.0rc1` / `v1.0.0-rc.1` to `1.0.0` / `v1.0.0`.
- [x] Move the validated changes from **Unreleased** into the dated `1.0.0` changelog section.
- [x] Finalize `docs/releases/v1.0.0.md` from the verified results.
- [ ] Review the exact GitHub diff and obtain explicit publication approval.
- [ ] Publish the approved stable-preparation commit and confirm all required checks.
- [ ] Obtain separate approval to create tag `v1.0.0` and the public GitHub release.

## Rollback

If stable promotion fails, keep `v1.0.0-rc.1` as the latest verified public version, revert only the stable-preparation commit, and reopen the failed gate with sanitized evidence.
