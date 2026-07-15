# v1.0.0-rc.1 Release Checklist

## Local release gate

- [x] Version constants report `1.0.0rc1` / `v1.0.0-rc.1`.
- [x] Exact release dependencies install and pass a compatibility check in a clean environment.
- [x] Python source compiles without errors.
- [x] Complete unit, safety, calculation, performance, and Streamlit smoke suite passes.
- [x] One-click sample walkthrough loads all five product areas without an exception.
- [x] Deterministic 100,000-row benchmark is within the documented 30-second envelope.
- [x] Docker image builds and its `/_stcore/health` endpoint responds in GitHub Actions.
- [x] Privacy, security, input limits, install paths, and known boundaries are documented.

The local verification environment did not provide a container engine. GitHub Actions run `29389780191` later verified the Docker build and health endpoint.

## Publication gate

- [x] Review the exact GitHub diff; exclude unrelated local worktree state.
- [x] Publish the release-candidate files to `main` only after explicit approval.
- [x] Confirm both GitHub Actions jobs pass on `main`.
- [x] Confirm the hosted Streamlit app redeploys and the sample walkthrough works.
- [x] Create annotated tag `v1.0.0-rc.1` only after separate approval.
- [x] Create the GitHub prerelease from `docs/releases/v1.0.0-rc.1.md` only after separate approval.
- [x] Record and resolve every release-blocking issue before promoting to `v1.0.0`.

## Rollback

If the hosted app fails after publication, revert only the Milestone 6.1 file commits, verify Milestone 6 behavior, and keep the prerelease unpublished until the release gate is green again.
