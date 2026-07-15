## What changed

Describe the problem and the user-visible behavior of this change.

## Verification

- [ ] `python -m compileall -q app.py ods tests scripts`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python -m scripts.validation_matrix`
- [ ] `python -m scripts.benchmark --rows 25000 --max-seconds 20`
- [ ] Docker build and health check completed, or the limitation is documented.

## Safety and compatibility

- [ ] Tests and screenshots use bundled or synthetic data only.
- [ ] No secrets, private data, identifying paths, or proprietary SQL are included.
- [ ] Privacy, resource limits, dependencies, supported formats, and exports were reviewed when affected.
- [ ] The change remains deterministic and explainable, or the new boundary is documented.
