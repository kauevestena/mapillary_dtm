# Agent guide: observed surface slopes from Mapillary

The user's September 2026 redesign replaces absolute elevation with observed local
surface slopes for accessibility. Read README.md, documentation/ARCHITECTURE.md and
documentation/METHOD.md. The primary package is `slope_from_mapillary`.

## Scientific integrity and execution

- All testing, validation and execution must use real data, actual model inference
  and authentic reconstruction paths. Do not fabricate geometry, model outputs,
  semantic labels, calibration or field accuracy to obtain a passing demo.
- Call the core production functions in tests and validation. Do not create a
  parallel implementation or ad-hoc test-only estimator.
- `qa/data/sample_dataset` is the authoritative real imagery/SfM fixture. The
  tracked `qa/data/qa_dtm.tif` is a separate real reference for GIS regressions.
- Coordinate transformations of the real geometry may test mathematical
  invariances, but do not establish slope accuracy. Keep numerical test tolerances
  separate from the deployed measurement policy and disclose their purpose.
- Missing evidence must remain missing. The existing sample lacks a documented
  independent vertical calibration and suitable surface labels for its reconstructed
  images. A blocked audit is expected until the evidence is supplied.
- Use repository-local `.venv` and `.venv/bin/python`. `./setup_local.sh` creates the
  environment. Optional GPU/model/SfM dependencies are not required for core parsing
  and estimation from cached geometry. If package installation is unavailable,
  report unexecuted checks honestly; do not substitute fake inference.

## Measurement contracts

- Never derive slope from GPS altitude differences or assumed camera height.
- Never infer gravity by leveling the same ground plane whose slope is being measured.
- Preserve native measured feature tracks and disconnected reconstruction frames.
- Vertical-reference evidence is mandatory; horizontal georeferencing uses only XY
  camera controls with one isotropic 3D scale.
- Keep roads, sidewalks and physical levels separate. Unobserved areas remain unknown.
- Fit precision is not accuracy. Shared-image reconstruction backends are correlated
  evidence. Do not label their agreement as independent validation.
- Paths require explicit surface IDs and direction. Reversing direction reverses
  both signed longitudinal and rightward cross slopes.
- Field reference measurements are held out of reconstruction/fitting.

## Changes and verification

Keep changes in the core package, update method/migration docs when contracts
change, and run `.venv/bin/python -m unittest discover -s tests/slope -v`.
Optional real-model checks use `SLOPE_TEST_REAL_MODELS=1` after installing model
dependencies. Record what was actually run and distinguish software correctness,
data sufficiency and field accuracy. Do not restore retired elevation/fill code.
