# Fix Runtime Readiness Roadmap

This document lays out an incremental plan to bring the pipeline from the current synthetic scaffolding to a deployable, end-to-end runtime. Each milestone builds on the previous one; do not skip ahead.

## Milestone 0 — Baseline Audit
- [x] Inventory external tooling requirements (COLMAP CLI, OpenSfM runner, GPU drivers if needed) and record versions we will support. _See `docs/runtime_baseline_audit.md`._
- [x] Document current stubs versus real integrations for SfM, VO, densification, and Mapillary ingestion. _See `docs/runtime_baseline_audit.md`._
- [x] Capture an example run configuration (AOI, expected outputs, tokens) we will target during the rollout. _See `docs/runtime_baseline_audit.md`._

## Milestone 1 — Environment & Dependency Harden
- [x] Define a reproducible environment spec: Python version, system packages, CUDA requirements, and optional toolchains. _See `docs/runtime_environment.md`._
- [x] Extend `requirements.txt` or add a `requirements-dev.txt` for heavy geo/ML deps that are currently implied. _See `requirements.txt`, `requirements-optional.txt`, `requirements-dev.txt`._
- [x] Add automated checks (e.g., `scripts/check_env.py`) that verify optional binaries and provide actionable errors instead of silent fallbacks. _Implemented in `scripts/check_env.py`._

## Milestone 2 — Real Data Ingestion
- [ ] Replace synthetic sequence discovery with Mapillary API calls; move tokens to config (`MAPILLARY_TOKEN` env, `.env` file, or secrets manager).
- [ ] Implement caching of raw metadata/imagery with clear directory layout and size guards.
- [ ] Add unit tests that mock API responses to keep CI deterministic.

## Milestone 3 — OpenSfM Track Activation
- [ ] Integrate the true OpenSfM pipeline: data adapters, config generation, invocation, and product ingestion (poses, cameras, tracks).
- [ ] Provide fallback paths for environments without OpenSfM (e.g., raising informative errors or running a trimmed synthetic variant).
- [ ] Add regression tests around the interface layer—e.g., verifying we parse poses/cameras/points correctly from a canned OpenSfM output bundle.

## Milestone 4 — COLMAP Track Activation
- [ ] Mirror Milestone 3 for COLMAP: prepare project directories, run sparse reconstruction, extract cameras/points.
- [ ] Ensure COLMAP outputs respect the same coordinate frames as OpenSfM; add consistency checks in `ReconstructionResult`.
- [ ] Surface configuration knobs (threads, GPU usage) via CLI flags and document defaults.

## Milestone 5 — VO + Dense Support
- [ ] Implement the VO chain against real image streams (e.g., OpenCV-based feature tracking) and expose scale metadata to the solver.
- [ ] Wire in mono-depth or plane-sweep modules using actual models; handle optional GPU acceleration.
- [ ] Expand `label_and_filter_points` to consume real VO/dense outputs and tag them with uncertainty estimates.

## Milestone 6 — Metric Scale & Consensus Validation
- [ ] Revisit `solve_scale_and_h` with real inputs; add numerical stability checks and clear error messages when constraints fail.
- [ ] Write integration tests around consensus voting using captured fixture datasets (store lightweight subsets under `qa/data/`).
- [ ] Document expected failure modes (insufficient overlap, GPS gaps) and how the pipeline responds.

## Milestone 7 — Breaklines & TIN Enforcement
- [ ] Validate curb extraction and 3D projection using real detections; add visualization/debug utilities to inspect results quickly.
- [ ] Ensure constrained TIN builds succeed with real breakline data, and add tests to guard against regressions.
- [ ] Tune parameters (`BREAKLINE_*`, `MAX_TIN_EXTRAPOLATION_M`) based on empirical runs and record guidance in docs.

## Milestone 8 — QA & Reporting
- [ ] Re-enable learned uncertainty calibration with real training loops or disable it behind a feature gate until data is ready.
- [ ] Expand QA outputs (`qa_internal`, `qa_external`, HTML report) to handle the richer data products and flag anomalies.
- [ ] Add smoke tests that run a tiny AOI end-to-end (using pre-downloaded imagery) to verify the CLI before releases.

## Milestone 9 — Operationalization
- [ ] Create deployment scripts or containers that encapsulate all dependencies and expected runtime flags.
- [ ] Add monitoring hooks (logging, metrics) for long AOI runs so operators can track progress and diagnose stalls.
- [ ] Publish updated documentation (README, agents guide, ops handbook) summarizing the new runtime expectations and maintenance tasks.

## Milestone 10 — Continuous Validation
- [ ] Automate nightly or weekly regression runs on a representative AOI; compare outputs against reference metrics (height RMSE, slope accuracy).
- [ ] Integrate alerts when QA thresholds are breached or stages abort unexpectedly.
- [ ] Schedule periodic dependency audits and hardware validation to keep the runtime healthy over time.
