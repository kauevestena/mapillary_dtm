# Migration from elevation to slope

This is an intentional breaking redesign. The GitHub repository URL is unchanged.

| Previous workflow | Replacement |
| --- | --- |
| `python -m dtm_from_mapillary.cli.pipeline run --aoi-bbox ...` | Prepare a `slope_project.json`; `python -m slope_from_mapillary run ...` |
| Ellipsoidal DTM as primary output | Observed slope and gradient fields by physical surface |
| Camera-height and GNSS-altitude scale/height solver | Independent vertical orientation and horizontal-only similarity |
| Camera-relative synthetic ground candidates | Real measured 3D points with native feature tracks |
| Binary ground masks | Class/instance labels at the observed feature pixels |
| Lower-envelope elevation fusion and smoothing | Local plane fitting with explicit support/rejection |
| TIN fill, corridor extrapolation and inner-block fill | Unknown cells remain NoData |
| Two matching backends count as independent | Shared images belong to the same evidence group |
| Height residual QA | Signed longitudinal/cross-slope comparisons and coverage |
| Automatic CUDA/ML setup for everything | Lightweight estimator; separate optional segmentation/reconstruction dependencies |

The old command module delegates to the new parser. Old elevation flags are rejected.
Retired measurement routines and their tests are available in Git history, rather
than continuing as a competing pipeline. Tracked raw imagery, reconstructions,
reference DTMs and historical results remain preserved for reproducibility.

Keep cached reconstructions only if camera calibration, projection model, feature
tracks and component boundaries are understood. They are **input geometry**, not
calibration evidence. Do not feed the old generated ground points or interpolated
DTM into this image-based estimator as though they were observations.

Existing binary SfM runner integrations and Mapillary acquisition utilities are
retained. The two runners now parse cached geometry through the new production
readers and label it `reconstruction`, not `enu`. The new primary package can also
consume reconstructions produced outside this repository.

For an initial field trial, collect a short sidewalk sequence with calibrated
vertical orientation and independent longitudinal/cross-slope measurements. Include
both flat and sloping patches, ramps, a curb boundary, occlusions and repeated
passes. Select accuracy/coverage targets from the intended accessibility application
before tuning acceptance thresholds on training/calibration data.
