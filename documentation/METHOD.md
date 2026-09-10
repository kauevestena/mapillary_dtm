# Measurement method and uncertainty

## What changes when elevation stops being the objective

Write a local surface as `z = aE + bN + c`. Its gradient is `(a, b)` and an upward
normal is `(-a, -b, 1) / sqrt(a²+b²+1)`.

- Maximum slope in percent: `100 * sqrt(a² + b²)`.
- Maximum slope in degrees: `atan(sqrt(a² + b²)) * 180/pi`.
- Along a horizontal unit travel vector `t`: `100 * dot((a,b), t)`.
- Across to its right: `100 * dot((a,b), (t_N, -t_E))`.

Changing `c` leaves these quantities unchanged. Uniformly scaling E, N and z
also leaves them unchanged. Scaling only z does not. Rotation relative to the
physical vertical changes slope and cannot be resolved by scale invariance.
A single image's apparent road angle and a moving camera's pitch are not ground
inclination measurements. A monocular depth map with spatially varying errors or
unknown additive depth shift is not made valid by calling slope scale-invariant.

## Required independent vertical reference

Every reconstruction record needs `vertical_reference` with:

| Key | Meaning |
| --- | --- |
| `up` | Three finite components in that reconstruction's coordinates; points **opposite gravity** |
| `source` | `calibrated_imu`, `survey_control`, or `validated_verticals` |
| `evidence` | Description/reference to the calibration, coordinate convention, timing and provenance |
| `uncertainty_deg` | Defensible angular allowance, or `null` when not quantified |

These fields deliberately have no default up vector. Acceptable ways to establish
one include calibrated and time-aligned inertial attitude; surveyed 3D control
that constrains tilt; or independently checked vertical scene lines with an
estimated calibration error. Raw accelerometer readings under acceleration are
not gravity alone. Buildings, poles and camera mounts cannot simply be assumed
perfectly vertical/level.

`reference.up_from_imu` transforms calibrated camera-frame up observations using
the corresponding native SfM rotations and returns their mean direction and
maximum angular scatter. The caller must still account for shared bias,
accelerations and sensor/camera calibration. Scatter is not an accuracy bound.

An OpenSfM/GNSS `reference_lla.json`, EXIF orientation flag or Mapillary
`computed_rotation` alone does not document this calibration. A datum/ENU label
is not proof that a reconstruction's vertical is accurate enough for accessibility.
No GPS altitude, camera-height prior, flat-road assumption or dominant ground
plane is used to establish vertical in this pipeline.

## Horizontal registration

Choose a local projected CRS in metres (usually the local UTM zone). Geographic
coordinates, feet and Web Mercator are rejected for fitting. Each reconstruction's
`georeference` supplies:

- `source`: provenance of the horizontal controls;
- `controls`: at least three distinct camera images with `image`, and either
  `x`,`y` in the project CRS or `longitude`,`latitude` in WGS84;
- `max_residual_m` (default 3 m) and `min_baseline_m` (default 10 m).

The fit estimates only yaw, horizontal translation, and one uniform scale after
vertical orientation is fixed. Robust control selection requires at least 70%
consistent controls, a baseline, and residuals within tolerance. The same scale
is applied to vertical coordinates, preserving slope angles. Vertical translation
is arbitrary and is never exported as terrain elevation.

Horizontal GNSS noise can still place a result on the wrong narrow sidewalk or
level. Registration RMSE is reported independently; survey registration or
reviewed surface/path matching may be needed. A small pixel does not solve that
positional problem. Longitudinal variation across a support window is not represented
by a single plane; use rejected support and a documented measurement scale.

## Surface evidence and physical levels

The input reconstruction must include real 3D points with measured image tracks.
Semantic votes sample those exact pixels. Visibility is not inferred by projecting
arbitrary points into nearby images. Missing/low-confidence observations stay in
the agreement denominator. The defaults require three agreeing views, semantic
probability at least 0.8, at least 80% track agreement, reprojection error at most
2 pixels, and at least 2 degrees of triangulation baseline angle.

Masks are NPZ files named `<exact-image-basename>.npz`, for example
`487967815974661.jpg.npz`. Their arrays are:

| Key | Content |
| --- | --- |
| `labels` | 2D integer class/instance IDs; 0 means unknown/non-surface |
| `confidence` | Same-size probabilities in [0,1] |
| `image_name` | Exact registered image name |
| `source` | Actual model or reviewed annotation provenance |

Use project `surfaces` to map positive label codes to unique `id` and `kind`.
Kinds are road, sidewalk, ramp, crossing, or terrain. Separate a sidewalk and road,
and separate bridge decks/underpasses even when the semantic kind is the same.
Automated SegFormer masks provide road/sidewalk/terrain kinds, not complete instance
or level separation. Manual reviewed masks use the same production input contract.
Legacy binary ground masks and mask-wide averages are not accepted substitutes.

## Local model and supported coverage

Defaults: 2 m support radius, at least 12 points, 0.5 m minimum span, 0.75 m maximum
point-to-cell support gap, 4 cm residual gate, 85% inlier fraction, condition limit
100, maximum slope 40 degrees, and maximum fit precision 2 degrees. These are
configurable **screening policies**, not achieved accuracy statements. The support
radius controls spatial averaging; the 0.5 m output cell is not the measurement
footprint. Smaller sidewalk details may remain unsupported by sparse SfM.

The fit centers coordinates, applies robust iteratively reweighted least squares,
checks inliers, and rejects line-like or ill-conditioned support. Cell corners must
be inside the inlier hull and close to actual observations. Curbs and abrupt changes
must be separated or rejected, not interpolated into ramps. No method can recover
an unobserved sidewalk's cross slope from a neighboring road plane.

## Uncertainty and directional screening

`fit_precision_deg` comes from local residuals and the fitted gradient covariance.
It omits camera calibration bias, rolling shutter, spatially correlated SfM errors,
segmentation bias and independent vertical-reference error. It is not a calibrated
confidence interval and does not shrink simply because another matching algorithm
reuses the same images.

For threshold screening, optionally supply `geometry_uncertainty_deg` and
`geometry_uncertainty_evidence` for each reconstruction. Both geometry and vertical
terms must be quantified to produce an angular allowance:

`allowance = vertical allowance + geometry allowance + 1.96 * fit precision`.

This is a conservative engineering budget whose validity depends on supplied
calibration; the factor 1.96 does not make it a 95% confidence interval. Fusion uses
the largest input allowance plus full-normal disagreement. Unknown inputs remain
unknown. Directional screening projects the normal's uncertainty cone into the
longitudinal/cross-slope plane before testing the user's threshold; it does not
naively reuse the normal-angle allowance for a steep perpendicular slope.

With `paths`, set explicit `screening_thresholds.longitudinal_pct` and
`screening_thresholds.cross_slope_pct`. Outputs are `below_threshold`,
`above_threshold`, `uncertain`, or `uncertainty_unknown`; missing geometry is
`unknown`. Those statuses concern the chosen numerical threshold only. Route
accessibility also depends on width, surface, steps, obstacles and other factors.

## Primary format references

- [OpenSfM coordinate conventions](https://opensfm.readthedocs.io/en/latest/cam_coord_system.html):
  world-to-camera axis-angle rotations; center `C = -Rᵀt`; camera right/down/forward.
- [COLMAP model format](https://colmap.github.io/format.html): text point tracks and
  `(qw,qx,qy,qz)` world-to-camera quaternion convention; actual keypoint indices.
- [Mapillary spatial metadata](https://mapillary.github.io/mapillary-js/api/interfaces/api.SpatialImageEnt/):
  altitude, computed geometry/rotation and reconstruction metadata descriptions.
- [SegFormer model card](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024):
  the default optional surface segmentation model; check its source license.
