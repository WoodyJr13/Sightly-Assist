# Ground-Truth Annotation and Quantitative Scoring

## Purpose

This protocol separates independent evidence from Sightly Assist predictions. A detector box, estimated depth, fitted velocity, risk output, or warning must never be copied into the ground-truth file and then used to score the same system.

The annotation and scoring version introduced here is `1.0`.

## Files

A scored experiment uses three primary files:

```text
recording/manifest.json
annotations/ground_truth.json
reports/evaluation/frame_results.jsonl
```

The scoring command produces:

```text
reports/scores/score.json
```

The output records SHA-256 fingerprints of the annotation and result files so the evidence pair used for a reported result can be recovered exactly.

## Create an annotation template

```bash
sightly init-annotations \
  recording/manifest.json \
  --annotator ved \
  --output annotations/ground_truth.json
```

The template contains one empty entry for every recorded frame, preserving the manifest frame ID and nanosecond timestamp.

## Coordinate frame

Three-dimensional labels use the same right/down/forward camera convention as the perception pipeline:

- positive `x`: camera right;
- positive `y`: camera down;
- positive `z`: forward from the camera;
- velocity units: meters per second;
- position and depth units: meters.

Orientation-compensated experiments must document whether the external ground truth is expressed in the initial stabilized camera axes or the instantaneous camera axes. Do not mix coordinate frames within a sequence.

## Object annotations

Each visible physical object may contain:

- a persistent `object_id` that remains unchanged throughout the sequence;
- `class_name`;
- a pixel bounding box;
- independently measured forward depth;
- independently measured 3D position;
- independently measured 3D velocity;
- occlusion and ignore flags;
- notes.

`depth_m` and `position.z_m`, when both supplied, must agree within 0.05 m. This validation catches unit and transcription mistakes; it is not an accuracy claim.

### Acceptable sources of quantitative labels

Examples include:

- surveyed or tape-measured static geometry;
- marked floor positions with known timestamps;
- rigid motion rigs with controlled speed;
- external motion capture;
- independently calibrated ranging equipment;
- synthetic scenarios whose geometry is generated directly from locked parameters.

A label source, calibration procedure, uncertainty estimate, and operator should be recorded in metadata or experiment notes.

## Ignore regions

Use `ignore: true` only when the region is intentionally excluded from scoring, such as an ambiguous reflection, privacy mask, or object outside the experiment definition. Ignore regions remain visible to the scorer so overlapping predicted tracks are not incorrectly counted as false positives.

Ignore flags must not be added after viewing a poor model result merely to improve a score.

## Hazard events

A hazard event identifies:

- the physical `object_id`;
- event start time;
- critical collision or closest-approach time;
- event end time;
- minimum acceptable warning action.

The critical timestamp is used for warning lead time. It should come from the staged experiment geometry or external measurement, not the model's own time-to-closest-approach prediction.

A successful warning must:

1. be emitted rather than only held internally;
2. refer to the matched event object;
3. meet or exceed the required action severity;
4. occur no later than the critical timestamp.

Severity ordering is:

```text
awareness < slow < stop
```

`NO_ALERT` and `ABSTAIN` cannot be minimum event-warning actions.

## Matching

The transparent baseline scorer uses greedy one-to-one bounding-box matching per frame.

Default behavior:

- class-aware matching;
- minimum intersection-over-union of 0.30;
- candidates ordered from highest to lowest IoU;
- each ground-truth object and predicted track used at most once;
- ignored regions removed from the predicted-track denominator.

Run a class-agnostic sensitivity analysis only as a separately named experiment.

## Metrics

### Tracking

- track precision: matched predicted track observations divided by scorable predicted observations;
- track recall: matched ground-truth object observations divided by labeled object observations;
- identity switches: a physical object changes matched predicted track ID;
- fragmentations: a previously matched visible object becomes unmatched and later matches again.

### Geometry

When corresponding independent labels are available, the report includes count, signed mean error, mean absolute error, median absolute error, root-mean-square error, 95th-percentile absolute error, and maximum absolute error for:

- forward depth;
- 3D position magnitude;
- 3D velocity magnitude;
- time to closest approach;
- distance at closest approach.

Closest-approach ground truth is derived from labeled 3D position and velocity with the same documented safety boundary used by the evaluated configuration. Frames without independent position and velocity labels are excluded from CPA error metrics rather than treated as correct.

### Collision frames

Collision-frame precision, recall, and F1 compare each risk-eligible labeled object-frame against the matched prediction. A collision-positive prediction on an unmatched, non-ignored track is a false positive.

### Collision events

Contiguous collision-positive frames sharing a predicted track ID form one predicted event. A predicted event is correct when it overlaps the labeled interval for the same matched physical object.

The report includes:

- number of labeled events;
- number of predicted events;
- number of detected labeled events;
- event precision, recall, and F1.

### Warnings

The report includes:

- warning-event recall;
- warning lead-time distribution;
- false warning count;
- false warnings per minute;
- abstention-frame count.

An emitted hazard warning is considered false when its matched object has no active labeled event at that timestamp. Abstention is reported separately and is not counted as a hazard warning.

## Score an evaluation

```bash
sightly score-replay \
  annotations/ground_truth.json \
  reports/evaluation/frame_results.jsonl \
  --output reports/scores/score.json
```

Optional matching sensitivity run:

```bash
sightly score-replay \
  annotations/ground_truth.json \
  reports/evaluation/frame_results.jsonl \
  --match-iou 0.50 \
  --output reports/scores/score-iou-050.json
```

Do not replace the primary preregistered threshold after seeing the result. Additional thresholds should be labeled as sensitivity analyses.

## Required experiment record

For every reported score, preserve:

- recording manifest and checksums;
- ground-truth JSON and checksum;
- frame-result JSONL and checksum;
- score JSON;
- detector model checksum;
- software commit SHA;
- hardware and execution provider;
- scoring configuration;
- annotator and verification status;
- measurement and calibration notes;
- exclusions and failures.

## Quality control

At least a subset of the final dataset should be independently reviewed. For high-impact metrics, use double annotation or an instrumented reference where practical. Disagreements must be resolved by a documented rule rather than by choosing the label that favors the system.

## Current limitations

- the annotation interface is JSON rather than graphical;
- matching is a transparent baseline, not HOTA or TrackEval;
- event grouping is track-ID based and can split when tracking fragments;
- uncertainty of the ground-truth measurement is not yet propagated into confidence intervals;
- statistical aggregation across recordings is a later milestone;
- real-world mobility decisions remain outside the validated scope.
