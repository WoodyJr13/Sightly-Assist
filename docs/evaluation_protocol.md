# Evaluation Protocol

## Goal

Determine whether Sightly Assist improves collision-hazard warning quality over simpler approaches while remaining fast enough for edge deployment.

## Baselines

### B0: Detection only

Warn when a relevant object class is detected above a confidence threshold.

### B1: Distance threshold

Warn when a valid depth estimate is below a fixed threshold.

### B2: Frame-to-frame motion

Estimate approach from consecutive depth measurements without persistent temporal fitting or camera-motion compensation.

### B3: Geometric trajectory baseline

Use tracking, robust depth, and temporal motion with closest-point-of-approach geometry, but without uncertainty gating, free-space context, or camera-motion compensation.

### F: Full Sightly Assist

Use the complete tested configuration: detection, tracking, robust depth, temporal motion, camera-motion compensation, collision geometry, uncertainty gating, free-space context, and constrained warning policy.

## Units of evaluation

- frame-level observations;
- continuous object tracks;
- complete scenario trials;
- warning events;
- system sessions.

Primary conclusions should be based on scenario-level or event-level outcomes rather than inflated frame counts.

## Ground truth

Ground truth should be generated using one or more of:

- precisely scripted synthetic scenarios;
- measured object and camera paths;
- marked floor coordinates and synchronized timestamps;
- independent reference video;
- known static geometry;
- manually reviewed event annotations by at least two passes;
- hardware reference measurements when available.

Ground-truth definitions and annotation rules must be frozen before final evaluation.

## Primary metrics

### Collision-event performance

- precision;
- recall;
- F1 score;
- false warnings per minute;
- missed hazardous events;
- late-warning rate;
- median and lower-quartile warning lead time.

### Prediction accuracy

- time-to-closest-approach absolute error;
- closest-distance absolute error;
- predicted-collision classification accuracy;
- calibration curve if the engineering risk score is later calibrated.

### Selective reliability

- abstention rate;
- error rate on non-abstained predictions;
- unsafe confident-error rate;
- risk-coverage curve.

### Depth and motion

- median absolute depth error;
- valid-depth percentage;
- velocity-vector error;
- direction-angle error;
- stationary-object false-speed distribution;
- warm-up time to a valid estimate.

### Tracking

- track recall;
- identity switches;
- fragmentation;
- lost-track duration;
- reacquisition time.

### Runtime

- per-stage latency;
- end-to-end latency;
- achieved frames per second;
- dropped-frame rate;
- memory use;
- CPU/GPU utilization;
- power and thermal measurements on final hardware.

## Scenario split

Use separate development and final-evaluation scenarios. Final test sequences must not be used to tune thresholds.

Recommended split:

- 60% development and calibration;
- 20% validation and threshold selection;
- 20% locked final evaluation.

Where data volume is small, use grouped cross-validation by physical setup or recording session so nearly identical frames do not leak across splits.

## Repetition

Each controlled condition should be repeated across:

- multiple speeds;
- multiple approach angles;
- multiple starting distances;
- multiple lighting conditions;
- multiple object classes or sizes;
- multiple camera motion patterns.

Record the random seed and exact configuration for every generated or simulated trial.

## Statistical reporting

Report:

- sample count and scenario count;
- median and interquartile range for skewed timing/error data;
- mean and standard deviation where appropriate;
- bootstrap confidence intervals for key event-level metrics;
- paired comparisons when the same scenario is processed by multiple baselines;
- effect sizes, not only statistical significance.

## Ablation tests

Evaluate the full system after removing one component at a time:

- robust depth filtering;
- persistent tracking;
- temporal motion fitting;
- camera-motion compensation;
- uncertainty gate;
- free-space context;
- warning hysteresis;
- class-specific collision geometry.

## Failure taxonomy

Every false warning, missed hazard, and abstention should be assigned one primary cause:

- detector miss or misclassification;
- tracker identity failure;
- depth missing or contaminated;
- camera-motion failure;
- motion-estimation error;
- collision-model assumption;
- uncertainty threshold;
- warning-policy threshold;
- ground-truth ambiguity;
- runtime overload or dropped data.

## Reproducibility requirements

Each reported result must identify:

- Git commit SHA;
- model filename and checksum;
- hardware and software versions;
- scenario manifest;
- configuration file;
- random seed;
- raw result file;
- analysis script and output figure.
