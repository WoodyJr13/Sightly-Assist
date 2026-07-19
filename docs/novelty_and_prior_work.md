# Novelty and Prior-Work Framework

## Purpose

This document defines what Sightly Assist may reasonably claim as original and what must be attributed to established work.

## Established components that are not claimed as novel

- convolutional or transformer object detectors;
- YOLO-family model architectures and pretrained weights;
- ONNX and ONNX Runtime;
- OpenCV image processing and video I/O;
- intersection-over-union tracking;
- ByteTrack, BoT-SORT, Kalman filtering, and optical flow;
- stereo or structured-light depth cameras;
- pinhole-camera deprojection;
- closest-point-of-approach geometry;
- visual odometry and inertial sensing;
- text-to-speech, stereo panning, and mobile dashboards.

## Proposed original contribution

Sightly Assist combines these established tools into a conservative edge-AI pipeline that:

1. estimates relative 3D trajectories from noisy RGB-D observations;
2. distinguishes likely path conflict from mere object presence;
3. explicitly withholds risk claims when motion quality is inadequate;
4. combines object trajectories with depth-derived corridor context;
5. restricts user-facing outputs to awareness, slow, stop, or abstain;
6. records enough intermediate state to audit why each warning occurred;
7. evaluates the complete system against simpler safety baselines.

## Strongest potential research claim

A hybrid geometric and learned edge-perception system can improve warning usefulness by predicting future path intersection instead of applying fixed object-presence or distance thresholds.

This claim must be supported by controlled data. It should be weakened or rejected if the experiments do not show a meaningful advantage.

## Required prior-work areas

The formal report should review primary sources in:

- assistive computer vision for blind and low-vision users;
- wearable obstacle and navigation systems;
- edge object detection;
- multi-object tracking;
- monocular and RGB-D motion estimation;
- visual-inertial odometry;
- time-to-collision and closest-approach methods;
- uncertainty calibration and selective prediction;
- free-space and traversability estimation;
- human factors for auditory warnings;
- privacy and reliability of on-device inference.

## Comparison questions for each related system

- Does it identify objects, estimate distance, or predict trajectories?
- Does it run locally or use cloud inference?
- Does it account for camera motion?
- Does it report uncertainty or abstain?
- Does it distinguish nearby objects from actual path conflicts?
- What hardware and latency does it require?
- Was it evaluated using controlled collision-like scenarios?
- Are the code, data, and metrics reproducible?

## Claim discipline

The final report must distinguish among:

- implemented features;
- experimentally verified findings;
- engineering assumptions;
- hypotheses not yet supported;
- planned future work.

Terms such as “collision probability,” “safe path,” “prevents collisions,” and “navigation system” must not be used unless the corresponding claim has been validated at that level. Until calibration is complete, the output is an interpretable risk score and constrained warning recommendation, not a certified probability or guarantee of safety.
