# Experiment Matrix

## Purpose

This matrix defines the controlled conditions required to evaluate the project beyond a single demonstration.

## Core trajectory scenarios

| ID | Scenario | Expected geometric outcome |
|---|---|---|
| T01 | Static obstacle centered ahead | Collision risk increases as wearer approaches |
| T02 | Static obstacle beside path | Low risk despite short Euclidean distance |
| T03 | Person approaching head-on | Early collision warning |
| T04 | Person receding | No collision warning after stable motion estimate |
| T05 | Person crossing left to right | Risk depends on crossing time and closest distance |
| T06 | Person crossing right to left | Mirror of T05 |
| T07 | Diagonal convergence | Collision prediction before minimum distance |
| T08 | Offset near miss | Awareness or no alert, not STOP |
| T09 | Sudden appearance from occlusion | Measure recovery and warning latency |
| T10 | Two simultaneous conflicts | Highest-severity track selected consistently |

## Environmental scenarios

| ID | Condition | Primary subsystem stressed |
|---|---|---|
| E01 | Normal indoor lighting | Reference condition |
| E02 | Low light | Detector and tracking |
| E03 | Bright backlight | Detector and depth |
| E04 | Reflective surface | Depth rejection |
| E05 | Thin pole or chair leg | Detection and depth sampling |
| E06 | Partially occluded object | Tracking and depth |
| E07 | Crowded scene | NMS, tracking, warning prioritization |
| E08 | Missing depth region | Uncertainty and abstention |
| E09 | Repeated textures | Camera-motion estimation |
| E10 | Motion blur | Detector, tracker, and egomotion |

## Camera-motion scenarios

| ID | Camera motion | Expected behavior |
|---|---|---|
| C01 | Stationary camera | Static objects remain stationary |
| C02 | Forward translation | Static scene compensated; true relative closing remains correct |
| C03 | Backward translation | Static scene compensated as receding |
| C04 | Lateral translation | Reduced false crossing motion |
| C05 | Yaw left/right | Rotation does not create false lateral object velocity |
| C06 | Pitch change | Ground and horizon changes handled conservatively |
| C07 | Walking bounce | Stable estimates or explicit abstention |
| C08 | Rapid head turn | Temporary uncertainty rather than confident warning |

## Free-space scenarios

| ID | Scene | Desired corridor result |
|---|---|---|
| F01 | Clear wide hallway | Center corridor clear |
| F02 | Center blocked, sides visible | Center blocked; side context reported without steering command |
| F03 | Narrow doorway | Narrowing detected |
| F04 | Clutter left | Left occupancy higher than center/right |
| F05 | Clutter right | Right occupancy higher than center/left |
| F06 | Unknown-depth center | Unknown/abstain rather than clear |
| F07 | Low obstacle | Ground-level obstruction detected |
| F08 | Overhead obstacle | Elevated obstruction represented separately |
| F09 | Sloped or irregular ground proxy | Low confidence or ground-model failure |
| F10 | Drop-off or missing-ground proxy | Experimental hazard flag; no safe-path claim |

## Object classes

At minimum, controlled data should include:

- person;
- bicycle or bicycle proxy;
- chair;
- box;
- table;
- pole;
- doorway;
- wall or barrier;
- hanging nonliving object;
- vehicle sequences from suitable recorded datasets where safe controlled collection is not available.

## Parameter levels

Each applicable trajectory should vary:

- starting distance: near, medium, far;
- relative speed: slow, normal walking, fast;
- lateral offset: centered, partial overlap, safe offset;
- approach angle: 0°, 30°, 60°, 90°;
- object size: small, medium, large;
- depth quality: clean, sparse, contaminated;
- detector confidence: high and borderline cases.

## Required outputs per trial

- synchronized input manifest;
- detector outputs;
- track assignments;
- depth associations;
- camera-motion estimate;
- 3D motion estimate;
- free-space result;
- risk result;
- warning decision;
- latency trace;
- annotated video;
- audio cue manifest;
- ground-truth annotation;
- failure classification if incorrect.

## Locked evaluation rule

After the final test matrix is recorded and labeled, threshold tuning must stop. Any subsequent code change requires rerunning the complete locked set and reporting the new commit SHA.
