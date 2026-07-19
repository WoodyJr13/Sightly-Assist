# Code Attribution Index

## Purpose

This index will become the basis of the final Davidson code documentation. Update it whenever a new package, model, algorithm, adapted example, or generated component enters the repository.

## Original project modules

The following modules contain project-specific integration, validation, safety policy, experiment logic, or mathematical application. Their final authorship status must be documented in the assistance ledger.

| Area | Current module or location | Function |
|---|---|---|
| Geometry and risk | `src/sightly_assist/geometry.py`, `risk.py` | Relative motion, closest approach, collision boundary, engineering risk score |
| Simulation | `simulation.py`, `scenarios/` | Deterministic synthetic scenario execution and exports |
| Perception contracts | `perception.py` | Detector, tracker, frame, box, and observation interfaces |
| Replay | `replay.py`, `replay_pipeline.py` | Recorded RGB-D sequence validation and processing |
| Baseline tracking | `iou_tracker.py` | Transparent class-aware IoU assignment baseline |
| ByteTrack integration | `bytetrack_adapter.py` | Conversion between project detections and the acquired Roboflow ByteTrack backend |
| Tracking evaluation | `tracking_benchmark.py`, `tracking_scenarios.py`, `tracking_report.py` | Locked scenarios, MOT metrics, latency, downstream collision metrics, and JSON reporting |
| Depth association | `depth_association.py` | Robust ROI depth filtering and 3D deprojection |
| Motion estimation | `motion_estimation.py` | Multi-frame robust 3D velocity estimation |
| Perception risk | `perception_risk.py` | Conversion of measured tracks into CPA risk results |
| Warning policy | `warning_policy.py` | Constrained state machine, hysteresis, abstention, repeat control |
| Visualization | `visualization.py`, `annotated_video.py` | Top-down plots and annotated replay exports |
| Audio | `audio_cues.py` | Deterministic stereo warning-cue generation and WAV export |
| ONNX detector adapter | `onnx_detector.py` | Preprocessing, runtime invocation, output decoding, class filtering, NMS |

## Third-party libraries

| Library | Role | Acquisition status | Required final attribution |
|---|---|---|---|
| NumPy | Numerical arrays and statistics | Acquired dependency | Version, license, project URL |
| Pydantic | Runtime schemas and validation | Acquired dependency | Version, license, project URL |
| OpenCV | Image decoding, drawing, video export | Optional acquired dependency | Version, license, project URL |
| ONNX Runtime | Neural-model execution | Optional acquired dependency | Version, license, project URL |
| Roboflow `trackers` | Acquired ByteTrack implementation | Optional acquired dependency pinned to 2.5.0 | Version, Apache-2.0 license, repository, ByteTrack paper |
| Supervision | Detection container used by `trackers` | Transitive acquired dependency | Version, license, project URL |
| PyYAML | Scenario configuration | Acquired dependency | Version, license, project URL |
| Typer | Command-line interface | Acquired dependency | Version, license, project URL |
| Matplotlib | Simulation visualization | Acquired dependency | Version, license, project URL |
| Pytest/Hypothesis | Testing | Development dependencies | Versions, licenses, project URLs |

The repository does not claim the ByteTrack algorithm or the Roboflow implementation as original work. The original project work in this area is the typed adapter, locked scenario design, evaluation logic, downstream risk coupling, and interpretation of results.

## Models and datasets

No final detector weights or research dataset should be considered selected until an entry records:

- model name and version;
- architecture source;
- weights source and checksum;
- training dataset;
- license and redistribution terms;
- export procedure;
- modifications or fine-tuning;
- benchmark configuration.

## Adapted algorithms

For every algorithm based on a paper, textbook, documentation example, or external repository, record:

- exact source;
- relevant section, equation, or file;
- what was reused conceptually;
- what code, if any, was adapted;
- what changed in the project implementation;
- tests used to verify behavior.

## AI-assisted development

AI-generated or AI-revised code must not be described as independently authored without qualification. For each major component, the final documentation should state:

- whether AI proposed the design;
- whether AI generated an initial implementation;
- what the applicant personally reviewed, rewrote, tested, or rejected;
- how correctness was independently established;
- whether the applicant can explain and reproduce the central method.

## Final package conversion

Before submission, convert this living Markdown index into a clean PDF containing:

- directory tree;
- main entry point;
- module-by-module descriptions;
- original versus acquired code labels;
- dependency and model licenses;
- experiment and test locations;
- reproducibility commands;
- repository commit SHA represented in the package.
