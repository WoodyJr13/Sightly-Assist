# Replay Evaluation Procedure

## Purpose

The replay evaluator runs one recorded RGB-D sequence through the same ordered perception and safety pipeline used by the project. It is intended to produce reproducible engineering evidence for model comparisons, tracker comparisons, ablation studies, and final hardware measurements.

The evaluator does not establish that the system is safe for independent mobility use. It records what the current implementation produced under a documented configuration.

## Command

```text
sightly evaluate-replay \
  datasets/hallway-head-on-001/manifest.json \
  models/detector.onnx \
  --output reports/evaluations/hallway-head-on-001 \
  --provider CPUExecutionProvider
```

Useful controls:

```text
--video / --no-video
--audio / --no-audio
--checksums / --no-checksums
--require-orientation / --allow-missing-orientation
--overwrite
```

Checksum verification should remain enabled for formal experiments. `--no-checksums` exists for legacy or synthetic datasets that do not contain stored file hashes.

## Ordered evaluation stack

```text
manifest and file integrity
          ↓
RGB decoding
          ↓
ONNX detector
          ↓
ByteTrack
          ↓
object-depth association
          ↓
coarse free-space analysis
          ↓
rotation-compensated temporal 3D motion
          ↓
closest-point-of-approach collision risk
          ↓
constrained warning policy
          ↓
JSON, video, audio, and latency evidence
```

The detector model is fingerprinted with SHA-256. This prevents two evaluations using different weight files from being treated as the same experiment merely because the filenames match.

## Output layout

```text
<evaluation>/
├── summary.json
├── frame_results.jsonl
├── annotated.mp4              # optional
└── audio/                     # optional
    ├── audio_manifest.json
    └── *.wav
```

### `frame_results.jsonl`

One JSON object is written per input frame. Each record contains:

- original frame metadata and synchronization status;
- detector outputs;
- persistent tracks;
- object-depth associations;
- free-space analysis;
- temporal motion estimates;
- collision-risk results;
- warning-policy decision;
- measured stage timings.

JSON Lines is used so large recordings can be processed incrementally by later analysis tools without loading the entire result into memory.

### `summary.json`

The aggregate summary records:

- sequence and source identifiers;
- detector-model SHA-256;
- frame count and recorded duration;
- detection and track-observation counts;
- valid depth, motion, and risk counts;
- frames containing a predicted collision;
- warning-state counts and emitted-warning count;
- center-corridor free-space counts;
- synchronized-frame count;
- estimated processing FPS and real-time factor;
- mean, median, P95, and maximum timing for every processing stage;
- paths to generated evidence files.

## Timing definitions

The following wall-clock stages are measured independently:

- image loading;
- detector inference and decoding;
- tracking;
- depth loading;
- depth association;
- free-space analysis;
- temporal motion estimation;
- collision-risk evaluation;
- warning-policy evaluation;
- total frame processing.

Timing uses a monotonic high-resolution host clock. File export, annotated-video rendering, and audio synthesis occur after pipeline processing and are not included in per-frame inference latency.

Shared CI runner timings are regression diagnostics only. Formal performance claims must use repeated measurements on the final Jetson and OAK-D configuration with:

- model warm-up separated from measured frames;
- fixed power mode and clock policy;
- recorded device temperature;
- execution provider and precision documented;
- at least three repeated runs;
- median and tail latency reported;
- dropped frames and synchronization failures reported;
- background processes controlled as far as practical.

## Required experiment identity

A formal result should record, outside or alongside the generated files:

- repository commit SHA;
- dataset manifest SHA-256;
- detector model SHA-256;
- ONNX Runtime version and execution provider;
- tracker implementation and parameters;
- depth, free-space, motion, risk, and warning configuration;
- hardware model, power mode, and software image;
- operator;
- date and environment;
- scenario and ground-truth source;
- whether orientation was required;
- whether media rendering was enabled.

## Interpreting counts

Counts are descriptive pipeline outputs, not accuracy metrics. For example, a high valid-risk fraction does not prove that the risk predictions are correct. Accuracy requires independent ground truth for object identity, depth, trajectory, collision state, and event timing.

The next evaluation layer must compare exported predictions with locked ground-truth annotations to calculate event precision, recall, warning lead time, closest-approach error, and false warnings per minute.

## Failure behavior

The evaluator stops before inference when:

- the manifest is missing;
- the model is missing;
- required replay files are absent;
- camera intrinsics are unavailable;
- a stored file checksum does not match;
- the output directory exists without explicit overwrite permission.

When orientation is required, frames without orientation do not silently fall back to uncompensated motion. Motion remains unavailable until trustworthy orientation data is present.

## Current limitations

- No final detector model or weight source has been selected.
- Ground-truth scoring is not included in this milestone.
- Video codecs vary by host environment.
- The evaluator processes a recording offline and is not yet the live OAK-D application loop.
- Current free-space analysis is coarse and does not identify drop-offs, stairs, floor geometry, or overhead hazards.
- Generated audio is an offline evidence artifact rather than the final low-latency playback system.
