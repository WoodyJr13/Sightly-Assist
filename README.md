# Sightly Assist

Sightly Assist is a research project exploring offline, uncertainty-aware pedestrian hazard forecasting for blind and low-vision users.

The planned system combines a fast geometric risk loop with an optional event-triggered vision-language semantic module. The language model is not the safety controller. Final alerts must be constrained by trajectory, clearance, uncertainty, data freshness, and sensor-health checks.

## Current status

The active development branches implement:

- typed two-dimensional motion and risk schemas;
- closest-point-of-approach collision geometry;
- transparent baseline risk scoring;
- deterministic YAML scenarios and top-down visualization;
- recorded RGB/RGB-D replay manifests;
- OpenCV image decoding;
- a real ONNX Runtime YOLO-style detector backend;
- confidence filtering and class-aware non-maximum suppression;
- a deterministic IoU tracking baseline;
- robust per-track depth association;
- deprojection into camera-frame 3D positions;
- analytical, property-based, integration, and real-runtime tests;
- automated linting, formatting, strict typing, and test checks.

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for verified progress and limitations. Detector and depth details are documented in [`docs/perception_and_depth.md`](docs/perception_and_depth.md).

## Development setup

Sightly Assist currently requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For detector runtime dependencies without the full development toolchain:

```bash
python -m pip install -e ".[detector]"
```

Run the checks:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

Run a controlled simulation:

```bash
sightly doctor
sightly simulate scenarios/crossing.yaml --output reports/runs/crossing
```

The simulation command exports `summary.json`, `timeline.csv`, and a top-down PNG. It returns a nonzero exit code when the scenario does not match its expected hazard classification.

## Perception architecture

The implemented recorded-data path is:

```text
RGB frame
        -> ONNX Runtime detector
        -> class filtering and NMS
        -> persistent object tracking
        -> synchronized depth association
        -> camera-frame 3D object position
```

The next milestone will estimate temporally filtered 3D velocity, compensate for camera ego-motion, and connect measured object motion to the existing closest-approach risk engine.

The eventual full pipeline is:

```text
RGB + depth + IMU
        -> detection and tracking
        -> relative trajectory estimation
        -> collision risk and free-space analysis
        -> constrained warning policy
        -> nonblocking audio alerts
```

A slower semantic model may be triggered for ambiguous context, but it will not be allowed to issue final movement commands.

## Safety status

This repository contains an experimental research prototype. It is not a medical device, mobility-aid replacement, or safety guarantee. It is not suitable for real-world mobility decisions. Directional movement guidance is disabled by default.
