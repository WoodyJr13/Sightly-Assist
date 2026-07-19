# Sightly Assist

Sightly Assist is a research project exploring offline, uncertainty-aware pedestrian hazard forecasting for blind and low-vision users.

The system combines a fast geometric risk loop with an optional event-triggered semantic module. The language model is not the safety controller. Final alerts are constrained by trajectory, clearance, uncertainty, data freshness, and sensor-health checks.

## Current integration status

The active integration branch implements:

- ONNX Runtime object detection with class filtering and NMS;
- IoU and ByteTrack tracking backends with a reproducible benchmark;
- synchronized RGB-D replay and OAK-D dataset recording;
- robust depth association and camera-frame 3D deprojection;
- temporally filtered 3D relative-velocity estimation;
- rotational camera-motion compensation;
- closest-point-of-approach collision prediction;
- conservative left, center, and right free-space analysis;
- uncertainty-aware warning actions with hysteresis and abstention;
- annotated video and stereo audio exports;
- end-to-end latency and replay evaluation;
- independent ground-truth annotation and quantitative scoring;
- an Expo Go phone application and LAN telemetry bridge;
- automated Python and mobile validation in GitHub Actions.

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md), [`docs/davidson_submission_plan.md`](docs/davidson_submission_plan.md), and [`docs/code_attribution_index.md`](docs/code_attribution_index.md) for verified progress, research framing, attribution, and limitations.

## Expo Go preview

The mobile preview supports two test modes:

1. **Phone Demo** runs a synthetic approaching/crossing-object sequence entirely on the phone.
2. **Backend Demo** connects the phone to the Python telemetry bridge over the local network.

The interface includes warning states, spoken and haptic cues, object boxes, distance, velocity, collision risk, closest-approach values, free-space corridors, synchronization status, and latency diagnostics.

### Windows one-command launch

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_expo_preview.ps1
```

The script installs the required dependencies, starts the demo bridge, prints the computer LAN address, and launches Expo. Scan the displayed QR code using Expo Go.

### macOS or Linux one-command launch

```bash
bash scripts/start_expo_preview.sh
```

### Phone-only launch

```bash
cd mobile
npm install
npm start
```

Phone-only mode begins without the Python backend or OAK-D hardware. Full setup and network troubleshooting are documented in [`mobile/README.md`](mobile/README.md).

## Python development setup

Sightly Assist requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,visualization,mobile]"
```

Run the checks:

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest
```

Run the mobile bridge directly:

```bash
sightly-mobile serve --demo
```

Run a controlled simulation:

```bash
sightly simulate scenarios/crossing.yaml --output reports/runs/crossing
```

Record an OAK-D dataset:

```bash
sightly record-oakd --sequence-id controlled-001 --output datasets/controlled-001
```

Evaluate a recorded sequence:

```bash
sightly evaluate-replay \
  datasets/controlled-001/manifest.json \
  models/detector.onnx \
  --output reports/evaluations/controlled-001
```

## Implemented pipeline

```text
RGB + aligned depth + IMU
        -> ONNX detection
        -> persistent tracking
        -> robust object depth and 3D position
        -> orientation-stabilized relative motion
        -> closest-approach collision prediction
        -> free-space and uncertainty context
        -> constrained warning policy
        -> phone telemetry, audio, video, and research metrics
```

A slower semantic model may later be triggered for ambiguous context, but it will not be allowed to issue final movement commands or override a STOP condition.

## Mobile architecture

```text
Python/OAK-D perception and risk backend
        -> versioned FastAPI/WebSocket telemetry
        -> Expo Go monitoring and warning interface
```

The Expo app does not duplicate the safety calculations or run the final ONNX model on the phone. A later Expo development build will be used when custom native background operation or lower-latency audio becomes necessary.

## Safety status

This repository contains an experimental research prototype. It is not a medical device, mobility-aid replacement, or safety guarantee. It is not suitable for independent real-world mobility decisions. Directional movement commands remain disabled.
