# Project Status

## Current milestone

Milestone 2: recorded-data perception, real ONNX object detection, tracking baseline, and robust depth association.

## Completed

- Repository and Python package foundation
- Typed ground-plane scenario and risk schemas
- Constant-velocity closest-point-of-approach geometry
- Transparent baseline risk score
- YAML-driven deterministic simulator
- JSON, CSV, and top-down PNG simulation exports
- Six controlled regression scenarios
- GitHub Actions installation, linting, formatting, strict typing, and tests
- Typed replay manifests and synchronized frame metadata
- RGB and NumPy depth-file validation
- OpenCV image decoding with manifest dimension checks
- Hardware-independent detector and tracker contracts
- Real ONNX Runtime YOLO-style detector backend
- Letterbox preprocessing and source-coordinate restoration
- Raw Ultralytics, YOLOv5-style, and end-to-end NMS output decoding
- Configurable confidence and hazard-class filtering
- Class-aware non-maximum suppression
- Deterministic IoU tracking baseline with persistent IDs and expiration
- Robust central-region depth association
- Invalid-depth and extreme-outlier rejection
- Median depth, valid fraction, inlier count, and MAD uncertainty statistics
- Camera-intrinsic deprojection to camera-frame 3D coordinates
- Replay integration from RGB decoding through detection, tracking, and depth association
- Actual ONNX Runtime inference test using a generated model graph

## In progress

- Select and benchmark a production detector weight file
- Compare the IoU tracker with ByteTrack or another motion-aware tracker
- Export replay detections, tracks, depth, and timing measurements for experiments

## Not started

- Per-track temporal depth filtering
- 3D velocity and acceleration estimation
- Camera ego-motion compensation using IMU or visual odometry
- Uncertainty propagation and Monte Carlo collision probability
- Free-space corridor estimation
- Safety policy state machine
- Event-triggered semantic VLM
- Audio alerts
- Jetson deployment and TensorRT benchmarks
- Controlled physical benchmark dataset

## Known limitations

- The repository does not include a large detector weight file; a compatible ONNX model is supplied locally.
- Detector behavior still needs evaluation on project-specific pedestrian hazard recordings.
- The IoU tracker is a transparent baseline rather than the intended final tracker.
- Depth association is frame-local and is not yet temporally filtered.
- Camera ego-motion is not yet removed from tracked-object motion.
- The simulation risk score is an engineering baseline, not a calibrated probability.
- The prototype is not suitable for mobility decisions or real-world safety use.

## Next milestone acceptance criteria

1. A selected lightweight detector is benchmarked on representative recorded scenes.
2. Tracking identity stability and latency are compared against the IoU baseline.
3. Per-track 3D position history is converted into filtered relative velocity.
4. Ego-motion compensation prevents camera movement from appearing as obstacle movement.
5. Tracked 3D motion feeds the closest-approach risk engine with explicit uncertainty.
