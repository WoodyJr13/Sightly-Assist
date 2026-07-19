# Project Status

## Current milestone

Milestone 0–1 foundation: repository, typed schemas, deterministic simulation, collision geometry, baseline risk scoring, regression scenarios, and CI.

## Completed

- Repository initialization and protected development branch workflow
- Python package configuration
- Typed ground-plane scenario and risk schemas
- Constant-velocity closest-point-of-approach geometry
- Transparent baseline risk score
- YAML-driven deterministic simulator
- JSON and CSV run exports
- CLI environment and simulation commands
- Six controlled regression scenarios
- Analytical, property-based, and scenario tests
- GitHub Actions lint, formatting, typing, and test workflow

## In progress

- Validate the first CI run and repair any failures
- Add top-down visualization output
- Expand mathematical documentation

## Not started

- Recorded-video replay pipeline
- Object detector and tracker interfaces
- RGB-D sensor integration
- Uncertainty propagation and Monte Carlo collision probability
- Free-space corridor estimation
- Safety policy state machine
- Event-triggered semantic VLM
- Audio alerts
- Jetson deployment
- Controlled physical benchmark dataset

## Known limitations

- Motion is currently two-dimensional and constant velocity.
- Risk score is a transparent engineering baseline, not a calibrated probability.
- No camera, depth, IMU, detector, tracker, VLM, or audio component is connected.
- The prototype is not suitable for mobility decisions or real-world safety use.

## Next milestone acceptance criteria

1. All CI checks pass.
2. Every scenario produces reproducible JSON and CSV outputs.
3. A top-down plot is generated without changing simulation results.
4. Mathematical assumptions and units are documented.
5. Safe and hazardous scenarios remain covered by regression tests.
