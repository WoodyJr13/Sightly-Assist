# Sightly Assist

Sightly Assist is a research project exploring offline, uncertainty-aware pedestrian hazard forecasting for blind and low-vision users.

The planned system combines a fast geometric risk loop with an optional event-triggered vision-language semantic module. The language model is not the safety controller. Final alerts must be constrained by trajectory, clearance, uncertainty, data freshness, and sensor-health checks.

## Current status

The first development branch implements the mathematical and software foundation before camera or edge-hardware integration:

- typed two-dimensional motion and risk schemas;
- closest-point-of-approach collision geometry;
- transparent baseline risk scoring;
- deterministic YAML scenarios;
- JSON and CSV experiment exports;
- a command-line interface;
- analytical, property-based, and regression tests;
- automated linting, formatting, typing, and test checks.

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for verified progress and limitations.

## Development setup

Sightly Assist currently requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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

The simulation command exports `summary.json` and `timeline.csv` and returns a nonzero exit code when the scenario does not match its expected hazard classification.

## Architecture

The eventual pipeline is:

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
