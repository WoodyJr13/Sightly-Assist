# Architecture

## Goal

Sightly Assist is being developed as an offline research platform for predicting short-horizon pedestrian collision hazards. The eventual system will combine fast geometric reasoning with slower, event-triggered semantic reasoning.

## Planned data flow

```text
RGB + depth + IMU
        |
        v
sensor synchronization and health checks
        |
        v
object detection and tracking
        |
        v
depth association and ego-motion compensation
        |
        v
relative trajectory estimation
        |
        v
closest approach, collision risk, and free-space corridors
        |                         \
        |                          -> event-triggered semantic VLM
        |                                      |
        +--------------------------------------+
                           |
                           v
                 constrained safety policy
                           |
                           v
                  nonblocking audio alerts
```

## Current implementation

The first milestone implements only the deterministic mathematical core:

- two-dimensional ground-plane positions;
- constant-velocity observer and obstacle motion;
- relative position and velocity;
- time to closest approach;
- separation at closest approach;
- radial closing speed;
- circular safety boundaries;
- a transparent, uncalibrated baseline risk score;
- YAML scenario loading and reproducible JSON/CSV output.

## Safety boundary

The vision-language model is not allowed to produce final movement commands. Its future role is limited to structured semantic evidence. Final alerts must remain constrained by geometry, uncertainty, corridor clearance, data freshness, and sensor health.

## Runtime modes

The full project will support:

- `SIMULATION`: exact synthetic ground truth;
- `REPLAY`: recorded sensor data;
- `LIVE`: physical sensors and edge hardware;
- `BENCHMARK`: repeatable performance and accuracy measurements.

Simulation and replay must continue to work even when optional hardware dependencies are unavailable.
