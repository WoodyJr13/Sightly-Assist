# Engineering Decisions

## 2026-07-18 — Use a hybrid safety architecture

**Decision:** Build a fast geometric trajectory and risk loop, with an optional event-triggered vision-language model for semantic context.

**Reason:** A generative model should not be the sole source of time-critical movement guidance. Geometry, clearance, uncertainty, and sensor health must constrain all alerts.

## 2026-07-18 — Begin with deterministic 2D simulation

**Decision:** Validate constant-velocity ground-plane mathematics before integrating cameras or neural models.

**Alternatives considered:** Start directly with object detection or Jetson deployment.

**Reason:** Synthetic scenarios provide exact ground truth, reproducible failures, and analytical tests. This isolates mathematical errors before sensor noise is introduced.

## 2026-07-18 — Keep risk scoring transparent

**Decision:** Use an interpretable weighted baseline score and label it as a score, not a probability.

**Reason:** A calibrated collision probability requires uncertainty estimation and empirical validation that do not yet exist.

## 2026-07-18 — Disable directional movement guidance by default

**Decision:** Initial policy work will support awareness, slow, stop, and abstain outputs. Directional commands will remain experimental and configuration-gated.

**Reason:** A direction that appears open in a single frame may be unsafe over the prediction horizon.
