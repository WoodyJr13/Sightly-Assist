"""Estimate per-track 3D velocity from temporally filtered depth positions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sightly_assist.depth_association import CameraPoint, DepthAssociation, DepthStatus
from sightly_assist.perception import FramePacket


class MotionStatus(StrEnum):
    """State of one track's temporal motion estimate."""

    WARMING_UP = "warming_up"
    VALID = "valid"
    NO_DEPTH = "no_depth"
    STALE = "stale"
    OUTLIER = "outlier"


class Velocity3D(BaseModel):
    """Velocity in the right/down/forward camera coordinate frame."""

    x_mps: float
    y_mps: float
    z_mps: float

    @property
    def speed_mps(self) -> float:
        return sqrt(self.x_mps**2 + self.y_mps**2 + self.z_mps**2)


class MotionEstimatorConfig(BaseModel):
    """Controls temporal history, smoothing, and outlier rejection."""

    history_size: int = Field(ge=2, default=8)
    minimum_samples: int = Field(ge=2, default=3)
    minimum_time_span_s: float = Field(gt=0, default=0.10)
    maximum_history_age_s: float = Field(gt=0, default=1.25)
    maximum_sample_gap_s: float = Field(gt=0, default=0.50)
    stale_after_s: float = Field(gt=0, default=0.75)
    velocity_smoothing_alpha: float = Field(gt=0, le=1, default=0.45)
    maximum_speed_mps: float = Field(gt=0, default=20.0)
    residual_mad_multiplier: float = Field(gt=0, default=3.5)
    minimum_residual_band_m: float = Field(gt=0, default=0.04)

    @model_validator(mode="after")
    def validate_windows(self) -> MotionEstimatorConfig:
        if self.minimum_samples > self.history_size:
            raise ValueError("minimum_samples cannot exceed history_size")
        if self.stale_after_s < self.maximum_sample_gap_s:
            raise ValueError("stale_after_s must be at least maximum_sample_gap_s")
        return self


class MotionEstimate(BaseModel):
    """Position, velocity, and fit quality for one tracked object."""

    track_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    status: MotionStatus
    sample_count: int = Field(ge=0)
    inlier_count: int = Field(ge=0)
    time_span_s: float = Field(ge=0)
    position: CameraPoint | None = None
    velocity: Velocity3D | None = None
    residual_rms_m: float | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class _PositionSample:
    timestamp_ns: int
    frame_id: int
    point: CameraPoint


class TrackMotionEstimator:
    """Maintain robust temporal 3D velocity estimates for active tracks."""

    def __init__(self, config: MotionEstimatorConfig | None = None) -> None:
        self.config = config or MotionEstimatorConfig()
        self._history: dict[int, deque[_PositionSample]] = {}
        self._smoothed_velocity: dict[int, np.ndarray] = {}

    def update(
        self,
        frame: FramePacket,
        associations: tuple[DepthAssociation, ...],
    ) -> tuple[MotionEstimate, ...]:
        """Update all depth-associated tracks for one chronological frame."""

        estimates: list[MotionEstimate] = []
        current_track_ids = {association.track_id for association in associations}
        self._drop_long_absent_tracks(frame.timestamp_ns, current_track_ids)

        for association in associations:
            if association.frame_id != frame.frame_id:
                raise ValueError(
                    "Depth association belongs to the wrong frame: "
                    f"expected {frame.frame_id}, got {association.frame_id}"
                )
            estimates.append(self._update_track(frame, association))

        return tuple(sorted(estimates, key=lambda item: item.track_id))

    def _update_track(
        self,
        frame: FramePacket,
        association: DepthAssociation,
    ) -> MotionEstimate:
        history = self._history.setdefault(
            association.track_id,
            deque(maxlen=self.config.history_size),
        )

        if association.status is not DepthStatus.VALID or association.camera_point is None:
            if not history:
                return self._empty_estimate(frame, association.track_id, MotionStatus.NO_DEPTH)
            age_s = (frame.timestamp_ns - history[-1].timestamp_ns) / 1_000_000_000
            if age_s < 0:
                raise ValueError("Replay timestamps must be nondecreasing")
            status = (
                MotionStatus.STALE if age_s >= self.config.stale_after_s else MotionStatus.NO_DEPTH
            )
            if status is MotionStatus.STALE:
                history.clear()
                self._smoothed_velocity.pop(association.track_id, None)
            return MotionEstimate(
                track_id=association.track_id,
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
                status=status,
                sample_count=len(history),
                inlier_count=0,
                time_span_s=0.0,
            )

        if history:
            delta_s = (frame.timestamp_ns - history[-1].timestamp_ns) / 1_000_000_000
            if delta_s <= 0:
                raise ValueError("Track samples must have strictly increasing timestamps")
            if delta_s > self.config.maximum_sample_gap_s:
                history.clear()
                self._smoothed_velocity.pop(association.track_id, None)

        history.append(
            _PositionSample(
                timestamp_ns=frame.timestamp_ns,
                frame_id=frame.frame_id,
                point=association.camera_point,
            )
        )
        self._prune_history(history, frame.timestamp_ns)

        span_s = self._history_span_s(history)
        if len(history) < self.config.minimum_samples or span_s < self.config.minimum_time_span_s:
            return MotionEstimate(
                track_id=association.track_id,
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
                status=MotionStatus.WARMING_UP,
                sample_count=len(history),
                inlier_count=len(history),
                time_span_s=span_s,
                position=association.camera_point,
            )

        raw_velocity, residual_rms_m, inlier_count = self._robust_velocity(tuple(history))
        raw_speed = float(np.linalg.norm(raw_velocity))
        if raw_speed > self.config.maximum_speed_mps:
            history.pop()
            return MotionEstimate(
                track_id=association.track_id,
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
                status=MotionStatus.OUTLIER,
                sample_count=len(history),
                inlier_count=inlier_count,
                time_span_s=span_s,
                position=association.camera_point,
                residual_rms_m=residual_rms_m,
            )

        previous = self._smoothed_velocity.get(association.track_id)
        if previous is None:
            filtered = raw_velocity
        else:
            alpha = self.config.velocity_smoothing_alpha
            filtered = alpha * raw_velocity + (1.0 - alpha) * previous
        self._smoothed_velocity[association.track_id] = filtered

        return MotionEstimate(
            track_id=association.track_id,
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            status=MotionStatus.VALID,
            sample_count=len(history),
            inlier_count=inlier_count,
            time_span_s=span_s,
            position=association.camera_point,
            velocity=Velocity3D(
                x_mps=float(filtered[0]),
                y_mps=float(filtered[1]),
                z_mps=float(filtered[2]),
            ),
            residual_rms_m=residual_rms_m,
        )

    def _robust_velocity(
        self,
        samples: tuple[_PositionSample, ...],
    ) -> tuple[np.ndarray, float, int]:
        times = np.asarray(
            [(sample.timestamp_ns - samples[0].timestamp_ns) / 1_000_000_000 for sample in samples],
            dtype=np.float64,
        )
        positions = np.asarray(
            [[sample.point.x_m, sample.point.y_m, sample.point.z_m] for sample in samples],
            dtype=np.float64,
        )

        slopes: list[np.ndarray] = []
        for left in range(len(samples) - 1):
            for right in range(left + 1, len(samples)):
                delta_t = times[right] - times[left]
                if delta_t > 0:
                    slopes.append((positions[right] - positions[left]) / delta_t)
        velocity = np.asarray(np.median(np.stack(slopes), axis=0), dtype=np.float64)
        intercept = np.median(positions - times[:, np.newaxis] * velocity, axis=0)
        predictions = intercept + times[:, np.newaxis] * velocity
        residuals = np.linalg.norm(positions - predictions, axis=1)
        median_residual = float(np.median(residuals))
        residual_mad = float(np.median(np.abs(residuals - median_residual)))
        band = max(
            self.config.minimum_residual_band_m,
            self.config.residual_mad_multiplier * 1.4826 * residual_mad,
        )
        inlier_mask = residuals <= median_residual + band
        if int(np.count_nonzero(inlier_mask)) >= self.config.minimum_samples:
            inlier_times = times[inlier_mask]
            inlier_positions = positions[inlier_mask]
            velocity = self._least_squares_velocity(inlier_times, inlier_positions)
            centered_times = inlier_times - float(np.mean(inlier_times))
            center_position = np.mean(inlier_positions, axis=0)
            fitted = center_position + centered_times[:, np.newaxis] * velocity
            fit_residuals = np.linalg.norm(inlier_positions - fitted, axis=1)
            residual_rms = float(np.sqrt(np.mean(fit_residuals**2)))
            return velocity, residual_rms, int(inlier_positions.shape[0])

        residual_rms = float(np.sqrt(np.mean(residuals**2)))
        return velocity, residual_rms, len(samples)

    @staticmethod
    def _least_squares_velocity(times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        centered_times = times - float(np.mean(times))
        denominator = float(np.dot(centered_times, centered_times))
        if denominator <= 0:
            return np.zeros(3, dtype=np.float64)
        centered_positions = positions - np.mean(positions, axis=0)
        numerator = np.sum(
            centered_times[:, np.newaxis] * centered_positions,
            axis=0,
        )
        return np.asarray(numerator / denominator, dtype=np.float64)

    def _prune_history(self, history: deque[_PositionSample], timestamp_ns: int) -> None:
        maximum_age_ns = int(self.config.maximum_history_age_s * 1_000_000_000)
        while history and timestamp_ns - history[0].timestamp_ns > maximum_age_ns:
            history.popleft()

    def _drop_long_absent_tracks(self, timestamp_ns: int, current_track_ids: set[int]) -> None:
        stale_ns = int(self.config.stale_after_s * 1_000_000_000)
        expired = [
            track_id
            for track_id, history in self._history.items()
            if track_id not in current_track_ids
            and history
            and timestamp_ns - history[-1].timestamp_ns >= stale_ns
        ]
        for track_id in expired:
            del self._history[track_id]
            self._smoothed_velocity.pop(track_id, None)

    @staticmethod
    def _history_span_s(history: deque[_PositionSample]) -> float:
        if len(history) < 2:
            return 0.0
        return (history[-1].timestamp_ns - history[0].timestamp_ns) / 1_000_000_000

    @staticmethod
    def _empty_estimate(
        frame: FramePacket,
        track_id: int,
        status: MotionStatus,
    ) -> MotionEstimate:
        return MotionEstimate(
            track_id=track_id,
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            status=status,
            sample_count=0,
            inlier_count=0,
            time_span_s=0.0,
        )
