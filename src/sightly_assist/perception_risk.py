"""Convert tracked 3D relative motion into closest-approach risk results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from sightly_assist.motion_estimation import MotionEstimate, MotionStatus
from sightly_assist.perception import TrackObservation
from sightly_assist.risk import assess_relative_risk
from sightly_assist.schemas import RiskAssessment, Vector2


class PerceptionRiskStatus(StrEnum):
    """Whether a track has enough trustworthy motion for risk assessment."""

    VALID = "valid"
    MOTION_UNAVAILABLE = "motion_unavailable"
    EXCESSIVE_UNCERTAINTY = "excessive_uncertainty"


class PerceptionRiskConfig(BaseModel):
    """Safety geometry and quality thresholds for measured tracks."""

    prediction_horizon_s: float = Field(gt=0, default=5.0)
    observer_radius_m: float = Field(gt=0, default=0.30)
    safety_margin_m: float = Field(ge=0, default=0.35)
    default_obstacle_radius_m: float = Field(gt=0, default=0.40)
    maximum_motion_residual_rms_m: float = Field(gt=0, default=0.35)
    class_radius_m: dict[str, float] = Field(
        default_factory=lambda: {
            "person": 0.35,
            "bicycle": 0.70,
            "car": 1.10,
            "motorcycle": 0.75,
            "bus": 1.30,
            "truck": 1.30,
            "dog": 0.35,
            "chair": 0.40,
        }
    )


class PerceptionRiskResult(BaseModel):
    """Risk output or explicit reason that a track was not assessed."""

    track_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    status: PerceptionRiskStatus
    motion_status: MotionStatus
    assessment: RiskAssessment | None = None


def assess_perception_risks(
    tracks: tuple[TrackObservation, ...],
    motions: tuple[MotionEstimate, ...],
    timestamp_s: float,
    config: PerceptionRiskConfig | None = None,
) -> tuple[PerceptionRiskResult, ...]:
    """Assess ground-plane CPA risk for every temporally estimated track."""

    settings = config or PerceptionRiskConfig()
    tracks_by_id = {track.track_id: track for track in tracks}
    if len(tracks_by_id) != len(tracks):
        raise ValueError("Track IDs must be unique within a frame")

    results: list[PerceptionRiskResult] = []
    for motion in motions:
        track = tracks_by_id.get(motion.track_id)
        if track is None:
            raise ValueError(f"Motion estimate references unknown track ID {motion.track_id}")

        if (
            motion.status is not MotionStatus.VALID
            or motion.position is None
            or motion.velocity is None
        ):
            results.append(
                PerceptionRiskResult(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    status=PerceptionRiskStatus.MOTION_UNAVAILABLE,
                    motion_status=motion.status,
                )
            )
            continue

        if (
            motion.residual_rms_m is not None
            and motion.residual_rms_m > settings.maximum_motion_residual_rms_m
        ):
            results.append(
                PerceptionRiskResult(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    status=PerceptionRiskStatus.EXCESSIVE_UNCERTAINTY,
                    motion_status=motion.status,
                )
            )
            continue

        obstacle_radius = settings.class_radius_m.get(
            track.class_name,
            settings.default_obstacle_radius_m,
        )
        boundary = settings.observer_radius_m + obstacle_radius + settings.safety_margin_m
        assessment = assess_relative_risk(
            obstacle_id=f"track-{track.track_id}",
            timestamp_s=timestamp_s,
            relative_position=Vector2(x=motion.position.x_m, z=motion.position.z_m),
            relative_velocity=Vector2(
                x=motion.velocity.x_mps,
                z=motion.velocity.z_mps,
            ),
            prediction_horizon_s=settings.prediction_horizon_s,
            collision_boundary_m=boundary,
        )
        results.append(
            PerceptionRiskResult(
                track_id=track.track_id,
                class_name=track.class_name,
                status=PerceptionRiskStatus.VALID,
                motion_status=motion.status,
                assessment=assessment,
            )
        )

    return tuple(sorted(results, key=lambda item: item.track_id))
