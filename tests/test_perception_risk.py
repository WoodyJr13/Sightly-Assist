"""Tests for converting measured 3D motion into collision risk."""

from __future__ import annotations

import pytest

from sightly_assist.depth_association import CameraPoint
from sightly_assist.motion_estimation import (
    MotionEstimate,
    MotionStatus,
    Velocity3D,
)
from sightly_assist.perception import BoundingBox, TrackObservation
from sightly_assist.perception_risk import (
    PerceptionRiskConfig,
    PerceptionRiskStatus,
    assess_perception_risks,
)
from sightly_assist.schemas import RiskLevel


def _track(class_name: str = "person") -> TrackObservation:
    return TrackObservation(
        track_id=3,
        frame_id=10,
        class_name=class_name,
        confidence=0.95,
        bbox=BoundingBox(x_min=100, y_min=80, x_max=180, y_max=220),
        age_frames=5,
        missed_frames=0,
    )


def _motion(
    *,
    x_m: float,
    z_m: float,
    vx_mps: float,
    vz_mps: float,
    residual_rms_m: float = 0.02,
    status: MotionStatus = MotionStatus.VALID,
) -> MotionEstimate:
    return MotionEstimate(
        track_id=3,
        frame_id=10,
        timestamp_ns=1_000_000_000,
        status=status,
        sample_count=5,
        inlier_count=5,
        time_span_s=0.4,
        position=CameraPoint(x_m=x_m, y_m=0.0, z_m=z_m) if status is MotionStatus.VALID else None,
        velocity=(
            Velocity3D(x_mps=vx_mps, y_mps=0.0, z_mps=vz_mps)
            if status is MotionStatus.VALID
            else None
        ),
        residual_rms_m=residual_rms_m,
    )


def test_head_on_approach_is_predicted_collision() -> None:
    result = assess_perception_risks(
        (_track(),),
        (_motion(x_m=0.0, z_m=5.0, vx_mps=0.0, vz_mps=-2.0),),
        timestamp_s=1.0,
    )[0]

    assert result.status is PerceptionRiskStatus.VALID
    assert result.assessment is not None
    assert result.assessment.predicted_collision
    assert result.assessment.time_to_closest_approach_s == pytest.approx(2.5)
    assert result.assessment.distance_at_closest_approach_m == pytest.approx(0.0)
    assert result.assessment.risk_level is RiskLevel.CRITICAL


def test_offset_pass_is_not_predicted_collision() -> None:
    result = assess_perception_risks(
        (_track(),),
        (_motion(x_m=2.0, z_m=5.0, vx_mps=0.0, vz_mps=-2.0),),
        timestamp_s=1.0,
    )[0]

    assert result.assessment is not None
    assert not result.assessment.predicted_collision
    assert result.assessment.distance_at_closest_approach_m == pytest.approx(2.0)


def test_receding_object_has_no_future_collision() -> None:
    result = assess_perception_risks(
        (_track(),),
        (_motion(x_m=0.0, z_m=3.0, vx_mps=0.0, vz_mps=1.0),),
        timestamp_s=1.0,
    )[0]

    assert result.assessment is not None
    assert result.assessment.time_to_closest_approach_s == pytest.approx(0.0)
    assert not result.assessment.predicted_collision
    assert result.assessment.closing_speed_mps < 0.0


def test_excessive_motion_fit_error_withholds_risk() -> None:
    config = PerceptionRiskConfig(maximum_motion_residual_rms_m=0.10)
    result = assess_perception_risks(
        (_track(),),
        (
            _motion(
                x_m=0.0,
                z_m=5.0,
                vx_mps=0.0,
                vz_mps=-2.0,
                residual_rms_m=0.50,
            ),
        ),
        timestamp_s=1.0,
        config=config,
    )[0]

    assert result.status is PerceptionRiskStatus.EXCESSIVE_UNCERTAINTY
    assert result.assessment is None


def test_warming_motion_withholds_risk() -> None:
    result = assess_perception_risks(
        (_track(),),
        (
            _motion(
                x_m=0.0,
                z_m=5.0,
                vx_mps=0.0,
                vz_mps=0.0,
                status=MotionStatus.WARMING_UP,
            ),
        ),
        timestamp_s=1.0,
    )[0]

    assert result.status is PerceptionRiskStatus.MOTION_UNAVAILABLE
    assert result.assessment is None


def test_unknown_motion_track_is_rejected() -> None:
    motion = _motion(x_m=0.0, z_m=5.0, vx_mps=0.0, vz_mps=-2.0).model_copy(
        update={"track_id": 99}
    )

    with pytest.raises(ValueError, match="unknown track"):
        assess_perception_risks((_track(),), (motion,), timestamp_s=1.0)
