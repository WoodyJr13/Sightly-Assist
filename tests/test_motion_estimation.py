"""Tests for robust per-track temporal 3D motion estimation."""

from __future__ import annotations

import pytest

from sightly_assist.depth_association import (
    CameraPoint,
    DepthAssociation,
    DepthStatus,
)
from sightly_assist.motion_estimation import (
    MotionEstimatorConfig,
    MotionStatus,
    TrackMotionEstimator,
)
from sightly_assist.perception import FramePacket


def _frame(frame_id: int, timestamp_s: float) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=int(timestamp_s * 1_000_000_000),
        width_px=640,
        height_px=480,
        rgb_path=f"rgb/{frame_id:06d}.png",
        depth_path=f"depth/{frame_id:06d}.npy",
    )


def _association(frame_id: int, z_m: float, x_m: float = 0.0) -> DepthAssociation:
    return DepthAssociation(
        track_id=4,
        frame_id=frame_id,
        status=DepthStatus.VALID,
        roi_sample_count=100,
        valid_sample_count=90,
        inlier_sample_count=85,
        valid_fraction=0.9,
        depth_m=z_m,
        median_absolute_deviation_m=0.01,
        pixel_u=320,
        pixel_v=240,
        camera_point=CameraPoint(x_m=x_m, y_m=0.0, z_m=z_m),
    )


def _missing_association(frame_id: int) -> DepthAssociation:
    return DepthAssociation(
        track_id=4,
        frame_id=frame_id,
        status=DepthStatus.NO_DEPTH,
        roi_sample_count=0,
        valid_sample_count=0,
        inlier_sample_count=0,
        valid_fraction=0.0,
    )


def _estimator(**overrides: float | int) -> TrackMotionEstimator:
    values: dict[str, float | int] = {
        "history_size": 8,
        "minimum_samples": 3,
        "minimum_time_span_s": 0.1,
        "maximum_history_age_s": 2.0,
        "maximum_sample_gap_s": 0.5,
        "stale_after_s": 0.75,
        "velocity_smoothing_alpha": 1.0,
        "maximum_speed_mps": 20.0,
    }
    values.update(overrides)
    return TrackMotionEstimator(MotionEstimatorConfig.model_validate(values))


def test_estimates_constant_forward_velocity() -> None:
    estimator = _estimator()

    first = estimator.update(_frame(0, 0.0), (_association(0, 5.0),))[0]
    second = estimator.update(_frame(1, 0.1), (_association(1, 4.8),))[0]
    third = estimator.update(_frame(2, 0.2), (_association(2, 4.6),))[0]

    assert first.status is MotionStatus.WARMING_UP
    assert second.status is MotionStatus.WARMING_UP
    assert third.status is MotionStatus.VALID
    assert third.velocity is not None
    assert third.velocity.x_mps == pytest.approx(0.0, abs=1e-8)
    assert third.velocity.z_mps == pytest.approx(-2.0, rel=1e-6)
    assert third.velocity.speed_mps == pytest.approx(2.0, rel=1e-6)
    assert third.residual_rms_m == pytest.approx(0.0, abs=1e-8)


def test_handles_irregular_frame_intervals() -> None:
    estimator = _estimator()
    samples = ((0, 0.0, 1.0), (1, 0.15, 1.3), (2, 0.40, 1.8))

    result = None
    for frame_id, timestamp_s, x_m in samples:
        result = estimator.update(
            _frame(frame_id, timestamp_s),
            (_association(frame_id, z_m=3.0, x_m=x_m),),
        )[0]

    assert result is not None
    assert result.status is MotionStatus.VALID
    assert result.velocity is not None
    assert result.velocity.x_mps == pytest.approx(2.0, rel=1e-6)
    assert result.velocity.z_mps == pytest.approx(0.0, abs=1e-8)


def test_rejects_one_extreme_position_outlier() -> None:
    estimator = _estimator(minimum_samples=4, maximum_speed_mps=30.0)
    positions = (5.0, 4.9, 12.0, 4.7, 4.6)

    result = None
    for frame_id, z_m in enumerate(positions):
        result = estimator.update(
            _frame(frame_id, frame_id * 0.1),
            (_association(frame_id, z_m),),
        )[0]

    assert result is not None
    assert result.status is MotionStatus.VALID
    assert result.velocity is not None
    assert result.velocity.z_mps == pytest.approx(-1.0, abs=0.15)
    assert result.inlier_count < result.sample_count


def test_marks_missing_depth_stale_and_restarts_history() -> None:
    estimator = _estimator()
    estimator.update(_frame(0, 0.0), (_association(0, 5.0),))
    estimator.update(_frame(1, 0.1), (_association(1, 4.9),))

    missing = estimator.update(_frame(2, 0.2), (_missing_association(2),))[0]
    stale = estimator.update(_frame(3, 1.0), (_missing_association(3),))[0]
    restarted = estimator.update(_frame(4, 1.1), (_association(4, 4.0),))[0]

    assert missing.status is MotionStatus.NO_DEPTH
    assert stale.status is MotionStatus.STALE
    assert restarted.status is MotionStatus.WARMING_UP
    assert restarted.sample_count == 1


def test_rejects_nonincreasing_track_timestamps() -> None:
    estimator = _estimator()
    estimator.update(_frame(0, 0.1), (_association(0, 5.0),))

    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.update(_frame(1, 0.1), (_association(1, 4.9),))


def test_rejects_depth_association_from_wrong_frame() -> None:
    estimator = _estimator()

    with pytest.raises(ValueError, match="wrong frame"):
        estimator.update(_frame(0, 0.0), (_association(1, 5.0),))
