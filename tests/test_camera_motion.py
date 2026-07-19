from __future__ import annotations

from math import cos, sin

import numpy as np
import pytest

from sightly_assist.camera_motion import (
    CameraOrientation,
    RotationCompensatedMotionEstimator,
    stabilize_camera_point,
)
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


def _orientation_tuple(angle_rad: float) -> tuple[float, float, float, float]:
    return cos(angle_rad / 2.0), 0.0, sin(angle_rad / 2.0), 0.0


def _frame(
    frame_id: int,
    timestamp_ns: int,
    orientation: tuple[float, float, float, float] | None,
) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        width_px=640,
        height_px=480,
        rgb_path=f"rgb/{frame_id:04d}.png",
        depth_path=f"depth/{frame_id:04d}.npy",
        orientation_wxyz=orientation,
    )


def _association(frame_id: int, point: CameraPoint) -> DepthAssociation:
    return DepthAssociation(
        track_id=3,
        frame_id=frame_id,
        status=DepthStatus.VALID,
        roi_sample_count=100,
        valid_sample_count=100,
        inlier_sample_count=100,
        valid_fraction=1.0,
        depth_m=point.z_m,
        median_absolute_deviation_m=0.0,
        pixel_u=320.0,
        pixel_v=240.0,
        camera_point=point,
    )


def _config() -> MotionEstimatorConfig:
    return MotionEstimatorConfig(
        history_size=5,
        minimum_samples=3,
        minimum_time_span_s=0.1,
        velocity_smoothing_alpha=1.0,
        maximum_speed_mps=100.0,
    )


def test_quaternion_rotation_normalizes_input() -> None:
    orientation = CameraOrientation(w=2.0, x=0.0, y=0.0, z=0.0)

    assert orientation.rotation_matrix() == pytest.approx(np.eye(3))


def test_zero_quaternion_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        CameraOrientation(w=0.0, x=0.0, y=0.0, z=0.0)


def test_reference_orientation_stabilizes_camera_point() -> None:
    reference = CameraOrientation.from_wxyz(_orientation_tuple(0.0))
    current = CameraOrientation.from_wxyz(_orientation_tuple(0.25))
    world_point = np.asarray([0.0, 0.0, 5.0])
    camera_point_vector = current.rotation_matrix().T @ world_point
    camera_point = CameraPoint(
        x_m=float(camera_point_vector[0]),
        y_m=float(camera_point_vector[1]),
        z_m=float(camera_point_vector[2]),
    )

    stabilized = stabilize_camera_point(camera_point, current, reference)

    assert stabilized.x_m == pytest.approx(0.0, abs=1e-9)
    assert stabilized.y_m == pytest.approx(0.0, abs=1e-9)
    assert stabilized.z_m == pytest.approx(5.0, abs=1e-9)


def test_rotation_compensation_removes_false_lateral_velocity() -> None:
    raw = TrackMotionEstimator(_config())
    compensated = RotationCompensatedMotionEstimator(_config())
    raw_result = None
    compensated_result = None
    world_point = np.asarray([0.0, 0.0, 5.0])

    for frame_id, angle in enumerate((0.0, 0.1, 0.2)):
        orientation_values = _orientation_tuple(angle)
        orientation = CameraOrientation.from_wxyz(orientation_values)
        camera_vector = orientation.rotation_matrix().T @ world_point
        point = CameraPoint(
            x_m=float(camera_vector[0]),
            y_m=float(camera_vector[1]),
            z_m=float(camera_vector[2]),
        )
        frame = _frame(frame_id, frame_id * 100_000_000, orientation_values)
        association = _association(frame_id, point)
        raw_result = raw.update(frame, (association,))[0]
        compensated_result = compensated.update(frame, (association,))[0]

    assert raw_result is not None
    assert compensated_result is not None
    assert raw_result.status is MotionStatus.VALID
    assert compensated_result.status is MotionStatus.VALID
    assert raw_result.velocity is not None
    assert compensated_result.velocity is not None
    assert abs(raw_result.velocity.x_mps) > 1.0
    assert compensated_result.velocity.speed_mps < 1e-6


def test_rotation_compensation_preserves_relative_forward_translation() -> None:
    estimator = RotationCompensatedMotionEstimator(_config())
    result = None
    orientation = _orientation_tuple(0.0)

    for frame_id, z_m in enumerate((5.0, 4.8, 4.6)):
        frame = _frame(frame_id, frame_id * 100_000_000, orientation)
        result = estimator.update(
            frame,
            (_association(frame_id, CameraPoint(x_m=0.0, y_m=0.0, z_m=z_m)),),
        )[0]

    assert result is not None
    assert result.status is MotionStatus.VALID
    assert result.velocity is not None
    assert result.velocity.x_mps == pytest.approx(0.0, abs=1e-9)
    assert result.velocity.z_mps == pytest.approx(-2.0, abs=1e-9)


def test_required_orientation_withholds_unstabilized_motion() -> None:
    estimator = RotationCompensatedMotionEstimator(
        _config(),
        require_orientation=True,
    )
    result = estimator.update(
        _frame(0, 0, None),
        (_association(0, CameraPoint(x_m=0.0, y_m=0.0, z_m=4.0)),),
    )[0]

    assert result.status is MotionStatus.NO_DEPTH
    assert result.position is None
    assert result.velocity is None
