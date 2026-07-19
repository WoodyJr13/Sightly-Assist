"""Regression tests for robust depth association and deprojection."""

from __future__ import annotations

import numpy as np
import pytest

from sightly_assist.depth_association import (
    CameraIntrinsics,
    DepthAssociationConfig,
    DepthStatus,
    associate_track_depth,
    associate_tracks_depth,
    deproject_pixel,
)
from sightly_assist.perception import BoundingBox, FramePacket, TrackObservation


def _frame() -> FramePacket:
    return FramePacket(
        frame_id=4,
        timestamp_ns=400,
        width_px=8,
        height_px=6,
        rgb_path="rgb/000004.png",
        depth_path="depth/000004.npy",
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=8,
        height_px=6,
        fx_px=4.0,
        fy_px=4.0,
        cx_px=4.0,
        cy_px=3.0,
    )


def _track(track_id: int = 2) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        frame_id=4,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(x_min=2, y_min=1, x_max=6, y_max=5),
        age_frames=3,
        missed_frames=0,
    )


def test_depth_association_scales_millimeters_and_rejects_outlier() -> None:
    depth = np.full((6, 8), 2000, dtype=np.uint16)
    depth[2, 3] = 0
    depth[2, 4] = 10000
    result = associate_track_depth(
        depth,
        _frame(),
        _track(),
        _intrinsics(),
        DepthAssociationConfig(
            inner_box_fraction=1.0,
            minimum_valid_samples=4,
            depth_scale_m=0.001,
            maximum_depth_m=15.0,
        ),
    )

    assert result.status is DepthStatus.VALID
    assert result.depth_m == pytest.approx(2.0)
    assert result.valid_sample_count == 15
    assert result.inlier_sample_count == 14
    assert result.camera_point is not None
    assert result.camera_point.x_m == pytest.approx(0.0)
    assert result.camera_point.y_m == pytest.approx(0.0)
    assert result.camera_point.z_m == pytest.approx(2.0)


def test_depth_association_reports_insufficient_samples() -> None:
    depth = np.zeros((6, 8), dtype=np.float32)
    depth[2, 3] = 1.5
    result = associate_track_depth(
        depth,
        _frame(),
        _track(),
        _intrinsics(),
        DepthAssociationConfig(inner_box_fraction=1.0, minimum_valid_samples=2),
    )

    assert result.status is DepthStatus.INSUFFICIENT_SAMPLES
    assert result.depth_m is None
    assert result.valid_sample_count == 1


def test_depth_association_handles_absent_depth() -> None:
    result = associate_track_depth(None, _frame(), _track(), _intrinsics())

    assert result.status is DepthStatus.NO_DEPTH
    assert result.camera_point is None
    assert result.valid_fraction == 0.0


def test_deprojection_uses_pinhole_camera_coordinates() -> None:
    point = deproject_pixel(6.0, 1.0, 2.0, _intrinsics())

    assert point.x_m == pytest.approx(1.0)
    assert point.y_m == pytest.approx(-1.0)
    assert point.z_m == pytest.approx(2.0)


def test_batch_association_preserves_track_order() -> None:
    depth = np.full((6, 8), 3.0, dtype=np.float32)
    tracks = (_track(track_id=8), _track(track_id=3))

    results = associate_tracks_depth(depth, _frame(), tracks, _intrinsics())

    assert [result.track_id for result in results] == [8, 3]
    assert all(result.status is DepthStatus.VALID for result in results)


def test_depth_map_and_intrinsics_must_match_frame() -> None:
    with pytest.raises(ValueError, match="intrinsics dimensions"):
        associate_track_depth(
            np.full((6, 8), 2.0),
            _frame(),
            _track(),
            CameraIntrinsics(
                width_px=9,
                height_px=6,
                fx_px=4,
                fy_px=4,
                cx_px=4,
                cy_px=3,
            ),
        )

    with pytest.raises(ValueError, match="depth_map dimensions"):
        associate_track_depth(
            np.full((5, 8), 2.0),
            _frame(),
            _track(),
            _intrinsics(),
        )
