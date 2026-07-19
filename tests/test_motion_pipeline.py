"""Integration tests for replay depth and temporal motion estimation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.motion_estimation import (
    MotionEstimatorConfig,
    MotionStatus,
    TrackMotionEstimator,
)
from sightly_assist.perception import BoundingBox, Detection, FramePacket
from sightly_assist.replay import ReplayManifest
from sightly_assist.replay_pipeline import process_replay


class PersonDetector:
    """Return one stable person box for each decoded image."""

    def predict(
        self,
        frame: FramePacket,
        image_bgr: np.ndarray,
    ) -> Sequence[Detection]:
        assert image_bgr.shape == (10, 20, 3)
        return (
            Detection(
                frame_id=frame.frame_id,
                class_id=0,
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(x_min=4, y_min=2, x_max=16, y_max=8),
            ),
        )


def _manifest() -> ReplayManifest:
    return ReplayManifest(
        sequence_id="motion-sequence",
        source="unit-test",
        frames=[
            FramePacket(
                frame_id=frame_id,
                timestamp_ns=frame_id * 100_000_000,
                width_px=20,
                height_px=10,
                rgb_path=f"rgb/{frame_id}.png",
                depth_path=f"depth/{frame_id}.npy",
            )
            for frame_id in range(3)
        ],
    )


def test_replay_pipeline_outputs_temporal_3d_velocity(tmp_path: Path) -> None:
    depths_m = {0: 5.0, 1: 4.8, 2: 4.6}

    def image_loader(frame: FramePacket, root: Path) -> np.ndarray:
        assert root == tmp_path
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    def depth_loader(frame: FramePacket, root: Path) -> np.ndarray:
        assert root == tmp_path
        return np.full(
            (frame.height_px, frame.width_px),
            depths_m[frame.frame_id],
            dtype=np.float32,
        )

    intrinsics = CameraIntrinsics(
        width_px=20,
        height_px=10,
        fx_px=10,
        fy_px=10,
        cx_px=10,
        cy_px=5,
    )
    estimator = TrackMotionEstimator(
        MotionEstimatorConfig(
            minimum_samples=3,
            minimum_time_span_s=0.1,
            velocity_smoothing_alpha=1.0,
        )
    )

    results = list(
        process_replay(
            _manifest(),
            tmp_path,
            PersonDetector(),
            IoUTracker(),
            image_loader=image_loader,
            depth_loader=depth_loader,
            camera_intrinsics=intrinsics,
            motion_estimator=estimator,
        )
    )

    assert results[0].motion_estimates[0].status is MotionStatus.WARMING_UP
    assert results[1].motion_estimates[0].status is MotionStatus.WARMING_UP
    final = results[2].motion_estimates[0]
    assert final.status is MotionStatus.VALID
    assert final.velocity is not None
    assert final.velocity.z_mps == pytest.approx(-2.0, rel=1e-5)
    assert final.velocity.x_mps == pytest.approx(0.0, abs=1e-8)
    assert final.track_id == results[2].tracks[0].track_id


def test_motion_pipeline_requires_depth_configuration(tmp_path: Path) -> None:
    def image_loader(frame: FramePacket, root: Path) -> np.ndarray:
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="requires depth_loader"):
        list(
            process_replay(
                _manifest(),
                tmp_path,
                PersonDetector(),
                IoUTracker(),
                image_loader=image_loader,
                motion_estimator=TrackMotionEstimator(),
            )
        )
