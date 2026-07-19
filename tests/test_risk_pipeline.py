"""End-to-end replay test through measured motion and collision risk."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.motion_estimation import MotionEstimatorConfig, TrackMotionEstimator
from sightly_assist.perception import BoundingBox, Detection, FramePacket
from sightly_assist.perception_risk import PerceptionRiskConfig, PerceptionRiskStatus
from sightly_assist.replay import ReplayManifest
from sightly_assist.replay_pipeline import process_replay


class CenterPersonDetector:
    """Return one central person detection for each frame."""

    def predict(
        self,
        frame: FramePacket,
        image_bgr: np.ndarray,
    ) -> Sequence[Detection]:
        return (
            Detection(
                frame_id=frame.frame_id,
                class_id=0,
                class_name="person",
                confidence=0.98,
                bbox=BoundingBox(x_min=8, y_min=2, x_max=12, y_max=8),
            ),
        )


def _manifest() -> ReplayManifest:
    return ReplayManifest(
        sequence_id="risk-sequence",
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


def test_replay_pipeline_predicts_head_on_collision(tmp_path: Path) -> None:
    depths_m = {0: 5.0, 1: 4.8, 2: 4.6}

    def image_loader(frame: FramePacket, root: Path) -> np.ndarray:
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    def depth_loader(frame: FramePacket, root: Path) -> np.ndarray:
        return np.full(
            (frame.height_px, frame.width_px),
            depths_m[frame.frame_id],
            dtype=np.float32,
        )

    results = list(
        process_replay(
            _manifest(),
            tmp_path,
            CenterPersonDetector(),
            IoUTracker(),
            image_loader=image_loader,
            depth_loader=depth_loader,
            camera_intrinsics=CameraIntrinsics(
                width_px=20,
                height_px=10,
                fx_px=10,
                fy_px=10,
                cx_px=10,
                cy_px=5,
            ),
            motion_estimator=TrackMotionEstimator(
                MotionEstimatorConfig(
                    minimum_samples=3,
                    minimum_time_span_s=0.1,
                    velocity_smoothing_alpha=1.0,
                )
            ),
            risk_config=PerceptionRiskConfig(prediction_horizon_s=5.0),
        )
    )

    assert results[0].risk_results[0].status is PerceptionRiskStatus.MOTION_UNAVAILABLE
    assert results[1].risk_results[0].status is PerceptionRiskStatus.MOTION_UNAVAILABLE
    final = results[2].risk_results[0]
    assert final.status is PerceptionRiskStatus.VALID
    assert final.assessment is not None
    assert final.assessment.predicted_collision
    assert final.assessment.obstacle_id == "track-0"


def test_risk_configuration_requires_motion_estimator(tmp_path: Path) -> None:
    def image_loader(frame: FramePacket, root: Path) -> np.ndarray:
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    try:
        list(
            process_replay(
                _manifest(),
                tmp_path,
                CenterPersonDetector(),
                IoUTracker(),
                image_loader=image_loader,
                risk_config=PerceptionRiskConfig(),
            )
        )
    except ValueError as exc:
        assert "requires motion_estimator" in str(exc)
    else:
        raise AssertionError("risk_config without motion_estimator should fail")
