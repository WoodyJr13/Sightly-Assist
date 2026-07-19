"""End-to-end test from replay frames to a constrained warning action."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.motion_estimation import MotionEstimatorConfig, TrackMotionEstimator
from sightly_assist.perception import BoundingBox, Detection, FramePacket
from sightly_assist.perception_risk import PerceptionRiskConfig
from sightly_assist.replay import ReplayManifest
from sightly_assist.replay_pipeline import process_replay
from sightly_assist.warning_policy import AlertAction, WarningPolicy


class CentralPersonDetector:
    """Return one person centered in the camera path."""

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
                confidence=0.99,
                bbox=BoundingBox(x_min=8, y_min=2, x_max=12, y_max=8),
            ),
        )


def _manifest() -> ReplayManifest:
    return ReplayManifest(
        sequence_id="warning-sequence",
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


def test_replay_pipeline_escalates_from_abstain_to_stop(tmp_path: Path) -> None:
    depth_by_frame = {0: 5.0, 1: 4.8, 2: 4.6}

    def image_loader(frame: FramePacket, root: Path) -> np.ndarray:
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    def depth_loader(frame: FramePacket, root: Path) -> np.ndarray:
        return np.full(
            (frame.height_px, frame.width_px),
            depth_by_frame[frame.frame_id],
            dtype=np.float32,
        )

    results = list(
        process_replay(
            _manifest(),
            tmp_path,
            CentralPersonDetector(),
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
            risk_config=PerceptionRiskConfig(),
            warning_policy=WarningPolicy(),
        )
    )

    assert results[0].warning_decision is not None
    assert results[0].warning_decision.action is AlertAction.ABSTAIN
    assert results[2].warning_decision is not None
    assert results[2].warning_decision.action is AlertAction.STOP
    assert results[2].warning_decision.should_emit


def test_warning_policy_requires_risk_configuration(tmp_path: Path) -> None:
    def image_loader(frame: FramePacket, root: Path) -> np.ndarray:
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="requires risk_config"):
        list(
            process_replay(
                _manifest(),
                tmp_path,
                CentralPersonDetector(),
                IoUTracker(),
                image_loader=image_loader,
                warning_policy=WarningPolicy(),
            )
        )
