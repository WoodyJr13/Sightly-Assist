"""Tests for the replay perception and depth processing loop."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sightly_assist.depth_association import CameraIntrinsics, DepthStatus
from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.perception import BoundingBox, Detection, FramePacket
from sightly_assist.replay import ReplayManifest
from sightly_assist.replay_pipeline import process_replay


class SequenceDetector:
    """Return one deterministic person detection per frame."""

    def predict(self, frame: FramePacket) -> Sequence[Detection]:
        return (
            Detection(
                frame_id=frame.frame_id,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x_min=2, y_min=1, x_max=10, y_max=9),
            ),
        )


class WrongFrameDetector:
    """Return an invalid detection for regression testing."""

    def predict(self, frame: FramePacket) -> Sequence[Detection]:
        return (
            Detection(
                frame_id=frame.frame_id + 1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x_min=2, y_min=1, x_max=10, y_max=9),
            ),
        )


def _manifest() -> ReplayManifest:
    return ReplayManifest(
        sequence_id="test-sequence",
        source="unit-test",
        frames=[
            FramePacket(
                frame_id=0,
                timestamp_ns=0,
                width_px=20,
                height_px=10,
                rgb_path="rgb/0.png",
                depth_path="depth/0.npy",
            ),
            FramePacket(
                frame_id=1,
                timestamp_ns=1_000_000,
                width_px=20,
                height_px=10,
                rgb_path="rgb/1.png",
                depth_path="depth/1.npy",
            ),
        ],
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=20,
        height_px=10,
        fx_px=10,
        fy_px=10,
        cx_px=10,
        cy_px=5,
    )


def test_process_replay_preserves_track_identity(tmp_path: Path) -> None:
    loaded_frames: list[int] = []

    def fake_loader(frame: FramePacket, root: Path) -> np.ndarray:
        assert root == tmp_path
        loaded_frames.append(frame.frame_id)
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    results = list(
        process_replay(
            _manifest(),
            tmp_path,
            SequenceDetector(),
            IoUTracker(),
            image_loader=fake_loader,
        )
    )

    assert loaded_frames == [0, 1]
    assert [result.frame.frame_id for result in results] == [0, 1]
    assert results[0].tracks[0].track_id == results[1].tracks[0].track_id
    assert results[0].depth_associations == ()


def test_process_replay_associates_depth_with_tracks(tmp_path: Path) -> None:
    loaded_depth_frames: list[int] = []

    def fake_depth_loader(frame: FramePacket, root: Path) -> np.ndarray:
        assert root == tmp_path
        loaded_depth_frames.append(frame.frame_id)
        return np.full((frame.height_px, frame.width_px), 2.5, dtype=np.float32)

    results = list(
        process_replay(
            _manifest(),
            tmp_path,
            SequenceDetector(),
            IoUTracker(),
            depth_loader=fake_depth_loader,
            camera_intrinsics=_intrinsics(),
        )
    )

    assert loaded_depth_frames == [0, 1]
    assert len(results[0].depth_associations) == 1
    association = results[0].depth_associations[0]
    assert association.track_id == results[0].tracks[0].track_id
    assert association.status is DepthStatus.VALID
    assert association.depth_m == pytest.approx(2.5)
    assert association.camera_point is not None


def test_process_replay_requires_complete_depth_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both be provided"):
        list(
            process_replay(
                _manifest(),
                tmp_path,
                SequenceDetector(),
                IoUTracker(),
                depth_loader=lambda frame, root: None,
            )
        )


def test_process_replay_rejects_wrong_frame_detection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wrong frame"):
        list(
            process_replay(
                _manifest(),
                tmp_path,
                WrongFrameDetector(),
                IoUTracker(),
            )
        )
