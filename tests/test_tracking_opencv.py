"""Tests for baseline tracking and optional OpenCV replay loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.opencv_io import load_depth, load_rgb
from sightly_assist.perception import BoundingBox, Detection, FramePacket


def _frame(frame_id: int, rgb_path: str = "frame.png") -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000,
        width_px=20,
        height_px=10,
        rgb_path=rgb_path,
    )


def _detection(frame_id: int, x_offset: float = 0.0) -> Detection:
    return Detection(
        frame_id=frame_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(
            x_min=2.0 + x_offset,
            y_min=1.0,
            x_max=10.0 + x_offset,
            y_max=9.0,
        ),
    )


def test_iou_tracker_preserves_identity_for_overlapping_boxes() -> None:
    tracker = IoUTracker(minimum_iou=0.3)

    first = tracker.update(_frame(0), [_detection(0)])
    second = tracker.update(_frame(1), [_detection(1, x_offset=1.0)])

    assert first[0].track_id == second[0].track_id
    assert second[0].age_frames == 2


def test_iou_tracker_expires_missing_track() -> None:
    tracker = IoUTracker(max_missed_frames=0)

    first = tracker.update(_frame(0), [_detection(0)])
    assert first[0].track_id == 0

    assert tracker.update(_frame(1), []) == ()
    replacement = tracker.update(_frame(2), [_detection(2)])
    assert replacement[0].track_id == 1


def test_load_rgb_validates_manifest_dimensions(tmp_path: Path) -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), image)

    loaded = load_rgb(_frame(0), tmp_path)
    assert loaded.shape == (10, 20, 3)

    wrong = _frame(0).model_copy(update={"width_px": 21})
    with pytest.raises(ValueError, match="dimensions do not match"):
        load_rgb(wrong, tmp_path)


def test_load_depth_validates_shape(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.ones((10, 20), dtype=np.float32))
    frame = _frame(0).model_copy(update={"depth_path": "depth.npy"})

    loaded = load_depth(frame, tmp_path)
    assert loaded is not None
    assert loaded.shape == (10, 20)

    np.save(depth_path, np.ones((5, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="Depth dimensions do not match"):
        load_depth(frame, tmp_path)
