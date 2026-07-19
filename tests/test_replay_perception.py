"""Tests for hardware-independent replay and perception contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from sightly_assist.perception import BoundingBox, EmptyDetector, FramePacket, ObjectDetector
from sightly_assist.replay import ReplayManifest, iter_frames, load_manifest, validate_replay_files


def test_bounding_box_requires_positive_area() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x_min=4, y_min=2, x_max=4, y_max=8)


def test_empty_detector_satisfies_protocol() -> None:
    detector = EmptyDetector()
    frame = FramePacket(
        frame_id=0,
        timestamp_ns=1,
        width_px=640,
        height_px=480,
        rgb_path="rgb/000000.jpg",
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    assert isinstance(detector, ObjectDetector)
    assert detector.predict(frame, image) == ()


def test_manifest_round_trip_and_file_validation(tmp_path: Path) -> None:
    rgb_directory = tmp_path / "rgb"
    rgb_directory.mkdir()
    (rgb_directory / "000000.jpg").write_bytes(b"test")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sequence_id": "sequence-1",
                "source": "test",
                "frames": [
                    {
                        "frame_id": 0,
                        "timestamp_ns": 100,
                        "width_px": 640,
                        "height_px": 480,
                        "rgb_path": "rgb/000000.jpg",
                        "depth_path": "depth/000000.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert [frame.frame_id for frame in iter_frames(manifest)] == [0]
    assert validate_replay_files(manifest, tmp_path) == ["depth/000000.npy"]


def test_manifest_rejects_out_of_order_frames() -> None:
    with pytest.raises(ValidationError):
        ReplayManifest(
            sequence_id="bad-order",
            source="test",
            frames=[
                FramePacket(
                    frame_id=1,
                    timestamp_ns=200,
                    width_px=10,
                    height_px=10,
                    rgb_path="one.jpg",
                ),
                FramePacket(
                    frame_id=0,
                    timestamp_ns=100,
                    width_px=10,
                    height_px=10,
                    rgb_path="zero.jpg",
                ),
            ],
        )
