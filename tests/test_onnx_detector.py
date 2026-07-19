"""Tests for the ONNX YOLO detector backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from sightly_assist.onnx_detector import OnnxYoloDetector
from sightly_assist.perception import FramePacket


@dataclass
class FakeInput:
    name: str = "images"
    shape: Sequence[int | str | None] = (1, 3, 640, 640)


class FakeSession:
    def __init__(
        self,
        output: np.ndarray,
        shape: Sequence[int | str | None] = (1, 3, 640, 640),
    ) -> None:
        self.output = output
        self.input = FakeInput(shape=shape)
        self.last_feed: Mapping[str, np.ndarray] | None = None

    def get_inputs(self) -> Sequence[FakeInput]:
        return (self.input,)

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, np.ndarray],
    ) -> Sequence[np.ndarray]:
        assert output_names is None
        self.last_feed = input_feed
        return (self.output,)


def _frame() -> FramePacket:
    return FramePacket(
        frame_id=7,
        timestamp_ns=1,
        width_px=320,
        height_px=160,
        rgb_path="frame.png",
    )


def test_detector_decodes_ultralytics_raw_output_and_restores_coordinates() -> None:
    output = np.array(
        [
            [
                [320.0, 100.0],
                [320.0, 100.0],
                [320.0, 20.0],
                [160.0, 20.0],
                [0.90, 0.10],
                [0.10, 0.20],
                [0.05, 0.30],
            ]
        ],
        dtype=np.float32,
    )
    session = FakeSession(output)
    detector = OnnxYoloDetector(
        class_names=("person", "bicycle", "car"),
        allowed_class_names=None,
        session=session,
    )
    image = np.zeros((160, 320, 3), dtype=np.uint8)

    detections = detector.predict(_frame(), image)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.frame_id == 7
    assert detection.class_name == "person"
    assert detection.confidence == pytest.approx(0.9)
    assert detection.bbox.x_min == pytest.approx(80.0)
    assert detection.bbox.x_max == pytest.approx(240.0)
    assert detection.bbox.y_min == pytest.approx(40.0)
    assert detection.bbox.y_max == pytest.approx(120.0)
    assert session.last_feed is not None
    tensor = session.last_feed["images"]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_detector_applies_class_aware_non_maximum_suppression() -> None:
    output = np.array(
        [
            [320.0, 320.0, 320.0, 160.0, 0.90, 0.05, 0.05],
            [325.0, 320.0, 320.0, 160.0, 0.80, 0.10, 0.10],
            [320.0, 320.0, 320.0, 160.0, 0.05, 0.05, 0.75],
        ],
        dtype=np.float32,
    )[np.newaxis, ...]
    detector = OnnxYoloDetector(
        class_names=("person", "bicycle", "car"),
        allowed_class_names=None,
        iou_threshold=0.5,
        session=FakeSession(output),
    )

    detections = detector.predict(_frame(), np.zeros((160, 320, 3), dtype=np.uint8))

    assert [item.class_name for item in detections] == ["person", "car"]
    assert [item.confidence for item in detections] == pytest.approx([0.90, 0.75])


def test_detector_filters_disallowed_classes() -> None:
    output = np.array(
        [[[320.0, 320.0, 100.0, 100.0, 0.05, 0.10, 0.95]]],
        dtype=np.float32,
    )
    detector = OnnxYoloDetector(
        class_names=("person", "bicycle", "car"),
        allowed_class_names={"person"},
        session=FakeSession(output),
    )

    detections = detector.predict(_frame(), np.zeros((160, 320, 3), dtype=np.uint8))

    assert detections == ()


def test_dynamic_input_requires_explicit_size() -> None:
    output = np.zeros((1, 7, 1), dtype=np.float32)
    session = FakeSession(output, shape=(1, 3, "height", "width"))

    with pytest.raises(ValueError, match="Dynamic ONNX inputs"):
        OnnxYoloDetector(
            class_names=("person", "bicycle", "car"),
            session=session,
        )


def test_end_to_end_nms_output_is_supported() -> None:
    output = np.array(
        [[[160.0, 240.0, 480.0, 400.0, 0.88, 0.0]]],
        dtype=np.float32,
    )
    detector = OnnxYoloDetector(
        class_names=("person", "bicycle", "car"),
        allowed_class_names=None,
        session=FakeSession(output),
    )

    detections = detector.predict(_frame(), np.zeros((160, 320, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].bbox.x_min == pytest.approx(80.0)
    assert detections[0].bbox.y_min == pytest.approx(40.0)
