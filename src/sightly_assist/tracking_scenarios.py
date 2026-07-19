"""Locked synthetic sequences for reproducible tracker comparisons."""

from __future__ import annotations

from sightly_assist.depth_association import CameraPoint
from sightly_assist.perception import BoundingBox, Detection, FramePacket
from sightly_assist.tracking_benchmark import (
    GroundTruthObservation,
    TrackingBenchmarkFrame,
    TrackingBenchmarkSequence,
)


def locked_tracking_sequences() -> tuple[TrackingBenchmarkSequence, ...]:
    """Return deterministic approach, occlusion, and crossing scenarios."""

    return (
        _smooth_approach(),
        _short_occlusion_approach(),
        _same_class_crossing(),
    )


def _smooth_approach() -> TrackingBenchmarkSequence:
    frames: list[TrackingBenchmarkFrame] = []
    for frame_id in range(10):
        x = 120.0 + 3.0 * frame_id
        box = BoundingBox(x_min=x, y_min=80.0, x_max=x + 42.0, y_max=190.0)
        z_m = 5.5 - 0.45 * frame_id
        frames.append(
            _frame(
                frame_id,
                detections=(
                    Detection(
                        frame_id=frame_id,
                        class_id=0,
                        class_name="person",
                        confidence=0.92,
                        bbox=box,
                    ),
                ),
                truth=(
                    GroundTruthObservation(
                        object_id=1,
                        frame_id=frame_id,
                        class_name="person",
                        bbox=box,
                        camera_point=CameraPoint(x_m=0.05, y_m=0.0, z_m=z_m),
                        expected_collision=True,
                    ),
                ),
            )
        )
    return TrackingBenchmarkSequence(name="smooth_head_on_approach", frames=tuple(frames))


def _short_occlusion_approach() -> TrackingBenchmarkSequence:
    frames: list[TrackingBenchmarkFrame] = []
    for frame_id in range(12):
        x = 40.0 + 8.0 * frame_id
        box = BoundingBox(x_min=x, y_min=95.0, x_max=x + 40.0, y_max=205.0)
        z_m = 6.0 - 0.40 * frame_id
        detections: tuple[Detection, ...]
        if frame_id in {4, 5}:
            detections = ()
        else:
            confidence = 0.38 if frame_id in {3, 6} else 0.90
            detections = (
                Detection(
                    frame_id=frame_id,
                    class_id=0,
                    class_name="person",
                    confidence=confidence,
                    bbox=box,
                ),
            )
        frames.append(
            _frame(
                frame_id,
                detections=detections,
                truth=(
                    GroundTruthObservation(
                        object_id=2,
                        frame_id=frame_id,
                        class_name="person",
                        bbox=box,
                        camera_point=CameraPoint(x_m=0.10, y_m=0.0, z_m=z_m),
                        expected_collision=True,
                    ),
                ),
            )
        )
    return TrackingBenchmarkSequence(name="short_occlusion_approach", frames=tuple(frames))


def _same_class_crossing() -> TrackingBenchmarkSequence:
    frames: list[TrackingBenchmarkFrame] = []
    for frame_id in range(12):
        left_x = 35.0 + 12.0 * frame_id
        right_x = 245.0 - 12.0 * frame_id
        left_box = BoundingBox(
            x_min=left_x,
            y_min=70.0,
            x_max=left_x + 34.0,
            y_max=175.0,
        )
        right_box = BoundingBox(
            x_min=right_x,
            y_min=105.0,
            x_max=right_x + 34.0,
            y_max=210.0,
        )
        detections = (
            Detection(
                frame_id=frame_id,
                class_id=0,
                class_name="person",
                confidence=0.88,
                bbox=left_box,
            ),
            Detection(
                frame_id=frame_id,
                class_id=0,
                class_name="person",
                confidence=0.86,
                bbox=right_box,
            ),
        )
        truth = (
            GroundTruthObservation(
                object_id=10,
                frame_id=frame_id,
                class_name="person",
                bbox=left_box,
            ),
            GroundTruthObservation(
                object_id=11,
                frame_id=frame_id,
                class_name="person",
                bbox=right_box,
            ),
        )
        frames.append(_frame(frame_id, detections=detections, truth=truth))
    return TrackingBenchmarkSequence(name="same_class_crossing", frames=tuple(frames))


def _frame(
    frame_id: int,
    *,
    detections: tuple[Detection, ...],
    truth: tuple[GroundTruthObservation, ...],
) -> TrackingBenchmarkFrame:
    return TrackingBenchmarkFrame(
        frame=FramePacket(
            frame_id=frame_id,
            timestamp_ns=frame_id * 100_000_000,
            width_px=320,
            height_px=240,
            rgb_path=f"rgb/{frame_id:04d}.png",
        ),
        detections=detections,
        ground_truth=truth,
    )
