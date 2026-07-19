"""Tests for annotated replay frame and video rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sightly_assist.annotated_video import (
    VideoWriter,
    annotate_frame,
    export_annotated_video,
    infer_replay_fps,
)
from sightly_assist.depth_association import CameraPoint, DepthAssociation, DepthStatus
from sightly_assist.motion_estimation import MotionEstimate, MotionStatus, Velocity3D
from sightly_assist.perception import BoundingBox, FramePacket, TrackObservation
from sightly_assist.perception_risk import PerceptionRiskResult, PerceptionRiskStatus
from sightly_assist.replay import ReplayManifest
from sightly_assist.replay_pipeline import ReplayResult
from sightly_assist.risk import assess_relative_risk
from sightly_assist.schemas import Vector2
from sightly_assist.warning_policy import AlertAction, WarningDecision


class MemoryWriter:
    """Capture rendered frames without depending on a system video codec."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return True

    def write(self, image: np.ndarray) -> None:
        self.frames.append(image.copy())

    def release(self) -> None:
        self.released = True


def _frame(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=frame_id * 100_000_000,
        width_px=160,
        height_px=120,
        rgb_path=f"rgb/{frame_id}.png",
        depth_path=f"depth/{frame_id}.npy",
    )


def _manifest(count: int = 3) -> ReplayManifest:
    return ReplayManifest(
        sequence_id="annotated-sequence",
        source="unit-test",
        frames=[_frame(frame_id) for frame_id in range(count)],
    )


def _result(frame_id: int) -> ReplayResult:
    frame = _frame(frame_id)
    track = TrackObservation(
        track_id=4,
        frame_id=frame_id,
        class_name="person",
        confidence=0.96,
        bbox=BoundingBox(x_min=55, y_min=38, x_max=105, y_max=110),
        age_frames=3,
        missed_frames=0,
    )
    point = CameraPoint(x_m=0.1, y_m=0.0, z_m=4.6)
    velocity = Velocity3D(x_mps=0.0, y_mps=0.0, z_mps=-2.0)
    assessment = assess_relative_risk(
        obstacle_id="track-4",
        timestamp_s=frame.timestamp_ns / 1_000_000_000,
        relative_position=Vector2(x=point.x_m, z=point.z_m),
        relative_velocity=Vector2(x=velocity.x_mps, z=velocity.z_mps),
        prediction_horizon_s=5.0,
        collision_boundary_m=1.0,
    )
    return ReplayResult(
        frame=frame,
        detections=(),
        tracks=(track,),
        depth_associations=(
            DepthAssociation(
                track_id=4,
                frame_id=frame_id,
                status=DepthStatus.VALID,
                roi_sample_count=100,
                valid_sample_count=96,
                inlier_sample_count=94,
                valid_fraction=0.96,
                depth_m=point.z_m,
                median_absolute_deviation_m=0.02,
                pixel_u=80.0,
                pixel_v=74.0,
                camera_point=point,
            ),
        ),
        motion_estimates=(
            MotionEstimate(
                track_id=4,
                frame_id=frame_id,
                timestamp_ns=frame.timestamp_ns,
                status=MotionStatus.VALID,
                sample_count=3,
                inlier_count=3,
                time_span_s=0.2,
                position=point,
                velocity=velocity,
                residual_rms_m=0.01,
            ),
        ),
        risk_results=(
            PerceptionRiskResult(
                track_id=4,
                class_name="person",
                status=PerceptionRiskStatus.VALID,
                motion_status=MotionStatus.VALID,
                assessment=assessment,
            ),
        ),
        warning_decision=WarningDecision(
            timestamp_s=frame.timestamp_ns / 1_000_000_000,
            action=AlertAction.STOP,
            should_emit=True,
            reason="predicted collision requires immediate stop",
            track_id=4,
            risk_score=assessment.risk_score,
            time_to_closest_approach_s=assessment.time_to_closest_approach_s,
        ),
    )


def test_annotate_frame_draws_overlays_without_mutating_input() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    original = image.copy()

    annotated = annotate_frame(image, _result(2))

    assert annotated.shape == image.shape
    assert np.array_equal(image, original)
    assert np.count_nonzero(annotated) > 0


def test_infer_replay_fps_uses_median_timestamp_delta() -> None:
    assert infer_replay_fps(_manifest()) == pytest.approx(10.0)


def test_export_annotated_video_writes_every_manifest_frame(tmp_path: Path) -> None:
    writer = MemoryWriter()
    factory_calls: list[tuple[Path, float, tuple[int, int]]] = []

    def writer_factory(
        output_path: Path,
        fps: float,
        dimensions: tuple[int, int],
    ) -> tuple[VideoWriter, str]:
        factory_calls.append((output_path, fps, dimensions))
        return writer, "TEST"

    def image_loader(frame: FramePacket, root: Path) -> np.ndarray:
        assert root == tmp_path
        return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)

    output = tmp_path / "annotated.mp4"
    summary = export_annotated_video(
        _manifest(),
        (_result(frame_id) for frame_id in range(3)),
        tmp_path,
        output,
        image_loader=image_loader,
        writer_factory=writer_factory,
    )

    assert factory_calls == [(output, pytest.approx(10.0), (160, 120))]
    assert len(writer.frames) == 3
    assert writer.released
    assert summary.frame_count == 3
    assert summary.codec == "TEST"
    assert summary.fps == pytest.approx(10.0)


def test_export_rejects_result_count_mismatch_and_releases_writer(tmp_path: Path) -> None:
    writer = MemoryWriter()

    def writer_factory(
        output_path: Path,
        fps: float,
        dimensions: tuple[int, int],
    ) -> tuple[VideoWriter, str]:
        return writer, "TEST"

    with pytest.raises(ValueError, match="count must match"):
        export_annotated_video(
            _manifest(),
            [_result(0), _result(1)],
            tmp_path,
            tmp_path / "short.mp4",
            image_loader=lambda frame, root: np.zeros(
                (frame.height_px, frame.width_px, 3),
                dtype=np.uint8,
            ),
            writer_factory=writer_factory,
        )

    assert writer.released
