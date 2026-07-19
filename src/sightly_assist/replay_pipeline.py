"""Hardware-independent replay perception, geometry, risk, and warning loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from pydantic import BaseModel, Field

from sightly_assist.depth_association import (
    CameraIntrinsics,
    DepthAssociation,
    DepthAssociationConfig,
    associate_tracks_depth,
)
from sightly_assist.free_space import (
    FreeSpaceAnalysis,
    FreeSpaceConfig,
    analyze_free_space,
)
from sightly_assist.motion_estimation import MotionEstimate, TrackMotionEstimator
from sightly_assist.opencv_io import load_rgb
from sightly_assist.perception import (
    Detection,
    FramePacket,
    MultiObjectTracker,
    ObjectDetector,
    TrackObservation,
)
from sightly_assist.perception_risk import (
    PerceptionRiskConfig,
    PerceptionRiskResult,
    assess_perception_risks,
)
from sightly_assist.replay import ReplayManifest, iter_frames
from sightly_assist.warning_policy import WarningDecision, WarningPolicy

ImageLoader = Callable[[FramePacket, Path], np.ndarray]
DepthLoader = Callable[[FramePacket, Path], np.ndarray | None]


class ReplayStageTimings(BaseModel):
    """Wall-clock processing time for each replay stage in milliseconds."""

    image_load_ms: float = Field(ge=0)
    detector_ms: float = Field(ge=0)
    tracker_ms: float = Field(ge=0)
    depth_load_ms: float = Field(ge=0)
    depth_association_ms: float = Field(ge=0)
    free_space_ms: float = Field(ge=0)
    motion_ms: float = Field(ge=0)
    risk_ms: float = Field(ge=0)
    warning_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


@dataclass(frozen=True)
class ReplayResult:
    """Outputs produced for one replay frame."""

    frame: FramePacket
    detections: tuple[Detection, ...]
    tracks: tuple[TrackObservation, ...]
    depth_associations: tuple[DepthAssociation, ...] = ()
    free_space_analysis: FreeSpaceAnalysis | None = None
    motion_estimates: tuple[MotionEstimate, ...] = ()
    risk_results: tuple[PerceptionRiskResult, ...] = ()
    warning_decision: WarningDecision | None = None
    stage_timings: ReplayStageTimings | None = None


def process_replay(
    manifest: ReplayManifest,
    root: Path,
    detector: ObjectDetector,
    tracker: MultiObjectTracker,
    image_loader: ImageLoader | None = None,
    depth_loader: DepthLoader | None = None,
    camera_intrinsics: CameraIntrinsics | None = None,
    depth_config: DepthAssociationConfig | None = None,
    free_space_config: FreeSpaceConfig | None = None,
    motion_estimator: TrackMotionEstimator | None = None,
    risk_config: PerceptionRiskConfig | None = None,
    warning_policy: WarningPolicy | None = None,
) -> Iterator[ReplayResult]:
    """Run a recorded sequence through perception and warning decisions.

    OpenCV is used by default to decode RGB frames. Tests and alternate sensor
    backends can provide another loader. Object depth association and free-space
    analysis share the same synchronized depth map. Each later stage requires
    the outputs and configuration of its preceding stage.
    """

    if (depth_loader is None) != (camera_intrinsics is None):
        raise ValueError(
            "depth_loader and camera_intrinsics must either both be provided or both be omitted"
        )
    if motion_estimator is not None and (depth_loader is None or camera_intrinsics is None):
        raise ValueError("motion_estimator requires depth_loader and camera_intrinsics")
    if free_space_config is not None and (depth_loader is None or camera_intrinsics is None):
        raise ValueError("free_space_config requires depth_loader and camera_intrinsics")
    if risk_config is not None and motion_estimator is None:
        raise ValueError("risk_config requires motion_estimator")
    if warning_policy is not None and risk_config is None:
        raise ValueError("warning_policy requires risk_config")

    resolved_image_loader = image_loader or load_rgb
    for frame in iter_frames(manifest):
        frame_started_ns = perf_counter_ns()

        stage_started_ns = perf_counter_ns()
        image_bgr = resolved_image_loader(frame, root)
        image_load_ms = _elapsed_ms(stage_started_ns)
        _validate_image(frame, image_bgr)

        stage_started_ns = perf_counter_ns()
        detections = tuple(detector.predict(frame, image_bgr))
        detector_ms = _elapsed_ms(stage_started_ns)
        _validate_detection_frames(frame, detections)

        stage_started_ns = perf_counter_ns()
        tracks = tuple(tracker.update(frame, detections))
        tracker_ms = _elapsed_ms(stage_started_ns)

        depth_map: np.ndarray | None = None
        depth_associations: tuple[DepthAssociation, ...] = ()
        depth_load_ms = 0.0
        depth_association_ms = 0.0
        if depth_loader is not None and camera_intrinsics is not None:
            stage_started_ns = perf_counter_ns()
            depth_map = depth_loader(frame, root)
            depth_load_ms = _elapsed_ms(stage_started_ns)

            stage_started_ns = perf_counter_ns()
            depth_associations = associate_tracks_depth(
                depth_map,
                frame,
                tracks,
                camera_intrinsics,
                depth_config,
            )
            depth_association_ms = _elapsed_ms(stage_started_ns)

        free_space_analysis: FreeSpaceAnalysis | None = None
        free_space_ms = 0.0
        if free_space_config is not None and camera_intrinsics is not None:
            stage_started_ns = perf_counter_ns()
            free_space_analysis = analyze_free_space(
                depth_map,
                frame,
                camera_intrinsics,
                free_space_config,
            )
            free_space_ms = _elapsed_ms(stage_started_ns)

        motion_estimates: tuple[MotionEstimate, ...] = ()
        motion_ms = 0.0
        if motion_estimator is not None:
            stage_started_ns = perf_counter_ns()
            motion_estimates = motion_estimator.update(frame, depth_associations)
            motion_ms = _elapsed_ms(stage_started_ns)

        risk_results: tuple[PerceptionRiskResult, ...] = ()
        risk_ms = 0.0
        timestamp_s = frame.timestamp_ns / 1_000_000_000
        if risk_config is not None:
            stage_started_ns = perf_counter_ns()
            risk_results = assess_perception_risks(
                tracks,
                motion_estimates,
                timestamp_s,
                risk_config,
            )
            risk_ms = _elapsed_ms(stage_started_ns)

        warning_decision: WarningDecision | None = None
        warning_ms = 0.0
        if warning_policy is not None:
            stage_started_ns = perf_counter_ns()
            warning_decision = warning_policy.evaluate(risk_results, timestamp_s)
            warning_ms = _elapsed_ms(stage_started_ns)

        timings = ReplayStageTimings(
            image_load_ms=image_load_ms,
            detector_ms=detector_ms,
            tracker_ms=tracker_ms,
            depth_load_ms=depth_load_ms,
            depth_association_ms=depth_association_ms,
            free_space_ms=free_space_ms,
            motion_ms=motion_ms,
            risk_ms=risk_ms,
            warning_ms=warning_ms,
            total_ms=_elapsed_ms(frame_started_ns),
        )
        yield ReplayResult(
            frame=frame,
            detections=detections,
            tracks=tracks,
            depth_associations=depth_associations,
            free_space_analysis=free_space_analysis,
            motion_estimates=motion_estimates,
            risk_results=risk_results,
            warning_decision=warning_decision,
            stage_timings=timings,
        )


def _elapsed_ms(started_ns: int) -> float:
    return max(0.0, (perf_counter_ns() - started_ns) / 1_000_000)


def _validate_image(frame: FramePacket, image_bgr: np.ndarray) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("RGB loader must return a three-channel image")
    if image_bgr.shape[:2] != (frame.height_px, frame.width_px):
        raise ValueError("Decoded RGB dimensions must match the replay frame")
    if not np.issubdtype(image_bgr.dtype, np.number):
        raise ValueError("Decoded RGB image must use a numeric data type")


def _validate_detection_frames(
    frame: FramePacket,
    detections: tuple[Detection, ...],
) -> None:
    for detection in detections:
        if detection.frame_id != frame.frame_id:
            raise ValueError(
                "Detector returned a detection for the wrong frame: "
                f"expected {frame.frame_id}, got {detection.frame_id}"
            )
