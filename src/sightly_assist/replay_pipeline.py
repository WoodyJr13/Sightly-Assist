"""Hardware-independent replay perception, depth, motion, risk, and warning loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sightly_assist.depth_association import (
    CameraIntrinsics,
    DepthAssociation,
    DepthAssociationConfig,
    associate_tracks_depth,
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


@dataclass(frozen=True)
class ReplayResult:
    """Perception, depth, motion, risk, and warning outputs for one replay frame."""

    frame: FramePacket
    detections: tuple[Detection, ...]
    tracks: tuple[TrackObservation, ...]
    depth_associations: tuple[DepthAssociation, ...] = ()
    motion_estimates: tuple[MotionEstimate, ...] = ()
    risk_results: tuple[PerceptionRiskResult, ...] = ()
    warning_decision: WarningDecision | None = None


def process_replay(
    manifest: ReplayManifest,
    root: Path,
    detector: ObjectDetector,
    tracker: MultiObjectTracker,
    image_loader: ImageLoader | None = None,
    depth_loader: DepthLoader | None = None,
    camera_intrinsics: CameraIntrinsics | None = None,
    depth_config: DepthAssociationConfig | None = None,
    motion_estimator: TrackMotionEstimator | None = None,
    risk_config: PerceptionRiskConfig | None = None,
    warning_policy: WarningPolicy | None = None,
) -> Iterator[ReplayResult]:
    """Run the recorded-data pipeline through a constrained warning decision.

    OpenCV is used by default to decode RGB frames. Tests and alternate sensor
    backends can provide another loader. Each later stage requires the outputs
    and configuration of its preceding stage.
    """

    if (depth_loader is None) != (camera_intrinsics is None):
        raise ValueError(
            "depth_loader and camera_intrinsics must either both be provided or both be omitted"
        )
    if motion_estimator is not None and (depth_loader is None or camera_intrinsics is None):
        raise ValueError("motion_estimator requires depth_loader and camera_intrinsics")
    if risk_config is not None and motion_estimator is None:
        raise ValueError("risk_config requires motion_estimator")
    if warning_policy is not None and risk_config is None:
        raise ValueError("warning_policy requires risk_config")

    resolved_image_loader = image_loader or load_rgb
    for frame in iter_frames(manifest):
        image_bgr = resolved_image_loader(frame, root)
        _validate_image(frame, image_bgr)
        detections = tuple(detector.predict(frame, image_bgr))
        _validate_detection_frames(frame, detections)
        tracks = tuple(tracker.update(frame, detections))

        depth_associations: tuple[DepthAssociation, ...] = ()
        if depth_loader is not None and camera_intrinsics is not None:
            depth_map = depth_loader(frame, root)
            depth_associations = associate_tracks_depth(
                depth_map,
                frame,
                tracks,
                camera_intrinsics,
                depth_config,
            )

        motion_estimates: tuple[MotionEstimate, ...] = ()
        if motion_estimator is not None:
            motion_estimates = motion_estimator.update(frame, depth_associations)

        risk_results: tuple[PerceptionRiskResult, ...] = ()
        timestamp_s = frame.timestamp_ns / 1_000_000_000
        if risk_config is not None:
            risk_results = assess_perception_risks(
                tracks,
                motion_estimates,
                timestamp_s,
                risk_config,
            )

        warning_decision: WarningDecision | None = None
        if warning_policy is not None:
            warning_decision = warning_policy.evaluate(risk_results, timestamp_s)

        yield ReplayResult(
            frame=frame,
            detections=detections,
            tracks=tracks,
            depth_associations=depth_associations,
            motion_estimates=motion_estimates,
            risk_results=risk_results,
            warning_decision=warning_decision,
        )


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
