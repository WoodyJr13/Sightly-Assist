"""Hardware-independent replay perception and depth processing loop."""

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
from sightly_assist.opencv_io import load_rgb
from sightly_assist.perception import (
    Detection,
    FramePacket,
    MultiObjectTracker,
    ObjectDetector,
    TrackObservation,
)
from sightly_assist.replay import ReplayManifest, iter_frames

ImageLoader = Callable[[FramePacket, Path], np.ndarray]
DepthLoader = Callable[[FramePacket, Path], np.ndarray | None]


@dataclass(frozen=True)
class ReplayResult:
    """Perception and depth outputs for one replay frame."""

    frame: FramePacket
    detections: tuple[Detection, ...]
    tracks: tuple[TrackObservation, ...]
    depth_associations: tuple[DepthAssociation, ...] = ()


def process_replay(
    manifest: ReplayManifest,
    root: Path,
    detector: ObjectDetector,
    tracker: MultiObjectTracker,
    image_loader: ImageLoader | None = None,
    depth_loader: DepthLoader | None = None,
    camera_intrinsics: CameraIntrinsics | None = None,
    depth_config: DepthAssociationConfig | None = None,
) -> Iterator[ReplayResult]:
    """Decode, detect, track, and depth-associate a replay sequence.

    OpenCV is used by default to decode RGB frames. Tests and alternate sensor
    backends can provide another loader. Depth association is enabled only when
    both a depth loader and matching camera intrinsics are supplied.
    """

    if (depth_loader is None) != (camera_intrinsics is None):
        raise ValueError(
            "depth_loader and camera_intrinsics must either both be provided or both be omitted"
        )

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

        yield ReplayResult(
            frame=frame,
            detections=detections,
            tracks=tracks,
            depth_associations=depth_associations,
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
