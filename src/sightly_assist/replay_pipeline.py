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
    """Process a validated replay sequence in chronological order.

    RGB decoding remains optional so hardware-independent tests and simulations
    do not require OpenCV. Depth association is enabled only when both a depth
    loader and matching camera intrinsics are supplied. Frames without depth
    still produce explicit ``NO_DEPTH`` associations for every active track.
    """

    if (depth_loader is None) != (camera_intrinsics is None):
        raise ValueError(
            "depth_loader and camera_intrinsics must either both be provided or both be omitted"
        )

    for frame in iter_frames(manifest):
        if image_loader is not None:
            image_loader(frame, root)
        detections = tuple(detector.predict(frame))
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
