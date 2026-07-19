"""Hardware-independent replay perception processing loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sightly_assist.perception import (
    Detection,
    FramePacket,
    MultiObjectTracker,
    ObjectDetector,
    TrackObservation,
)
from sightly_assist.replay import ReplayManifest, iter_frames

ImageLoader = Callable[[FramePacket, Path], np.ndarray]


@dataclass(frozen=True)
class ReplayResult:
    """Perception outputs for one replay frame."""

    frame: FramePacket
    detections: tuple[Detection, ...]
    tracks: tuple[TrackObservation, ...]


def process_replay(
    manifest: ReplayManifest,
    root: Path,
    detector: ObjectDetector,
    tracker: MultiObjectTracker,
    image_loader: ImageLoader | None = None,
) -> Iterator[ReplayResult]:
    """Process a validated replay sequence in chronological order.

    The optional image loader validates or decodes each RGB file before the
    detector is called. Detectors currently receive frame metadata only; a
    later tensor-frame interface will carry decoded arrays without changing
    the replay sequencing and tracking contracts.
    """

    for frame in iter_frames(manifest):
        if image_loader is not None:
            image_loader(frame, root)
        detections = tuple(detector.predict(frame))
        _validate_detection_frames(frame, detections)
        tracks = tuple(tracker.update(frame, detections))
        yield ReplayResult(frame=frame, detections=detections, tracks=tracks)


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
