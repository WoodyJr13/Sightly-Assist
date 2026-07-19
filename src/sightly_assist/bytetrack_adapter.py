"""Adapter from Sightly Assist detections to a pinned real ByteTrack backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from sightly_assist.perception import BoundingBox, Detection, FramePacket, TrackObservation


class TrackingDependencyError(RuntimeError):
    """Raised when the optional tracking backend is unavailable."""


class ByteTrackConfig(BaseModel):
    """Pinned Supervision ByteTrack parameters used by the benchmark."""

    lost_track_buffer: int = Field(ge=0, default=30)
    track_activation_threshold: float = Field(ge=0, le=1, default=0.25)
    minimum_matching_threshold: float = Field(ge=0, le=1, default=0.80)
    frame_rate: float = Field(gt=0, default=10.0)
    minimum_consecutive_frames: int = Field(ge=1, default=1)


class ByteTrackAdapter:
    """Expose Supervision's real ByteTrack through the project tracker protocol."""

    def __init__(self, config: ByteTrackConfig | None = None) -> None:
        self.config = config or ByteTrackConfig()
        supervision = _import_tracking_dependency()
        self._supervision = supervision
        self._tracker: Any = supervision.ByteTrack(
            lost_track_buffer=self.config.lost_track_buffer,
            track_activation_threshold=self.config.track_activation_threshold,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
            minimum_consecutive_frames=self.config.minimum_consecutive_frames,
        )
        self._age_by_id: dict[int, int] = {}
        self._missed_by_id: dict[int, int] = {}
        self._last_frame_id: int | None = None
        self._last_timestamp_ns: int | None = None

    def update(
        self,
        frame: FramePacket,
        detections: Sequence[Detection],
    ) -> Sequence[TrackObservation]:
        """Associate one frame of detections using the real ByteTrack backend."""

        self._validate_chronology(frame)
        class_names = _class_name_map(detections)
        native = self._to_native_detections(detections)
        tracked = self._tracker.update_with_detections(native)

        xyxy = np.asarray(tracked.xyxy, dtype=np.float64)
        tracker_ids = np.asarray(tracked.tracker_id, dtype=np.int64)
        confidences = tracked.confidence
        class_ids = tracked.class_id
        if confidences is None or class_ids is None:
            raise RuntimeError("ByteTrack did not preserve confidence or class metadata")
        confidence_array = np.asarray(confidences, dtype=np.float64)
        class_id_array = np.asarray(class_ids, dtype=np.int64)
        if not (len(xyxy) == len(tracker_ids) == len(confidence_array) == len(class_id_array)):
            raise RuntimeError("ByteTrack returned inconsistent detection arrays")

        for track_id in tuple(self._age_by_id):
            self._age_by_id[track_id] += 1
            self._missed_by_id[track_id] += 1

        observations: list[TrackObservation] = []
        seen_ids: set[int] = set()
        for box, tracker_id_value, confidence, class_id_value in zip(
            xyxy,
            tracker_ids,
            confidence_array,
            class_id_array,
            strict=True,
        ):
            track_id = int(tracker_id_value)
            if track_id < 0:
                continue
            class_id = int(class_id_value)
            class_name = class_names.get(class_id)
            if class_name is None:
                raise RuntimeError(f"ByteTrack returned unknown class ID {class_id}")
            bbox = _clipped_box(box, frame)
            if bbox is None:
                continue

            if track_id not in self._age_by_id:
                self._age_by_id[track_id] = 1
            self._missed_by_id[track_id] = 0
            seen_ids.add(track_id)
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    frame_id=frame.frame_id,
                    class_name=class_name,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    bbox=bbox,
                    age_frames=self._age_by_id[track_id],
                    missed_frames=0,
                )
            )

        expired = [
            track_id
            for track_id, missed in self._missed_by_id.items()
            if track_id not in seen_ids and missed > self.config.lost_track_buffer
        ]
        for track_id in expired:
            self._age_by_id.pop(track_id, None)
            self._missed_by_id.pop(track_id, None)

        return tuple(sorted(observations, key=lambda item: item.track_id))

    def reset(self) -> None:
        """Clear native and adapter state before processing another sequence."""

        self._tracker.reset()
        self._age_by_id.clear()
        self._missed_by_id.clear()
        self._last_frame_id = None
        self._last_timestamp_ns = None

    def _to_native_detections(self, detections: Sequence[Detection]) -> Any:
        xyxy = np.asarray(
            [
                [
                    detection.bbox.x_min,
                    detection.bbox.y_min,
                    detection.bbox.x_max,
                    detection.bbox.y_max,
                ]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 4))
        confidence = np.asarray(
            [detection.confidence for detection in detections],
            dtype=np.float32,
        )
        class_id = np.asarray(
            [detection.class_id for detection in detections],
            dtype=np.int32,
        )
        return self._supervision.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    def _validate_chronology(self, frame: FramePacket) -> None:
        if self._last_frame_id is not None and frame.frame_id <= self._last_frame_id:
            raise ValueError("ByteTrack frames must have strictly increasing frame IDs")
        if self._last_timestamp_ns is not None and frame.timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("ByteTrack frames must have strictly increasing timestamps")
        self._last_frame_id = frame.frame_id
        self._last_timestamp_ns = frame.timestamp_ns


def _import_tracking_dependency() -> Any:
    try:
        import supervision
    except ImportError as exc:
        raise TrackingDependencyError(
            'ByteTrack is not installed. Install Sightly Assist with the "tracking" extra.'
        ) from exc
    return supervision


def _class_name_map(detections: Sequence[Detection]) -> dict[int, str]:
    names: dict[int, str] = {}
    for detection in detections:
        existing = names.get(detection.class_id)
        if existing is not None and existing != detection.class_name:
            raise ValueError(
                f"Class ID {detection.class_id} maps to both {existing!r} "
                f"and {detection.class_name!r}"
            )
        names[detection.class_id] = detection.class_name
    return names


def _clipped_box(box: np.ndarray, frame: FramePacket) -> BoundingBox | None:
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        return None
    x_min = float(np.clip(box[0], 0.0, frame.width_px - 1.0))
    y_min = float(np.clip(box[1], 0.0, frame.height_px - 1.0))
    x_max = float(np.clip(box[2], 0.0, float(frame.width_px)))
    y_max = float(np.clip(box[3], 0.0, float(frame.height_px)))
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
