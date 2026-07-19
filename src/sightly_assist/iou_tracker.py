"""Transparent intersection-over-union tracking baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sightly_assist.perception import Detection, FramePacket, TrackObservation


@dataclass
class _TrackState:
    track_id: int
    class_name: str
    confidence: float
    bbox: object
    age_frames: int
    missed_frames: int


class IoUTracker:
    """Associate detections between frames using class-aware box overlap.

    This tracker is intentionally simple and deterministic. It serves as a
    transparent baseline before integrating and benchmarking ByteTrack or
    another motion-aware tracker.
    """

    def __init__(self, minimum_iou: float = 0.3, max_missed_frames: int = 2) -> None:
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_iou must be between zero and one")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.minimum_iou = minimum_iou
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 0
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        frame: FramePacket,
        detections: Sequence[Detection],
    ) -> Sequence[TrackObservation]:
        """Return class-aware IoU assignments for one frame."""

        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indices = set(range(len(detections)))
        assignments: list[tuple[int, int]] = []

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if detection.class_name != track.class_name:
                    continue
                iou = track.bbox.intersection_over_union(detection.bbox)  # type: ignore[attr-defined]
                if iou >= self.minimum_iou:
                    candidates.append((iou, track_id, detection_index))

        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id not in unmatched_track_ids:
                continue
            if detection_index not in unmatched_detection_indices:
                continue
            assignments.append((track_id, detection_index))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(detection_index)

        observations: list[TrackObservation] = []
        for track_id, detection_index in assignments:
            detection = detections[detection_index]
            track = self._tracks[track_id]
            track.confidence = detection.confidence
            track.bbox = detection.bbox
            track.age_frames += 1
            track.missed_frames = 0
            observations.append(self._observation(frame, track))

        for detection_index in sorted(unmatched_detection_indices):
            detection = detections[detection_index]
            track = _TrackState(
                track_id=self._next_track_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                age_frames=1,
                missed_frames=0,
            )
            self._tracks[track.track_id] = track
            self._next_track_id += 1
            observations.append(self._observation(frame, track))

        expired: list[int] = []
        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.age_frames += 1
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                expired.append(track_id)

        for track_id in expired:
            del self._tracks[track_id]

        return tuple(sorted(observations, key=lambda item: item.track_id))

    @staticmethod
    def _observation(frame: FramePacket, track: _TrackState) -> TrackObservation:
        return TrackObservation(
            track_id=track.track_id,
            frame_id=frame.frame_id,
            class_name=track.class_name,
            confidence=track.confidence,
            bbox=track.bbox,  # type: ignore[arg-type]
            age_frames=track.age_frames,
            missed_frames=track.missed_frames,
        )
