"""Hardware-independent perception interfaces and data models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned image bounding box in pixel coordinates."""

    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(gt=0)
    y_max: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_extents(self) -> BoundingBox:
        """Require positive width and height."""

        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        return self

    @property
    def width(self) -> float:
        """Return box width in pixels."""

        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return box height in pixels."""

        return self.y_max - self.y_min


class FramePacket(BaseModel):
    """Metadata for one synchronized frame without embedding image bytes."""

    frame_id: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    rgb_path: str = Field(min_length=1)
    depth_path: str | None = None
    synchronized: bool = True


class Detection(BaseModel):
    """One object detection produced from a frame."""

    frame_id: int = Field(ge=0)
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class TrackObservation(BaseModel):
    """One tracked object observation in image coordinates."""

    track_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    age_frames: int = Field(ge=1)
    missed_frames: int = Field(ge=0)


@runtime_checkable
class ObjectDetector(Protocol):
    """Interface implemented by mock, OpenCV, ONNX, and TensorRT detectors."""

    def predict(self, frame: FramePacket) -> Sequence[Detection]:
        """Return detections for one frame."""


@runtime_checkable
class MultiObjectTracker(Protocol):
    """Interface implemented by ByteTrack or another tracking backend."""

    def update(
        self,
        frame: FramePacket,
        detections: Sequence[Detection],
    ) -> Sequence[TrackObservation]:
        """Associate detections with persistent track identities."""


class EmptyDetector:
    """Deterministic detector used for pipeline and replay tests."""

    def predict(self, frame: FramePacket) -> Sequence[Detection]:
        """Return no detections."""

        del frame
        return ()
