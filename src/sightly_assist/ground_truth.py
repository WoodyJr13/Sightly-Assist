"""Ground-truth schemas and annotation-template generation for replay evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from sightly_assist.depth_association import CameraPoint
from sightly_assist.motion_estimation import Velocity3D
from sightly_assist.perception import BoundingBox
from sightly_assist.replay import ReplayManifest
from sightly_assist.warning_policy import AlertAction


class GroundTruthObject(BaseModel):
    """One annotated object in one replay frame."""

    object_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    bbox: BoundingBox
    depth_m: float | None = Field(default=None, gt=0)
    position: CameraPoint | None = None
    velocity: Velocity3D | None = None
    occluded: bool = False
    ignore: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def validate_depth_and_position(self) -> GroundTruthObject:
        """Keep separately entered depth and 3D position mutually consistent."""

        if (
            self.depth_m is not None
            and self.position is not None
            and abs(self.depth_m - self.position.z_m) > 0.05
        ):
            raise ValueError("depth_m and position.z_m must agree within 0.05 m")
        return self


class GroundTruthFrame(BaseModel):
    """All human or instrument annotations for one replay frame."""

    frame_id: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    objects: tuple[GroundTruthObject, ...] = Field(default_factory=tuple)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_object_ids(self) -> GroundTruthFrame:
        """Require each physical object identifier at most once per frame."""

        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("ground-truth object IDs must be unique within each frame")
        return self


class GroundTruthEvent(BaseModel):
    """A labeled hazard interval for one physical object.

    ``critical_timestamp_ns`` is the measured or staged collision/closest-approach
    time used to calculate warning lead time. The interval may begin before that
    instant and end after it.
    """

    event_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    start_timestamp_ns: int = Field(ge=0)
    critical_timestamp_ns: int = Field(ge=0)
    end_timestamp_ns: int = Field(ge=0)
    minimum_warning_action: AlertAction = AlertAction.SLOW
    notes: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> GroundTruthEvent:
        """Require an ordered interval and an actual hazard-warning action."""

        if not self.start_timestamp_ns <= self.critical_timestamp_ns <= self.end_timestamp_ns:
            raise ValueError(
                "event timestamps must satisfy start <= critical <= end"
            )
        if self.minimum_warning_action in {AlertAction.NO_ALERT, AlertAction.ABSTAIN}:
            raise ValueError("minimum_warning_action must represent a hazard warning")
        return self


class GroundTruthAnnotationSet(BaseModel):
    """Versioned annotations for one replay sequence."""

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    sequence_id: str = Field(min_length=1)
    annotator: str = Field(min_length=1)
    coordinate_frame: str = Field(default="camera_right_down_forward", min_length=1)
    frames: tuple[GroundTruthFrame, ...] = Field(min_length=1)
    events: tuple[GroundTruthEvent, ...] = Field(default_factory=tuple)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sequence(self) -> GroundTruthAnnotationSet:
        """Require chronological frames and events referring to annotated objects."""

        frame_ids = [frame.frame_id for frame in self.frames]
        timestamps = [frame.timestamp_ns for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("annotation frame IDs must be unique")
        if frame_ids != sorted(frame_ids):
            raise ValueError("annotation frames must be ordered by frame_id")
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("annotation timestamps must be unique")
        if timestamps != sorted(timestamps):
            raise ValueError("annotation frames must be ordered by timestamp")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique")

        object_ids = {
            item.object_id
            for frame in self.frames
            for item in frame.objects
            if not item.ignore
        }
        minimum_timestamp = timestamps[0]
        maximum_timestamp = timestamps[-1]
        for event in self.events:
            if event.object_id not in object_ids:
                raise ValueError(
                    f"event {event.event_id!r} references unknown object {event.object_id!r}"
                )
            if event.start_timestamp_ns < minimum_timestamp:
                raise ValueError("event starts before the annotated sequence")
            if event.end_timestamp_ns > maximum_timestamp:
                raise ValueError("event ends after the annotated sequence")
        return self


def load_ground_truth(path: Path) -> GroundTruthAnnotationSet:
    """Load and validate a JSON ground-truth annotation set."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    return GroundTruthAnnotationSet.model_validate(raw)


def save_ground_truth(annotations: GroundTruthAnnotationSet, path: Path) -> Path:
    """Write validated annotations as deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(annotations.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def create_annotation_template(
    manifest: ReplayManifest,
    annotator: str,
) -> GroundTruthAnnotationSet:
    """Create an empty frame-aligned annotation set from a replay manifest."""

    if not annotator.strip():
        raise ValueError("annotator must not be blank")
    return GroundTruthAnnotationSet(
        sequence_id=manifest.sequence_id,
        annotator=annotator.strip(),
        frames=tuple(
            GroundTruthFrame(
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
            )
            for frame in manifest.frames
        ),
        metadata={
            "source": manifest.source,
            "instructions": (
                "Annotate persistent object IDs, boxes, and externally measured values; "
                "do not copy model predictions into ground truth."
            ),
        },
    )


def validate_annotations_against_manifest(
    annotations: GroundTruthAnnotationSet,
    manifest: ReplayManifest,
) -> None:
    """Require exact frame and timestamp alignment with the source replay."""

    if annotations.sequence_id != manifest.sequence_id:
        raise ValueError("annotation sequence_id does not match replay manifest")
    if len(annotations.frames) != len(manifest.frames):
        raise ValueError("annotations must contain exactly one entry per replay frame")
    for annotated, recorded in zip(annotations.frames, manifest.frames, strict=True):
        if annotated.frame_id != recorded.frame_id:
            raise ValueError("annotation frame IDs do not match replay manifest")
        if annotated.timestamp_ns != recorded.timestamp_ns:
            raise ValueError("annotation timestamps do not match replay manifest")
