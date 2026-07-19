"""Recorded-sequence manifest loading and validation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.perception import FramePacket


class ReplayManifest(BaseModel):
    """Description of a recorded RGB or RGB-D sequence."""

    sequence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    frames: list[FramePacket] = Field(min_length=1)
    intrinsics: CameraIntrinsics | None = None
    imu_path: str | None = None
    capture_metadata_path: str | None = None
    format_version: int = Field(ge=1, default=1)

    @model_validator(mode="after")
    def validate_sequence(self) -> ReplayManifest:
        """Require unique ordered frame IDs and nondecreasing timestamps."""

        frame_ids = [frame.frame_id for frame in self.frames]
        timestamps = [frame.timestamp_ns for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("frame IDs must be unique")
        if frame_ids != sorted(frame_ids):
            raise ValueError("frames must be ordered by frame_id")
        if timestamps != sorted(timestamps):
            raise ValueError("frames must be ordered by timestamp")
        return self


def load_manifest(path: Path) -> ReplayManifest:
    """Load and validate a JSON replay manifest."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    return ReplayManifest.model_validate(raw)


def iter_frames(manifest: ReplayManifest) -> Iterator[FramePacket]:
    """Yield replay frames in validated chronological order."""

    yield from manifest.frames


def validate_replay_files(manifest: ReplayManifest, root: Path) -> list[str]:
    """Return missing frame, IMU, or capture-metadata paths referenced by a manifest."""

    missing: list[str] = []
    for frame in manifest.frames:
        for relative_path in (frame.rgb_path, frame.depth_path):
            if relative_path is not None and not (root / relative_path).is_file():
                missing.append(relative_path)
    for relative_path in (manifest.imu_path, manifest.capture_metadata_path):
        if relative_path is not None and not (root / relative_path).is_file():
            missing.append(relative_path)
    return missing
