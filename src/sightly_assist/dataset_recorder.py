"""Write synchronized RGB-D and IMU capture streams as reproducible replay datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sightly_assist.capture import RgbdImuSource, validate_capture_frame
from sightly_assist.perception import FramePacket
from sightly_assist.replay import ReplayManifest


class DatasetRecorderConfig(BaseModel):
    """Limits and integrity rules for one capture session."""

    sequence_id: str = Field(min_length=1)
    maximum_frames: int | None = Field(default=None, gt=0)
    maximum_duration_s: float | None = Field(default=None, gt=0)
    maximum_sync_error_ns: int = Field(default=20_000_000, ge=0)
    png_compression: int = Field(default=3, ge=0, le=9)
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_limit(self) -> DatasetRecorderConfig:
        if self.maximum_frames is None and self.maximum_duration_s is None:
            raise ValueError("capture must have a frame or duration limit")
        return self


class RecordingSummary(BaseModel):
    """Serializable outcome of a completed atomic recording."""

    sequence_id: str
    output_directory: str
    frame_count: int = Field(gt=0)
    imu_sample_count: int = Field(ge=0)
    first_timestamp_ns: int = Field(ge=0)
    last_timestamp_ns: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    synchronized_frame_count: int = Field(ge=0)
    maximum_observed_sync_error_ns: int = Field(ge=0)
    manifest_path: str


def record_rgbd_dataset(
    source: RgbdImuSource,
    output_directory: Path,
    config: DatasetRecorderConfig,
) -> RecordingSummary:
    """Record a source into an atomic, checksummed, replay-compatible directory."""

    destination = output_directory.resolve()
    staging = destination.parent / f".{destination.name}.partial-{uuid4().hex}"
    if destination.exists() and not config.overwrite:
        raise FileExistsError(f"Capture output already exists: {destination}")
    if staging.exists():
        shutil.rmtree(staging)

    rgb_directory = staging / "rgb"
    depth_directory = staging / "depth"
    imu_directory = staging / "imu"
    rgb_directory.mkdir(parents=True)
    depth_directory.mkdir(parents=True)
    imu_directory.mkdir(parents=True)

    frames: list[FramePacket] = []
    imu_sample_count = 0
    maximum_sync_error_ns = 0
    first_timestamp_ns: int | None = None
    previous_frame_id: int | None = None
    previous_timestamp_ns: int | None = None
    imu_path = imu_directory / "samples.jsonl"

    try:
        with imu_path.open("w", encoding="utf-8", newline="\n") as imu_file:
            for capture in source:
                if config.maximum_frames is not None and len(frames) >= config.maximum_frames:
                    break
                validate_capture_frame(capture, source.intrinsics)
                if previous_frame_id is not None and capture.frame_id <= previous_frame_id:
                    raise ValueError("capture frame IDs must be strictly increasing")
                if (
                    previous_timestamp_ns is not None
                    and capture.timestamp_ns <= previous_timestamp_ns
                ):
                    raise ValueError("capture timestamps must be strictly increasing")
                if first_timestamp_ns is None:
                    first_timestamp_ns = capture.timestamp_ns
                elapsed_s = (capture.timestamp_ns - first_timestamp_ns) / 1_000_000_000
                if config.maximum_duration_s is not None and elapsed_s > config.maximum_duration_s:
                    break

                rgb_relative = Path("rgb") / f"frame_{capture.frame_id:08d}.png"
                depth_relative = Path("depth") / f"frame_{capture.frame_id:08d}.npy"
                rgb_path = staging / rgb_relative
                depth_path = staging / depth_relative
                _write_rgb_png(rgb_path, capture.rgb_bgr, config.png_compression)
                np.save(
                    depth_path, np.asarray(capture.depth_m, dtype=np.float32), allow_pickle=False
                )

                for imu_sample in capture.imu_samples:
                    imu_file.write(imu_sample.model_dump_json() + "\n")
                    imu_sample_count += 1

                maximum_sync_error_ns = max(maximum_sync_error_ns, capture.sync_error_ns)
                frames.append(
                    FramePacket(
                        frame_id=capture.frame_id,
                        timestamp_ns=capture.timestamp_ns,
                        width_px=source.intrinsics.width_px,
                        height_px=source.intrinsics.height_px,
                        rgb_path=rgb_relative.as_posix(),
                        depth_path=depth_relative.as_posix(),
                        synchronized=capture.sync_error_ns <= config.maximum_sync_error_ns,
                        orientation_wxyz=capture.orientation_wxyz,
                        device_sequence_num=capture.device_sequence_num,
                        sync_error_ns=capture.sync_error_ns,
                        rgb_sha256=_sha256(rgb_path),
                        depth_sha256=_sha256(depth_path),
                    )
                )
                previous_frame_id = capture.frame_id
                previous_timestamp_ns = capture.timestamp_ns

        if not frames or first_timestamp_ns is None:
            raise ValueError("capture source produced no frames")

        metadata_relative = Path("capture_metadata.json")
        manifest_relative = Path("manifest.json")
        _write_json(staging / metadata_relative, source.metadata.model_dump(mode="json"))
        manifest = ReplayManifest(
            sequence_id=config.sequence_id,
            source=source.source_name,
            frames=frames,
            intrinsics=source.intrinsics,
            imu_path=imu_path.relative_to(staging).as_posix(),
            capture_metadata_path=metadata_relative.as_posix(),
        )
        _write_json(staging / manifest_relative, manifest.model_dump(mode="json"))

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    last_timestamp_ns = frames[-1].timestamp_ns
    return RecordingSummary(
        sequence_id=config.sequence_id,
        output_directory=str(destination),
        frame_count=len(frames),
        imu_sample_count=imu_sample_count,
        first_timestamp_ns=first_timestamp_ns,
        last_timestamp_ns=last_timestamp_ns,
        duration_s=(last_timestamp_ns - first_timestamp_ns) / 1_000_000_000,
        synchronized_frame_count=sum(frame.synchronized for frame in frames),
        maximum_observed_sync_error_ns=maximum_sync_error_ns,
        manifest_path=str(destination / manifest_relative),
    )


def verify_dataset_checksums(manifest: ReplayManifest, root: Path) -> list[str]:
    """Return replay paths whose stored SHA-256 digest does not match the file."""

    mismatches: list[str] = []
    for frame in manifest.frames:
        for relative_path, expected in (
            (frame.rgb_path, frame.rgb_sha256),
            (frame.depth_path, frame.depth_sha256),
        ):
            if relative_path is None or expected is None:
                continue
            path = root / relative_path
            if not path.is_file() or _sha256(path) != expected:
                mismatches.append(relative_path)
    return mismatches


def _write_rgb_png(path: Path, image_bgr: np.ndarray, compression: int) -> None:
    cv2 = _import_cv2()
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    written = bool(cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, compression]))
    if not written:
        raise OSError(f"OpenCV could not encode RGB frame: {path}")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'OpenCV is required for dataset recording. Install the "vision" extra.'
        ) from exc
    return cv2
