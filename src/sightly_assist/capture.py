"""Hardware-independent contracts for synchronized RGB-D and IMU capture."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sightly_assist.depth_association import CameraIntrinsics


class ImuSample(BaseModel):
    """One timestamped inertial sample in the Luxonis right/down/forward frame."""

    timestamp_ns: int = Field(ge=0)
    sequence_num: int | None = Field(default=None, ge=0)
    acceleration_mps2: tuple[float, float, float] | None = None
    angular_velocity_radps: tuple[float, float, float] | None = None
    orientation_wxyz: tuple[float, float, float, float] | None = None
    orientation_accuracy_rad: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_measurements(self) -> ImuSample:
        values = (
            self.acceleration_mps2,
            self.angular_velocity_radps,
            self.orientation_wxyz,
        )
        if all(value is None for value in values):
            raise ValueError("an IMU sample must contain at least one measurement")
        for vector in values:
            if vector is not None and not all(isfinite(component) for component in vector):
                raise ValueError("IMU measurements must contain finite values")
        if self.orientation_accuracy_rad is not None and not isfinite(
            self.orientation_accuracy_rad
        ):
            raise ValueError("orientation accuracy must be finite")
        return self


class CaptureSourceMetadata(BaseModel):
    """Serializable device and stream information stored with every dataset."""

    source_name: str = Field(min_length=1)
    device_id: str | None = None
    device_model: str | None = None
    depthai_version: str | None = None
    firmware_version: str | None = None
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    requested_fps: float = Field(gt=0)
    depth_unit: str = "meters"
    imu_report_rate_hz: int | None = Field(default=None, gt=0)
    orientation_available: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RgbdCaptureFrame:
    """One synchronized color/depth pair plus inertial samples since the prior frame."""

    frame_id: int
    timestamp_ns: int
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    imu_samples: tuple[ImuSample, ...] = ()
    orientation_wxyz: tuple[float, float, float, float] | None = None
    device_sequence_num: int | None = None
    sync_error_ns: int = 0


def validate_capture_frame(
    frame: RgbdCaptureFrame,
    intrinsics: CameraIntrinsics,
) -> None:
    """Validate array geometry, chronology metadata, and finite orientation values."""

    if frame.frame_id < 0 or frame.timestamp_ns < 0:
        raise ValueError("capture frame IDs and timestamps must be non-negative")
    if frame.sync_error_ns < 0:
        raise ValueError("sync_error_ns must be non-negative")
    expected_shape = (intrinsics.height_px, intrinsics.width_px)
    if frame.rgb_bgr.shape != (*expected_shape, 3):
        raise ValueError(
            "RGB frame shape must match camera intrinsics: "
            f"expected {(*expected_shape, 3)}, got {frame.rgb_bgr.shape}"
        )
    if frame.depth_m.shape != expected_shape:
        raise ValueError(
            "Depth frame shape must match camera intrinsics: "
            f"expected {expected_shape}, got {frame.depth_m.shape}"
        )
    if not np.issubdtype(frame.rgb_bgr.dtype, np.number):
        raise ValueError("RGB frame must use a numeric data type")
    if not np.issubdtype(frame.depth_m.dtype, np.number):
        raise ValueError("Depth frame must use a numeric data type")
    if frame.orientation_wxyz is not None and not all(
        isfinite(value) for value in frame.orientation_wxyz
    ):
        raise ValueError("frame orientation must contain finite values")


@runtime_checkable
class RgbdImuSource(Protocol):
    """Interface implemented by OAK-D and deterministic test capture sources."""

    @property
    def source_name(self) -> str:
        """Return a stable source identifier for the replay manifest."""

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """Return calibration for the aligned RGB/depth stream."""

    @property
    def metadata(self) -> CaptureSourceMetadata:
        """Return serializable device and stream metadata."""

    def __iter__(self) -> Iterator[RgbdCaptureFrame]:
        """Yield chronological synchronized frames until capture stops."""

    def close(self) -> None:
        """Release the device or underlying data source."""
