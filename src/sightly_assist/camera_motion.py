"""Camera-orientation models and rotation-only 3D stabilization."""

from __future__ import annotations

from math import isfinite, sqrt

import numpy as np
from pydantic import BaseModel, model_validator

from sightly_assist.depth_association import CameraPoint


class CameraOrientation(BaseModel):
    """World-from-camera unit quaternion in w, x, y, z order."""

    w: float
    x: float
    y: float
    z: float

    @model_validator(mode="after")
    def validate_quaternion(self) -> CameraOrientation:
        values = (self.w, self.x, self.y, self.z)
        if not all(isfinite(value) for value in values):
            raise ValueError("camera orientation must contain finite values")
        if self.norm <= 1e-12:
            raise ValueError("camera orientation quaternion cannot be zero")
        return self

    @property
    def norm(self) -> float:
        """Return quaternion magnitude."""

        return sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    @classmethod
    def from_wxyz(
        cls,
        values: tuple[float, float, float, float],
    ) -> CameraOrientation:
        """Construct an orientation from a replay-frame tuple."""

        return cls(w=values[0], x=values[1], y=values[2], z=values[3])

    def rotation_matrix(self) -> np.ndarray:
        """Return the normalized world-from-camera rotation matrix."""

        norm = self.norm
        w = self.w / norm
        x = self.x / norm
        y = self.y / norm
        z = self.z / norm
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )


class StabilizedPoint(BaseModel):
    """3D point in orientation-stabilized right/down/forward axes."""

    x_m: float
    y_m: float
    z_m: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> StabilizedPoint:
        if not all(isfinite(value) for value in (self.x_m, self.y_m, self.z_m)):
            raise ValueError("stabilized coordinates must be finite")
        return self


def stabilize_camera_point(
    point: CameraPoint,
    orientation: CameraOrientation,
) -> StabilizedPoint:
    """Rotate a camera-frame point into stable axes without removing translation."""

    vector = np.asarray([point.x_m, point.y_m, point.z_m], dtype=np.float64)
    stabilized = orientation.rotation_matrix() @ vector
    return StabilizedPoint(
        x_m=float(stabilized[0]),
        y_m=float(stabilized[1]),
        z_m=float(stabilized[2]),
    )
