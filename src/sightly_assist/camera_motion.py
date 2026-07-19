"""Camera-orientation models and rotation-only 3D stabilization."""

from __future__ import annotations

from math import isfinite, sqrt

import numpy as np
from pydantic import BaseModel, model_validator

from sightly_assist.depth_association import (
    CameraPoint,
    DepthAssociation,
    DepthStatus,
)
from sightly_assist.motion_estimation import (
    MotionEstimate,
    MotionEstimatorConfig,
    TrackMotionEstimator,
)
from sightly_assist.perception import FramePacket


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
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
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
    reference_orientation: CameraOrientation | None = None,
) -> StabilizedPoint:
    """Rotate a camera point into reference axes without removing translation."""

    current_world_from_camera = orientation.rotation_matrix()
    if reference_orientation is None:
        reference_from_current = current_world_from_camera
    else:
        reference_from_current = (
            reference_orientation.rotation_matrix().T @ current_world_from_camera
        )
    vector = np.asarray([point.x_m, point.y_m, point.z_m], dtype=np.float64)
    stabilized = reference_from_current @ vector
    return StabilizedPoint(
        x_m=float(stabilized[0]),
        y_m=float(stabilized[1]),
        z_m=float(stabilized[2]),
    )


class RotationCompensatedMotionEstimator(TrackMotionEstimator):
    """Apply frame orientation before robust relative-motion estimation.

    Rotation is removed in axes fixed to the first valid orientation. Camera
    translation is intentionally preserved because it contributes to relative
    closing motion. The orientation must represent world-from-camera rotation.
    """

    def __init__(
        self,
        config: MotionEstimatorConfig | None = None,
        *,
        require_orientation: bool = False,
    ) -> None:
        super().__init__(config)
        self.require_orientation = require_orientation
        self._reference_orientation: CameraOrientation | None = None

    def update(
        self,
        frame: FramePacket,
        associations: tuple[DepthAssociation, ...],
    ) -> tuple[MotionEstimate, ...]:
        """Stabilize valid 3D points and run the inherited temporal estimator."""

        if frame.orientation_wxyz is None:
            if self.require_orientation:
                unavailable = tuple(
                    _remove_unstabilized_point(association) for association in associations
                )
                return super().update(frame, unavailable)
            return super().update(frame, associations)

        orientation = CameraOrientation.from_wxyz(frame.orientation_wxyz)
        if self._reference_orientation is None:
            self._reference_orientation = orientation

        stabilized = tuple(
            _stabilize_association(
                association,
                orientation,
                self._reference_orientation,
            )
            for association in associations
        )
        return super().update(frame, stabilized)


def _stabilize_association(
    association: DepthAssociation,
    orientation: CameraOrientation,
    reference_orientation: CameraOrientation,
) -> DepthAssociation:
    if association.status is not DepthStatus.VALID or association.camera_point is None:
        return association

    stabilized = stabilize_camera_point(
        association.camera_point,
        orientation,
        reference_orientation,
    )
    if stabilized.z_m <= 1e-6:
        return _remove_unstabilized_point(association)

    return association.model_copy(
        update={
            "camera_point": CameraPoint(
                x_m=stabilized.x_m,
                y_m=stabilized.y_m,
                z_m=stabilized.z_m,
            )
        }
    )


def _remove_unstabilized_point(association: DepthAssociation) -> DepthAssociation:
    if association.status is not DepthStatus.VALID:
        return association
    return association.model_copy(
        update={
            "status": DepthStatus.NO_DEPTH,
            "depth_m": None,
            "camera_point": None,
        }
    )
