"""Associate tracked image regions with robust depth and 3D camera positions."""

from __future__ import annotations

from enum import StrEnum
from math import ceil, floor

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sightly_assist.perception import FramePacket, TrackObservation


class DepthStatus(StrEnum):
    """Outcome of depth association for one tracked object."""

    VALID = "valid"
    NO_DEPTH = "no_depth"
    INSUFFICIENT_SAMPLES = "insufficient_samples"


class CameraIntrinsics(BaseModel):
    """Pinhole camera calibration in pixel units."""

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fx_px: float = Field(gt=0)
    fy_px: float = Field(gt=0)
    cx_px: float = Field(ge=0)
    cy_px: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_principal_point(self) -> CameraIntrinsics:
        if self.cx_px >= self.width_px:
            raise ValueError("cx_px must lie inside the image")
        if self.cy_px >= self.height_px:
            raise ValueError("cy_px must lie inside the image")
        return self


class CameraPoint(BaseModel):
    """Point in a right/down/forward camera coordinate frame."""

    x_m: float
    y_m: float
    z_m: float = Field(gt=0)


class DepthAssociationConfig(BaseModel):
    """Controls robust depth sampling inside a tracked bounding box."""

    inner_box_fraction: float = Field(gt=0, le=1, default=0.5)
    minimum_depth_m: float = Field(ge=0, default=0.15)
    maximum_depth_m: float = Field(gt=0, default=20.0)
    minimum_valid_samples: int = Field(gt=0, default=4)
    depth_scale_m: float = Field(gt=0, default=1.0)
    mad_multiplier: float = Field(gt=0, default=3.5)
    minimum_outlier_band_m: float = Field(gt=0, default=0.05)

    @model_validator(mode="after")
    def validate_depth_range(self) -> DepthAssociationConfig:
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum_depth_m must exceed minimum_depth_m")
        return self


class DepthAssociation(BaseModel):
    """Depth estimate and uncertainty for one tracked object."""

    track_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    status: DepthStatus
    roi_sample_count: int = Field(ge=0)
    valid_sample_count: int = Field(ge=0)
    inlier_sample_count: int = Field(ge=0)
    valid_fraction: float = Field(ge=0, le=1)
    depth_m: float | None = Field(default=None, gt=0)
    median_absolute_deviation_m: float | None = Field(default=None, ge=0)
    pixel_u: float | None = None
    pixel_v: float | None = None
    camera_point: CameraPoint | None = None


def associate_track_depth(
    depth_map: np.ndarray | None,
    frame: FramePacket,
    track: TrackObservation,
    intrinsics: CameraIntrinsics,
    config: DepthAssociationConfig | None = None,
) -> DepthAssociation:
    """Estimate robust object depth and deproject its image-box center."""

    settings = config or DepthAssociationConfig()
    _validate_inputs(depth_map, frame, intrinsics)
    if depth_map is None:
        return _empty_result(track, DepthStatus.NO_DEPTH)

    left, top, right, bottom = _inner_roi_bounds(frame, track, settings.inner_box_fraction)
    roi = np.asarray(depth_map[top:bottom, left:right], dtype=np.float64)
    metric = roi * settings.depth_scale_m
    valid_mask = (
        np.isfinite(metric)
        & (metric >= settings.minimum_depth_m)
        & (metric <= settings.maximum_depth_m)
    )
    valid_values = metric[valid_mask]
    roi_count = int(metric.size)
    valid_count = int(valid_values.size)
    valid_fraction = valid_count / roi_count if roi_count else 0.0

    if valid_count < settings.minimum_valid_samples:
        return DepthAssociation(
            track_id=track.track_id,
            frame_id=frame.frame_id,
            status=DepthStatus.INSUFFICIENT_SAMPLES,
            roi_sample_count=roi_count,
            valid_sample_count=valid_count,
            inlier_sample_count=0,
            valid_fraction=valid_fraction,
        )

    initial_median = float(np.median(valid_values))
    deviations = np.abs(valid_values - initial_median)
    initial_mad = float(np.median(deviations))
    robust_sigma = 1.4826 * initial_mad
    outlier_band = max(
        settings.minimum_outlier_band_m,
        settings.mad_multiplier * robust_sigma,
    )
    inliers = valid_values[deviations <= outlier_band]
    if inliers.size < settings.minimum_valid_samples:
        inliers = valid_values

    depth_m = float(np.median(inliers))
    mad_m = float(np.median(np.abs(inliers - depth_m)))
    pixel_u = min(
        max((track.bbox.x_min + track.bbox.x_max) / 2.0, 0.0),
        frame.width_px - 1.0,
    )
    pixel_v = min(
        max((track.bbox.y_min + track.bbox.y_max) / 2.0, 0.0),
        frame.height_px - 1.0,
    )
    point = deproject_pixel(pixel_u, pixel_v, depth_m, intrinsics)

    return DepthAssociation(
        track_id=track.track_id,
        frame_id=frame.frame_id,
        status=DepthStatus.VALID,
        roi_sample_count=roi_count,
        valid_sample_count=valid_count,
        inlier_sample_count=int(inliers.size),
        valid_fraction=valid_fraction,
        depth_m=depth_m,
        median_absolute_deviation_m=mad_m,
        pixel_u=pixel_u,
        pixel_v=pixel_v,
        camera_point=point,
    )


def associate_tracks_depth(
    depth_map: np.ndarray | None,
    frame: FramePacket,
    tracks: tuple[TrackObservation, ...],
    intrinsics: CameraIntrinsics,
    config: DepthAssociationConfig | None = None,
) -> tuple[DepthAssociation, ...]:
    """Associate all tracks in one frame with the same synchronized depth map."""

    return tuple(
        associate_track_depth(depth_map, frame, track, intrinsics, config) for track in tracks
    )


def deproject_pixel(
    pixel_u: float,
    pixel_v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> CameraPoint:
    """Convert one image pixel and forward depth into camera coordinates."""

    if depth_m <= 0 or not np.isfinite(depth_m):
        raise ValueError("depth_m must be finite and positive")
    x_m = (pixel_u - intrinsics.cx_px) * depth_m / intrinsics.fx_px
    y_m = (pixel_v - intrinsics.cy_px) * depth_m / intrinsics.fy_px
    return CameraPoint(x_m=x_m, y_m=y_m, z_m=depth_m)


def _validate_inputs(
    depth_map: np.ndarray | None,
    frame: FramePacket,
    intrinsics: CameraIntrinsics,
) -> None:
    if intrinsics.width_px != frame.width_px or intrinsics.height_px != frame.height_px:
        raise ValueError("Camera intrinsics dimensions must match the replay frame")
    if depth_map is None:
        return
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be two-dimensional")
    if depth_map.shape != (frame.height_px, frame.width_px):
        raise ValueError("depth_map dimensions must match the replay frame")
    if not np.issubdtype(depth_map.dtype, np.number):
        raise ValueError("depth_map must have a numeric data type")


def _inner_roi_bounds(
    frame: FramePacket,
    track: TrackObservation,
    fraction: float,
) -> tuple[int, int, int, int]:
    center_x = (track.bbox.x_min + track.bbox.x_max) / 2.0
    center_y = (track.bbox.y_min + track.bbox.y_max) / 2.0
    half_width = track.bbox.width * fraction / 2.0
    half_height = track.bbox.height * fraction / 2.0
    left = max(0, floor(center_x - half_width))
    top = max(0, floor(center_y - half_height))
    right = min(frame.width_px, ceil(center_x + half_width))
    bottom = min(frame.height_px, ceil(center_y + half_height))
    right = max(right, min(left + 1, frame.width_px))
    bottom = max(bottom, min(top + 1, frame.height_px))
    return left, top, right, bottom


def _empty_result(track: TrackObservation, status: DepthStatus) -> DepthAssociation:
    return DepthAssociation(
        track_id=track.track_id,
        frame_id=track.frame_id,
        status=status,
        roi_sample_count=0,
        valid_sample_count=0,
        inlier_sample_count=0,
        valid_fraction=0.0,
    )
