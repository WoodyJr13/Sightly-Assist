"""Conservative coarse free-space analysis from a synchronized depth image."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.perception import FramePacket


class CorridorDirection(StrEnum):
    """Coarse image-relative corridor direction."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class CorridorStatus(StrEnum):
    """Conservative clearance state for one corridor."""

    CLEAR = "clear"
    CONSTRAINED = "constrained"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class FreeSpaceContext(StrEnum):
    """Center-path context; this is not a steering instruction."""

    CENTER_CLEAR = "center_clear"
    CENTER_CONSTRAINED = "center_constrained"
    CENTER_BLOCKED = "center_blocked"
    UNKNOWN = "unknown"


class FreeSpaceConfig(BaseModel):
    """Controls depth coverage, corridor geometry, and clearance thresholds."""

    vertical_start_fraction: float = Field(ge=0, lt=1, default=0.45)
    vertical_end_fraction: float = Field(gt=0, le=1, default=0.95)
    reference_depth_m: float = Field(gt=0, default=2.0)
    center_half_width_m: float = Field(gt=0, default=0.40)
    analysis_half_width_m: float = Field(gt=0, default=1.20)
    minimum_depth_m: float = Field(ge=0, default=0.15)
    maximum_depth_m: float = Field(gt=0, default=8.0)
    depth_scale_m: float = Field(gt=0, default=1.0)
    blocked_depth_m: float = Field(gt=0, default=1.50)
    clear_depth_m: float = Field(gt=0, default=3.00)
    near_percentile: float = Field(gt=0, lt=50, default=10.0)
    minimum_valid_samples: int = Field(gt=0, default=40)
    minimum_valid_fraction: float = Field(gt=0, le=1, default=0.20)
    constrained_obstacle_fraction: float = Field(ge=0, le=1, default=0.03)
    blocked_obstacle_fraction: float = Field(gt=0, le=1, default=0.10)

    @model_validator(mode="after")
    def validate_ranges(self) -> FreeSpaceConfig:
        if self.vertical_end_fraction <= self.vertical_start_fraction:
            raise ValueError("vertical_end_fraction must exceed vertical_start_fraction")
        if self.analysis_half_width_m <= self.center_half_width_m:
            raise ValueError("analysis_half_width_m must exceed center_half_width_m")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum_depth_m must exceed minimum_depth_m")
        if self.clear_depth_m <= self.blocked_depth_m:
            raise ValueError("clear_depth_m must exceed blocked_depth_m")
        if self.blocked_obstacle_fraction <= self.constrained_obstacle_fraction:
            raise ValueError("blocked_obstacle_fraction must exceed constrained_obstacle_fraction")
        return self


class CorridorAssessment(BaseModel):
    """Coverage and clearance statistics for one coarse corridor."""

    direction: CorridorDirection
    status: CorridorStatus
    pixel_count: int = Field(ge=0)
    valid_sample_count: int = Field(ge=0)
    valid_fraction: float = Field(ge=0, le=1)
    near_depth_m: float | None = Field(default=None, gt=0)
    median_depth_m: float | None = Field(default=None, gt=0)
    obstacle_fraction: float | None = Field(default=None, ge=0, le=1)


class FreeSpaceAnalysis(BaseModel):
    """Coarse free-space context for one frame."""

    frame_id: int = Field(ge=0)
    context: FreeSpaceContext
    row_start: int = Field(ge=0)
    row_end: int = Field(gt=0)
    corridors: tuple[CorridorAssessment, CorridorAssessment, CorridorAssessment]
    reason: str = Field(min_length=1)

    def corridor(self, direction: CorridorDirection) -> CorridorAssessment:
        """Return one named corridor assessment."""

        for assessment in self.corridors:
            if assessment.direction is direction:
                return assessment
        raise RuntimeError(f"Missing corridor assessment for {direction.value}")


def analyze_free_space(
    depth_map: np.ndarray | None,
    frame: FramePacket,
    intrinsics: CameraIntrinsics,
    config: FreeSpaceConfig | None = None,
) -> FreeSpaceAnalysis:
    """Assess left, center, and right depth corridors without steering advice."""

    settings = config or FreeSpaceConfig()
    _validate_inputs(depth_map, frame, intrinsics)
    row_start = int(np.floor(frame.height_px * settings.vertical_start_fraction))
    row_end = int(np.ceil(frame.height_px * settings.vertical_end_fraction))
    row_start = min(max(row_start, 0), frame.height_px - 1)
    row_end = min(max(row_end, row_start + 1), frame.height_px)

    masks = _corridor_masks(frame, intrinsics, settings, row_start, row_end)
    if depth_map is None:
        corridors = tuple(
            _unknown_assessment(direction, int(mask.sum())) for direction, mask in masks
        )
        typed_corridors = _as_corridor_tuple(corridors)
        return FreeSpaceAnalysis(
            frame_id=frame.frame_id,
            context=FreeSpaceContext.UNKNOWN,
            row_start=row_start,
            row_end=row_end,
            corridors=typed_corridors,
            reason="no synchronized depth map",
        )

    metric_depth = np.asarray(depth_map[row_start:row_end], dtype=np.float64)
    metric_depth *= settings.depth_scale_m
    assessments = tuple(
        _assess_corridor(metric_depth, direction, mask, settings) for direction, mask in masks
    )
    typed_assessments = _as_corridor_tuple(assessments)
    center = next(
        assessment
        for assessment in typed_assessments
        if assessment.direction is CorridorDirection.CENTER
    )
    context, reason = _center_context(center)
    return FreeSpaceAnalysis(
        frame_id=frame.frame_id,
        context=context,
        row_start=row_start,
        row_end=row_end,
        corridors=typed_assessments,
        reason=reason,
    )


def _corridor_masks(
    frame: FramePacket,
    intrinsics: CameraIntrinsics,
    config: FreeSpaceConfig,
    row_start: int,
    row_end: int,
) -> tuple[
    tuple[CorridorDirection, np.ndarray],
    tuple[CorridorDirection, np.ndarray],
    tuple[CorridorDirection, np.ndarray],
]:
    columns = np.arange(frame.width_px, dtype=np.float64)
    reference_x_m = (columns - intrinsics.cx_px) * config.reference_depth_m / intrinsics.fx_px
    row_count = row_end - row_start

    left_columns = (reference_x_m >= -config.analysis_half_width_m) & (
        reference_x_m < -config.center_half_width_m
    )
    center_columns = np.abs(reference_x_m) <= config.center_half_width_m
    right_columns = (reference_x_m > config.center_half_width_m) & (
        reference_x_m <= config.analysis_half_width_m
    )

    return (
        (CorridorDirection.LEFT, np.broadcast_to(left_columns, (row_count, frame.width_px))),
        (
            CorridorDirection.CENTER,
            np.broadcast_to(center_columns, (row_count, frame.width_px)),
        ),
        (
            CorridorDirection.RIGHT,
            np.broadcast_to(right_columns, (row_count, frame.width_px)),
        ),
    )


def _assess_corridor(
    metric_depth: np.ndarray,
    direction: CorridorDirection,
    mask: np.ndarray,
    config: FreeSpaceConfig,
) -> CorridorAssessment:
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return _unknown_assessment(direction, 0)

    values = metric_depth[mask]
    valid_mask = (
        np.isfinite(values)
        & (values >= config.minimum_depth_m)
        & (values <= config.maximum_depth_m)
    )
    valid_values = values[valid_mask]
    valid_count = int(valid_values.size)
    valid_fraction = valid_count / pixel_count

    if valid_count < config.minimum_valid_samples or valid_fraction < config.minimum_valid_fraction:
        return CorridorAssessment(
            direction=direction,
            status=CorridorStatus.UNKNOWN,
            pixel_count=pixel_count,
            valid_sample_count=valid_count,
            valid_fraction=valid_fraction,
        )

    near_depth_m = float(np.percentile(valid_values, config.near_percentile))
    median_depth_m = float(np.median(valid_values))
    obstacle_fraction = float(np.mean(valid_values <= config.blocked_depth_m))

    if (
        near_depth_m <= config.blocked_depth_m
        and obstacle_fraction >= config.blocked_obstacle_fraction
    ):
        status = CorridorStatus.BLOCKED
    elif (
        near_depth_m < config.clear_depth_m
        or obstacle_fraction >= config.constrained_obstacle_fraction
    ):
        status = CorridorStatus.CONSTRAINED
    else:
        status = CorridorStatus.CLEAR

    return CorridorAssessment(
        direction=direction,
        status=status,
        pixel_count=pixel_count,
        valid_sample_count=valid_count,
        valid_fraction=valid_fraction,
        near_depth_m=near_depth_m,
        median_depth_m=median_depth_m,
        obstacle_fraction=obstacle_fraction,
    )


def _unknown_assessment(
    direction: CorridorDirection,
    pixel_count: int,
) -> CorridorAssessment:
    return CorridorAssessment(
        direction=direction,
        status=CorridorStatus.UNKNOWN,
        pixel_count=pixel_count,
        valid_sample_count=0,
        valid_fraction=0.0,
    )


def _center_context(
    center: CorridorAssessment,
) -> tuple[FreeSpaceContext, str]:
    if center.status is CorridorStatus.CLEAR:
        return FreeSpaceContext.CENTER_CLEAR, "center corridor meets clearance thresholds"
    if center.status is CorridorStatus.CONSTRAINED:
        return FreeSpaceContext.CENTER_CONSTRAINED, "center corridor has limited clearance"
    if center.status is CorridorStatus.BLOCKED:
        return FreeSpaceContext.CENTER_BLOCKED, "center corridor contains a near obstruction"
    return FreeSpaceContext.UNKNOWN, "center corridor has insufficient trustworthy depth"


def _as_corridor_tuple(
    corridors: tuple[CorridorAssessment, ...],
) -> tuple[CorridorAssessment, CorridorAssessment, CorridorAssessment]:
    if len(corridors) != 3:
        raise RuntimeError("free-space analysis requires exactly three corridors")
    return corridors[0], corridors[1], corridors[2]


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
