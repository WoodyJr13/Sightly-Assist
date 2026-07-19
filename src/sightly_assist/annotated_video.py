"""Render replay perception, motion, risk, and warning outputs onto video frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from sightly_assist.opencv_io import VisionDependencyError, load_rgb
from sightly_assist.perception import FramePacket
from sightly_assist.replay import ReplayManifest, iter_frames
from sightly_assist.replay_pipeline import ReplayResult
from sightly_assist.warning_policy import AlertAction


class AnnotationConfig(BaseModel):
    """Visual and encoding settings for annotated replay exports."""

    font_scale: float = Field(gt=0, default=0.45)
    line_thickness: int = Field(gt=0, default=1)
    box_thickness: int = Field(gt=0, default=2)
    text_line_height_px: int = Field(gt=0, default=17)
    banner_height_px: int = Field(gt=0, default=34)
    fallback_fps: float = Field(gt=0, default=30.0)
    minimum_fps: float = Field(gt=0, default=1.0)
    maximum_fps: float = Field(gt=0, default=120.0)


class VideoExportSummary(BaseModel):
    """Metadata describing one completed annotated video export."""

    output_path: str
    frame_count: int = Field(ge=0)
    fps: float = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    codec: str = Field(min_length=1)


class VideoWriter(Protocol):
    """Minimal interface implemented by OpenCV and test video writers."""

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        """Return whether the writer opened successfully."""

    def write(self, image: np.ndarray) -> None:
        """Write one BGR frame."""

    def release(self) -> None:
        """Close the writer and flush pending frames."""


ImageLoader = Callable[[FramePacket, Path], np.ndarray]
WriterFactory = Callable[[Path, float, tuple[int, int]], tuple[VideoWriter, str]]


def annotate_frame(
    image_bgr: np.ndarray,
    result: ReplayResult,
    config: AnnotationConfig | None = None,
) -> np.ndarray:
    """Return a copy of one BGR frame with pipeline diagnostics overlaid."""

    settings = config or AnnotationConfig()
    _validate_frame(image_bgr, result)
    cv2 = _import_cv2()
    annotated = np.asarray(image_bgr).copy()

    depths = {item.track_id: item for item in result.depth_associations}
    motions = {item.track_id: item for item in result.motion_estimates}
    risks = {item.track_id: item for item in result.risk_results}

    for track in result.tracks:
        risk = risks.get(track.track_id)
        color = _track_color(risk)
        left = int(max(0, min(round(track.bbox.x_min), result.frame.width_px - 1)))
        top = int(max(0, min(round(track.bbox.y_min), result.frame.height_px - 1)))
        right = int(max(0, min(round(track.bbox.x_max), result.frame.width_px - 1)))
        bottom = int(max(0, min(round(track.bbox.y_max), result.frame.height_px - 1)))
        cv2.rectangle(annotated, (left, top), (right, bottom), color, settings.box_thickness)

        lines = [f"{track.class_name} #{track.track_id} {track.confidence:.2f}"]
        depth = depths.get(track.track_id)
        if depth is not None and depth.depth_m is not None:
            lines.append(f"depth {depth.depth_m:.2f} m")
        elif depth is not None:
            lines.append(f"depth {depth.status.value}")

        motion = motions.get(track.track_id)
        if motion is not None and motion.velocity is not None:
            lines.append(
                "vel " f"x={motion.velocity.x_mps:+.2f} " f"z={motion.velocity.z_mps:+.2f} m/s"
            )
        elif motion is not None:
            lines.append(f"motion {motion.status.value}")

        if risk is not None and risk.assessment is not None:
            assessment = risk.assessment
            lines.append(
                f"risk {assessment.risk_level.value} {assessment.risk_score:.2f} "
                f"CPA {assessment.time_to_closest_approach_s:.1f}s/"
                f"{assessment.distance_at_closest_approach_m:.2f}m"
            )
        elif risk is not None:
            lines.append(f"risk {risk.status.value}")

        _draw_label_block(
            annotated,
            lines,
            left,
            max(settings.banner_height_px, top),
            color,
            settings,
            cv2,
        )

    _draw_header(annotated, result, settings, cv2)
    return annotated


def infer_replay_fps(
    manifest: ReplayManifest,
    config: AnnotationConfig | None = None,
) -> float:
    """Infer a stable constant playback rate from replay timestamps."""

    settings = config or AnnotationConfig()
    frames = tuple(iter_frames(manifest))
    if len(frames) < 2:
        return settings.fallback_fps

    deltas_s = np.diff(np.array([item.timestamp_ns for item in frames], dtype=np.float64))
    deltas_s /= 1_000_000_000
    positive = deltas_s[deltas_s > 0]
    if positive.size == 0:
        return settings.fallback_fps
    fps = 1.0 / float(np.median(positive))
    return min(max(fps, settings.minimum_fps), settings.maximum_fps)


def export_annotated_video(
    manifest: ReplayManifest,
    results: Iterable[ReplayResult],
    root: Path,
    output_path: Path,
    *,
    image_loader: ImageLoader = load_rgb,
    writer_factory: WriterFactory | None = None,
    config: AnnotationConfig | None = None,
) -> VideoExportSummary:
    """Render chronological replay results to an annotated constant-rate video."""

    settings = config or AnnotationConfig()
    frames = tuple(iter_frames(manifest))
    if not frames:
        raise ValueError("Replay manifest must contain at least one frame")

    fps = infer_replay_fps(manifest, settings)
    dimensions = (frames[0].width_px, frames[0].height_px)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_factory = writer_factory or _open_cv_writer
    writer, codec = resolved_factory(output_path, fps, dimensions)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open annotated video writer for {output_path}")

    frame_count = 0
    try:
        sentinel = object()
        for frame_value, result_value in zip_longest(frames, results, fillvalue=sentinel):
            if frame_value is sentinel or result_value is sentinel:
                raise ValueError("Replay results count must match the manifest frame count")
            if not isinstance(frame_value, FramePacket):
                raise TypeError("Replay manifest must contain FramePacket values")
            if not isinstance(result_value, ReplayResult):
                raise TypeError("Replay results must contain ReplayResult values")
            if result_value.frame.frame_id != frame_value.frame_id:
                raise ValueError(
                    "Replay result frame order does not match manifest: "
                    f"expected {frame_value.frame_id}, got {result_value.frame.frame_id}"
                )
            image = image_loader(frame_value, root)
            annotated = annotate_frame(image, result_value, settings)
            writer.write(annotated)
            frame_count += 1
    finally:
        writer.release()

    return VideoExportSummary(
        output_path=str(output_path),
        frame_count=frame_count,
        fps=fps,
        width_px=dimensions[0],
        height_px=dimensions[1],
        codec=codec,
    )


def _draw_header(
    image: np.ndarray,
    result: ReplayResult,
    settings: AnnotationConfig,
    cv2: Any,
) -> None:
    height = min(settings.banner_height_px, image.shape[0])
    decision = result.warning_decision
    action = decision.action if decision is not None else AlertAction.NO_ALERT
    color = _action_color(action)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, height - 1), color, -1)

    timestamp_s = result.frame.timestamp_ns / 1_000_000_000
    label = f"frame {result.frame.frame_id}  t={timestamp_s:.2f}s  {action.value.upper()}"
    if decision is not None and decision.should_emit:
        label += "  EMIT"
    cv2.putText(
        image,
        label,
        (6, max(14, height - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        settings.font_scale,
        (255, 255, 255),
        settings.line_thickness,
        cv2.LINE_AA,
    )


def _draw_label_block(
    image: np.ndarray,
    lines: list[str],
    left: int,
    top: int,
    color: tuple[int, int, int],
    settings: AnnotationConfig,
    cv2: Any,
) -> None:
    if not lines:
        return
    text_width = max(
        cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.font_scale,
            settings.line_thickness,
        )[0][0]
        for line in lines
    )
    block_height = settings.text_line_height_px * len(lines) + 6
    box_left = max(0, min(left, image.shape[1] - 1))
    box_top = max(settings.banner_height_px, top - block_height)
    box_right = min(image.shape[1] - 1, box_left + text_width + 8)
    box_bottom = min(image.shape[0] - 1, box_top + block_height)
    cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), color, -1)

    for index, line in enumerate(lines):
        y = box_top + 15 + index * settings.text_line_height_px
        if y >= image.shape[0]:
            break
        cv2.putText(
            image,
            line,
            (box_left + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.font_scale,
            (255, 255, 255),
            settings.line_thickness,
            cv2.LINE_AA,
        )


def _track_color(risk: Any) -> tuple[int, int, int]:
    if risk is None or risk.assessment is None:
        return (128, 128, 128)
    score = risk.assessment.risk_score
    if risk.assessment.predicted_collision or score >= 0.8:
        return (0, 0, 220)
    if score >= 0.6:
        return (0, 120, 255)
    if score >= 0.35:
        return (0, 220, 220)
    return (0, 180, 0)


def _action_color(action: AlertAction) -> tuple[int, int, int]:
    if action is AlertAction.STOP:
        return (0, 0, 180)
    if action is AlertAction.SLOW:
        return (0, 100, 210)
    if action is AlertAction.ABSTAIN:
        return (100, 60, 100)
    if action is AlertAction.NO_ALERT:
        return (50, 50, 50)
    return (0, 130, 130)


def _validate_frame(image_bgr: np.ndarray, result: ReplayResult) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Annotated video input must be a three-channel BGR image")
    expected = (result.frame.height_px, result.frame.width_px)
    if image_bgr.shape[:2] != expected:
        raise ValueError(f"Annotated frame dimensions must match replay metadata: {expected}")
    if not np.issubdtype(image_bgr.dtype, np.number):
        raise ValueError("Annotated video input must use a numeric data type")


def _open_cv_writer(
    output_path: Path,
    fps: float,
    dimensions: tuple[int, int],
) -> tuple[VideoWriter, str]:
    cv2 = _import_cv2()
    suffix = output_path.suffix.lower()
    codecs = ("mp4v", "avc1", "MJPG") if suffix == ".mp4" else ("MJPG", "XVID", "mp4v")
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            dimensions,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"No supported video codec could open {output_path}")


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VisionDependencyError(
            'OpenCV is required for annotated video. Install the "vision" extra.'
        ) from exc
    return cv2
