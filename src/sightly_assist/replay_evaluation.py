"""Run, measure, and export complete recorded-sequence evaluations."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from sightly_assist.annotated_video import export_annotated_video
from sightly_assist.audio_cues import export_warning_audio_cues
from sightly_assist.bytetrack_adapter import ByteTrackAdapter
from sightly_assist.camera_motion import RotationCompensatedMotionEstimator
from sightly_assist.dataset_recorder import verify_dataset_checksums
from sightly_assist.depth_association import DepthStatus
from sightly_assist.free_space import FreeSpaceConfig
from sightly_assist.motion_estimation import MotionStatus
from sightly_assist.onnx_detector import OnnxYoloDetector
from sightly_assist.opencv_io import load_depth, load_rgb
from sightly_assist.perception import MultiObjectTracker, ObjectDetector
from sightly_assist.perception_risk import PerceptionRiskConfig, PerceptionRiskStatus
from sightly_assist.replay import ReplayManifest, load_manifest, validate_replay_files
from sightly_assist.replay_pipeline import (
    DepthLoader,
    ImageLoader,
    ReplayResult,
    ReplayStageTimings,
    process_replay,
)
from sightly_assist.warning_policy import AlertAction, WarningPolicy


class ReplayEvaluationConfig(BaseModel):
    """Integrity, export, and motion-stabilization controls."""

    verify_checksums: bool = True
    render_video: bool = True
    render_audio: bool = True
    require_orientation: bool = False
    overwrite: bool = False


class LatencyDistribution(BaseModel):
    """Summary statistics for one measured processing stage."""

    mean_ms: float = Field(ge=0)
    median_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    maximum_ms: float = Field(ge=0)


class ReplayEvaluationSummary(BaseModel):
    """Machine-readable summary for one complete replay evaluation."""

    sequence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    frame_count: int = Field(gt=0)
    recording_duration_s: float = Field(ge=0)
    detection_count: int = Field(ge=0)
    track_observation_count: int = Field(ge=0)
    valid_depth_association_count: int = Field(ge=0)
    valid_motion_estimate_count: int = Field(ge=0)
    valid_risk_result_count: int = Field(ge=0)
    predicted_collision_frame_count: int = Field(ge=0)
    emitted_warning_count: int = Field(ge=0)
    warning_action_counts: dict[str, int]
    center_free_space_counts: dict[str, int]
    synchronized_frame_count: int = Field(ge=0)
    mean_processing_fps: float = Field(ge=0)
    realtime_factor: float | None = Field(default=None, ge=0)
    stage_latencies: dict[str, LatencyDistribution]
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    results_path: str
    summary_path: str
    annotated_video_path: str | None = None
    audio_directory: str | None = None


DetectorFactory = Callable[[], ObjectDetector]
TrackerFactory = Callable[[], MultiObjectTracker]


def run_onnx_replay_evaluation(
    manifest_path: Path,
    model_path: Path,
    output_directory: Path,
    config: ReplayEvaluationConfig | None = None,
    *,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> ReplayEvaluationSummary:
    """Load a dataset and ONNX model, then run the complete evaluation stack."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Replay manifest does not exist: {manifest_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    manifest = load_manifest(manifest_path)
    detector = OnnxYoloDetector(model_path=model_path, providers=providers)
    return evaluate_replay(
        manifest,
        manifest_path.parent,
        detector,
        ByteTrackAdapter(),
        output_directory,
        config,
        model_sha256=_sha256(model_path),
    )


def evaluate_replay(
    manifest: ReplayManifest,
    root: Path,
    detector: ObjectDetector,
    tracker: MultiObjectTracker,
    output_directory: Path,
    config: ReplayEvaluationConfig | None = None,
    *,
    model_sha256: str | None = None,
    image_loader: ImageLoader = load_rgb,
    depth_loader: DepthLoader = load_depth,
) -> ReplayEvaluationSummary:
    """Evaluate injected detector/tracker implementations on one recorded sequence."""

    settings = config or ReplayEvaluationConfig()
    if manifest.intrinsics is None:
        raise ValueError("Replay evaluation requires camera intrinsics in the manifest")
    missing = validate_replay_files(manifest, root)
    if missing:
        raise FileNotFoundError(f"Replay dataset is missing referenced files: {missing}")
    if settings.verify_checksums:
        mismatches = verify_dataset_checksums(manifest, root)
        if mismatches:
            raise ValueError(f"Replay dataset checksum mismatch: {mismatches}")
    if model_sha256 is not None and len(model_sha256) != 64:
        raise ValueError("model_sha256 must contain 64 lowercase hexadecimal characters")

    destination = output_directory.resolve()
    if destination.exists():
        if not settings.overwrite:
            raise FileExistsError(f"Evaluation output already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    results = tuple(
        process_replay(
            manifest,
            root,
            detector,
            tracker,
            image_loader=image_loader,
            depth_loader=depth_loader,
            camera_intrinsics=manifest.intrinsics,
            free_space_config=FreeSpaceConfig(),
            motion_estimator=RotationCompensatedMotionEstimator(
                require_orientation=settings.require_orientation
            ),
            risk_config=PerceptionRiskConfig(),
            warning_policy=WarningPolicy(),
        )
    )
    if len(results) != len(manifest.frames):
        raise RuntimeError("Replay processing did not produce exactly one result per frame")

    results_path = destination / "frame_results.jsonl"
    with results_path.open("w", encoding="utf-8", newline="\n") as output:
        for result in results:
            output.write(json.dumps(_serialize_result(result), sort_keys=True) + "\n")

    video_path: Path | None = None
    if settings.render_video:
        video_path = destination / "annotated.mp4"
        export_annotated_video(manifest, results, root, video_path)

    audio_directory: Path | None = None
    if settings.render_audio:
        audio_directory = destination / "audio"
        decisions = tuple(
            result.warning_decision
            for result in results
            if result.warning_decision is not None
        )
        export_warning_audio_cues(decisions, audio_directory)

    summary_path = destination / "summary.json"
    summary = _summarize(
        manifest,
        results,
        destination,
        results_path,
        summary_path,
        video_path,
        audio_directory,
        model_sha256,
    )
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _summarize(
    manifest: ReplayManifest,
    results: tuple[ReplayResult, ...],
    destination: Path,
    results_path: Path,
    summary_path: Path,
    video_path: Path | None,
    audio_directory: Path | None,
    model_sha256: str | None,
) -> ReplayEvaluationSummary:
    timings = tuple(
        result.stage_timings for result in results if result.stage_timings is not None
    )
    if len(timings) != len(results):
        raise RuntimeError("Every evaluated replay frame must contain stage timings")

    warning_counts = {action.value: 0 for action in AlertAction}
    center_counts: dict[str, int] = {}
    emitted_warning_count = 0
    predicted_collision_frames = 0
    for result in results:
        if result.warning_decision is not None:
            action = result.warning_decision.action.value
            warning_counts[action] += 1
            emitted_warning_count += int(result.warning_decision.should_emit)
        center = result.free_space_analysis
        if center is not None:
            state = center.center.state.value
            center_counts[state] = center_counts.get(state, 0) + 1
        if any(
            risk.status is PerceptionRiskStatus.VALID
            and risk.assessment is not None
            and risk.assessment.predicted_collision
            for risk in result.risk_results
        ):
            predicted_collision_frames += 1

    first_timestamp = manifest.frames[0].timestamp_ns
    last_timestamp = manifest.frames[-1].timestamp_ns
    recording_duration_s = max(0.0, (last_timestamp - first_timestamp) / 1_000_000_000)
    total_mean_ms = float(np.mean([timing.total_ms for timing in timings]))
    mean_processing_fps = 1000.0 / total_mean_ms if total_mean_ms > 0 else 0.0
    processing_duration_s = sum(timing.total_ms for timing in timings) / 1000.0
    realtime_factor = (
        processing_duration_s / recording_duration_s if recording_duration_s > 0 else None
    )

    return ReplayEvaluationSummary(
        sequence_id=manifest.sequence_id,
        source=manifest.source,
        frame_count=len(results),
        recording_duration_s=recording_duration_s,
        detection_count=sum(len(result.detections) for result in results),
        track_observation_count=sum(len(result.tracks) for result in results),
        valid_depth_association_count=sum(
            association.status is DepthStatus.VALID
            for result in results
            for association in result.depth_associations
        ),
        valid_motion_estimate_count=sum(
            motion.status is MotionStatus.VALID
            for result in results
            for motion in result.motion_estimates
        ),
        valid_risk_result_count=sum(
            risk.status is PerceptionRiskStatus.VALID
            for result in results
            for risk in result.risk_results
        ),
        predicted_collision_frame_count=predicted_collision_frames,
        emitted_warning_count=emitted_warning_count,
        warning_action_counts=warning_counts,
        center_free_space_counts=center_counts,
        synchronized_frame_count=sum(frame.synchronized for frame in manifest.frames),
        mean_processing_fps=mean_processing_fps,
        realtime_factor=realtime_factor,
        stage_latencies=_latency_summary(timings),
        model_sha256=model_sha256,
        results_path=str(results_path),
        summary_path=str(summary_path),
        annotated_video_path=str(video_path) if video_path is not None else None,
        audio_directory=str(audio_directory) if audio_directory is not None else None,
    )


def _latency_summary(
    timings: tuple[ReplayStageTimings, ...],
) -> dict[str, LatencyDistribution]:
    fields = tuple(ReplayStageTimings.model_fields)
    return {
        field: _distribution(np.asarray([getattr(item, field) for item in timings]))
        for field in fields
    }


def _distribution(values: np.ndarray) -> LatencyDistribution:
    if values.size == 0:
        return LatencyDistribution(mean_ms=0, median_ms=0, p95_ms=0, maximum_ms=0)
    return LatencyDistribution(
        mean_ms=float(np.mean(values)),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        maximum_ms=float(np.max(values)),
    )


def _serialize_result(result: ReplayResult) -> dict[str, object]:
    return {
        "frame": result.frame.model_dump(mode="json"),
        "detections": [item.model_dump(mode="json") for item in result.detections],
        "tracks": [item.model_dump(mode="json") for item in result.tracks],
        "depth_associations": [
            item.model_dump(mode="json") for item in result.depth_associations
        ],
        "free_space_analysis": (
            result.free_space_analysis.model_dump(mode="json")
            if result.free_space_analysis is not None
            else None
        ),
        "motion_estimates": [
            item.model_dump(mode="json") for item in result.motion_estimates
        ],
        "risk_results": [item.model_dump(mode="json") for item in result.risk_results],
        "warning_decision": (
            result.warning_decision.model_dump(mode="json")
            if result.warning_decision is not None
            else None
        ),
        "stage_timings": (
            result.stage_timings.model_dump(mode="json")
            if result.stage_timings is not None
            else None
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
