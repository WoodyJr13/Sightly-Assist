"""Deterministic multi-object tracking and downstream collision-risk evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter_ns

import numpy as np
from pydantic import BaseModel, Field, model_validator

from sightly_assist.depth_association import CameraPoint, DepthAssociation, DepthStatus
from sightly_assist.motion_estimation import MotionEstimatorConfig, TrackMotionEstimator
from sightly_assist.perception import (
    BoundingBox,
    Detection,
    FramePacket,
    MultiObjectTracker,
    TrackObservation,
)
from sightly_assist.perception_risk import (
    PerceptionRiskConfig,
    PerceptionRiskStatus,
    assess_perception_risks,
)


class GroundTruthObservation(BaseModel):
    """One visible object's identity, box, 3D position, and risk label."""

    object_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    bbox: BoundingBox
    camera_point: CameraPoint | None = None
    expected_collision: bool = False


class TrackingBenchmarkFrame(BaseModel):
    """Detector input and ground truth for one chronological frame."""

    frame: FramePacket
    detections: tuple[Detection, ...]
    ground_truth: tuple[GroundTruthObservation, ...]

    @model_validator(mode="after")
    def validate_frame_membership(self) -> TrackingBenchmarkFrame:
        for detection in self.detections:
            if detection.frame_id != self.frame.frame_id:
                raise ValueError("Benchmark detection belongs to the wrong frame")
        object_ids: set[int] = set()
        for truth in self.ground_truth:
            if truth.frame_id != self.frame.frame_id:
                raise ValueError("Ground-truth observation belongs to the wrong frame")
            if truth.object_id in object_ids:
                raise ValueError("Ground-truth object IDs must be unique within a frame")
            object_ids.add(truth.object_id)
        return self


class TrackingBenchmarkSequence(BaseModel):
    """One locked chronological tracking scenario."""

    name: str = Field(min_length=1)
    frames: tuple[TrackingBenchmarkFrame, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chronology(self) -> TrackingBenchmarkSequence:
        previous_frame_id: int | None = None
        previous_timestamp_ns: int | None = None
        for item in self.frames:
            if previous_frame_id is not None and item.frame.frame_id <= previous_frame_id:
                raise ValueError("Benchmark frame IDs must be strictly increasing")
            if (
                previous_timestamp_ns is not None
                and item.frame.timestamp_ns <= previous_timestamp_ns
            ):
                raise ValueError("Benchmark timestamps must be strictly increasing")
            previous_frame_id = item.frame.frame_id
            previous_timestamp_ns = item.frame.timestamp_ns
        return self


class TrackingBenchmarkConfig(BaseModel):
    """Matching, motion, and collision-evaluation settings."""

    matching_iou_threshold: float = Field(gt=0, le=1, default=0.50)
    risk_warmup_visible_frames: int = Field(ge=2, default=3)
    motion_config: MotionEstimatorConfig = Field(default_factory=MotionEstimatorConfig)
    risk_config: PerceptionRiskConfig = Field(default_factory=PerceptionRiskConfig)


class TrackingBenchmarkMetrics(BaseModel):
    """Aggregate MOT, runtime, and downstream collision metrics."""

    sequences: int = Field(ge=1)
    frames: int = Field(ge=1)
    ground_truth_observations: int = Field(ge=0)
    tracker_observations: int = Field(ge=0)
    matched_observations: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    identity_switches: int = Field(ge=0)
    fragmentations: int = Field(ge=0)
    latency_median_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    latency_max_ms: float = Field(ge=0)
    risk_eligible_observations: int = Field(ge=0)
    risk_valid_observations: int = Field(ge=0)
    risk_valid_fraction: float = Field(ge=0, le=1)
    collision_true_positives: int = Field(ge=0)
    collision_false_positives: int = Field(ge=0)
    collision_false_negatives: int = Field(ge=0)
    collision_true_negatives: int = Field(ge=0)
    collision_precision: float = Field(ge=0, le=1)
    collision_recall: float = Field(ge=0, le=1)
    collision_f1: float = Field(ge=0, le=1)
    collision_state_accuracy: float = Field(ge=0, le=1)


class TrackingBenchmarkResult(BaseModel):
    """Named tracker result with reproducible configuration metadata."""

    tracker_name: str = Field(min_length=1)
    metrics: TrackingBenchmarkMetrics


TrackerFactory = Callable[[], MultiObjectTracker]


def benchmark_tracker(
    tracker_name: str,
    tracker_factory: TrackerFactory,
    sequences: Sequence[TrackingBenchmarkSequence],
    config: TrackingBenchmarkConfig | None = None,
) -> TrackingBenchmarkResult:
    """Evaluate one tracker on locked scenarios and downstream collision labels."""

    if not sequences:
        raise ValueError("At least one benchmark sequence is required")
    settings = config or TrackingBenchmarkConfig()

    total_frames = 0
    total_ground_truth = 0
    total_tracks = 0
    total_matches = 0
    identity_switches = 0
    fragmentations = 0
    latencies_ms: list[float] = []
    risk_eligible = 0
    risk_valid = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for sequence in sequences:
        tracker = tracker_factory()
        motion_estimator = TrackMotionEstimator(settings.motion_config)
        previous_track_by_truth: dict[int, int] = {}
        ever_matched: set[int] = set()
        matched_previous_frame: set[int] = set()
        visible_count_by_truth: dict[int, int] = {}

        for item in sequence.frames:
            total_frames += 1
            total_ground_truth += len(item.ground_truth)
            for truth in item.ground_truth:
                visible_count_by_truth[truth.object_id] = (
                    visible_count_by_truth.get(truth.object_id, 0) + 1
                )

            started_ns = perf_counter_ns()
            tracks = tuple(tracker.update(item.frame, item.detections))
            latencies_ms.append((perf_counter_ns() - started_ns) / 1_000_000)
            total_tracks += len(tracks)

            matches = _match_tracks_to_ground_truth(
                tracks,
                item.ground_truth,
                settings.matching_iou_threshold,
            )
            total_matches += len(matches)
            current_matched_truth = set(matches)

            for truth_id, track in matches.items():
                previous_track = previous_track_by_truth.get(truth_id)
                if previous_track is not None and previous_track != track.track_id:
                    identity_switches += 1
                if truth_id in ever_matched and truth_id not in matched_previous_frame:
                    fragmentations += 1
                previous_track_by_truth[truth_id] = track.track_id
                ever_matched.add(truth_id)
            matched_previous_frame = current_matched_truth

            truth_by_id = {truth.object_id: truth for truth in item.ground_truth}
            associations = tuple(
                _depth_association(track, truth_by_id[truth_id].camera_point)
                for truth_id, track in matches.items()
                if truth_by_id[truth_id].camera_point is not None
            )
            motions = motion_estimator.update(item.frame, associations)
            risk_results = assess_perception_risks(
                tracks,
                motions,
                item.frame.timestamp_ns / 1_000_000_000,
                settings.risk_config,
            )
            risk_by_track = {result.track_id: result for result in risk_results}

            for truth in item.ground_truth:
                visible_count = visible_count_by_truth[truth.object_id]
                if (
                    truth.camera_point is None
                    or visible_count < settings.risk_warmup_visible_frames
                ):
                    continue
                risk_eligible += 1
                matched_track = matches.get(truth.object_id)
                predicted_collision = False
                if matched_track is not None:
                    result = risk_by_track.get(matched_track.track_id)
                    if (
                        result is not None
                        and result.status is PerceptionRiskStatus.VALID
                        and result.assessment is not None
                    ):
                        risk_valid += 1
                        predicted_collision = result.assessment.predicted_collision

                if truth.expected_collision and predicted_collision:
                    true_positive += 1
                elif truth.expected_collision:
                    false_negative += 1
                elif predicted_collision:
                    false_positive += 1
                else:
                    true_negative += 1

    precision = _safe_ratio(total_matches, total_tracks)
    recall = _safe_ratio(total_matches, total_ground_truth)
    collision_precision = _safe_ratio(true_positive, true_positive + false_positive)
    collision_recall = _safe_ratio(true_positive, true_positive + false_negative)
    collision_f1 = _f1(collision_precision, collision_recall)
    collision_accuracy = _safe_ratio(true_positive + true_negative, risk_eligible)
    latency_array = np.asarray(latencies_ms, dtype=np.float64)

    return TrackingBenchmarkResult(
        tracker_name=tracker_name,
        metrics=TrackingBenchmarkMetrics(
            sequences=len(sequences),
            frames=total_frames,
            ground_truth_observations=total_ground_truth,
            tracker_observations=total_tracks,
            matched_observations=total_matches,
            precision=precision,
            recall=recall,
            identity_switches=identity_switches,
            fragmentations=fragmentations,
            latency_median_ms=float(np.median(latency_array)),
            latency_p95_ms=float(np.percentile(latency_array, 95)),
            latency_max_ms=float(np.max(latency_array)),
            risk_eligible_observations=risk_eligible,
            risk_valid_observations=risk_valid,
            risk_valid_fraction=_safe_ratio(risk_valid, risk_eligible),
            collision_true_positives=true_positive,
            collision_false_positives=false_positive,
            collision_false_negatives=false_negative,
            collision_true_negatives=true_negative,
            collision_precision=collision_precision,
            collision_recall=collision_recall,
            collision_f1=collision_f1,
            collision_state_accuracy=collision_accuracy,
        ),
    )


def benchmark_trackers(
    tracker_factories: dict[str, TrackerFactory],
    sequences: Sequence[TrackingBenchmarkSequence],
    config: TrackingBenchmarkConfig | None = None,
) -> tuple[TrackingBenchmarkResult, ...]:
    """Run multiple tracker factories against exactly the same scenarios."""

    if not tracker_factories:
        raise ValueError("At least one tracker factory is required")
    return tuple(
        benchmark_tracker(name, factory, sequences, config)
        for name, factory in sorted(tracker_factories.items())
    )


def _match_tracks_to_ground_truth(
    tracks: tuple[TrackObservation, ...],
    ground_truth: tuple[GroundTruthObservation, ...],
    minimum_iou: float,
) -> dict[int, TrackObservation]:
    candidates: list[tuple[float, int, int]] = []
    for truth_index, truth in enumerate(ground_truth):
        for track_index, track in enumerate(tracks):
            if truth.class_name != track.class_name:
                continue
            iou = truth.bbox.intersection_over_union(track.bbox)
            if iou >= minimum_iou:
                candidates.append((iou, truth_index, track_index))

    matches: dict[int, TrackObservation] = {}
    used_tracks: set[int] = set()
    for _, truth_index, track_index in sorted(candidates, reverse=True):
        truth = ground_truth[truth_index]
        if truth.object_id in matches or track_index in used_tracks:
            continue
        matches[truth.object_id] = tracks[track_index]
        used_tracks.add(track_index)
    return matches


def _depth_association(
    track: TrackObservation,
    point: CameraPoint | None,
) -> DepthAssociation:
    if point is None:
        raise ValueError("A benchmark depth association requires a camera point")
    return DepthAssociation(
        track_id=track.track_id,
        frame_id=track.frame_id,
        status=DepthStatus.VALID,
        roi_sample_count=100,
        valid_sample_count=100,
        inlier_sample_count=100,
        valid_fraction=1.0,
        depth_m=point.z_m,
        median_absolute_deviation_m=0.0,
        pixel_u=(track.bbox.x_min + track.bbox.x_max) / 2.0,
        pixel_v=(track.bbox.y_min + track.bbox.y_max) / 2.0,
        camera_point=point,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
