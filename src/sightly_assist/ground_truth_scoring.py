"""Quantitative scoring of replay outputs against independent ground truth."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from sightly_assist.depth_association import DepthAssociation, DepthStatus
from sightly_assist.ground_truth import (
    GroundTruthAnnotationSet,
    GroundTruthEvent,
    GroundTruthFrame,
    GroundTruthObject,
    load_ground_truth,
)
from sightly_assist.motion_estimation import MotionEstimate, MotionStatus
from sightly_assist.perception import FramePacket, TrackObservation
from sightly_assist.perception_risk import (
    PerceptionRiskConfig,
    PerceptionRiskResult,
    PerceptionRiskStatus,
)
from sightly_assist.risk import assess_relative_risk
from sightly_assist.schemas import RiskAssessment, Vector2
from sightly_assist.warning_policy import AlertAction, WarningDecision


class GroundTruthScoringConfig(BaseModel):
    """Matching and collision-geometry settings for quantitative scoring."""

    match_iou_threshold: float = Field(gt=0, le=1, default=0.30)
    class_agnostic_matching: bool = False
    risk_config: PerceptionRiskConfig = Field(default_factory=PerceptionRiskConfig)


class ErrorDistribution(BaseModel):
    """Signed and absolute error statistics for one measured quantity."""

    count: int = Field(ge=1)
    mean_error: float
    mean_absolute_error: float = Field(ge=0)
    median_absolute_error: float = Field(ge=0)
    root_mean_square_error: float = Field(ge=0)
    p95_absolute_error: float = Field(ge=0)
    maximum_absolute_error: float = Field(ge=0)


class GroundTruthScoreReport(BaseModel):
    """Aggregate tracking, geometry, event, and warning metrics."""

    schema_version: str = "1.0"
    sequence_id: str = Field(min_length=1)
    frame_count: int = Field(gt=0)
    duration_s: float = Field(ge=0)
    annotations_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    results_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ground_truth_object_frame_count: int = Field(ge=0)
    predicted_track_observation_count: int = Field(ge=0)
    matched_object_frame_count: int = Field(ge=0)
    track_precision: float = Field(ge=0, le=1)
    track_recall: float = Field(ge=0, le=1)
    identity_switches: int = Field(ge=0)
    fragmentations: int = Field(ge=0)
    depth_error_m: ErrorDistribution | None = None
    position_error_m: ErrorDistribution | None = None
    velocity_error_mps: ErrorDistribution | None = None
    time_to_closest_approach_error_s: ErrorDistribution | None = None
    closest_approach_distance_error_m: ErrorDistribution | None = None
    collision_frame_true_positives: int = Field(ge=0)
    collision_frame_false_positives: int = Field(ge=0)
    collision_frame_false_negatives: int = Field(ge=0)
    collision_frame_precision: float = Field(ge=0, le=1)
    collision_frame_recall: float = Field(ge=0, le=1)
    collision_frame_f1: float = Field(ge=0, le=1)
    ground_truth_event_count: int = Field(ge=0)
    predicted_event_count: int = Field(ge=0)
    detected_event_count: int = Field(ge=0)
    collision_event_precision: float = Field(ge=0, le=1)
    collision_event_recall: float = Field(ge=0, le=1)
    collision_event_f1: float = Field(ge=0, le=1)
    warning_event_recall: float = Field(ge=0, le=1)
    warning_lead_time_s: ErrorDistribution | None = None
    false_warning_count: int = Field(ge=0)
    false_warnings_per_minute: float = Field(ge=0)
    abstention_frame_count: int = Field(ge=0)
    notes: tuple[str, ...] = ()


class ScoredReplayFrame(BaseModel):
    """Subset of one exported replay result required by the scorer."""

    frame: FramePacket
    tracks: tuple[TrackObservation, ...] = ()
    depth_associations: tuple[DepthAssociation, ...] = ()
    motion_estimates: tuple[MotionEstimate, ...] = ()
    risk_results: tuple[PerceptionRiskResult, ...] = ()
    warning_decision: WarningDecision | None = None


@dataclass(frozen=True)
class _FrameMatch:
    gt_to_track: dict[str, TrackObservation]
    track_to_gt: dict[int, str]
    ignored_track_ids: frozenset[int]


@dataclass(frozen=True)
class _PredictedEvent:
    track_id: int
    start_frame_index: int
    end_frame_index: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    object_ids: frozenset[str]


@dataclass
class _TrackHistoryState:
    previous_track_id: int | None = None
    was_matched: bool = False
    matched_previous_frame: bool = False


def load_scored_replay(path: Path) -> tuple[ScoredReplayFrame, ...]:
    """Load frame-level JSONL emitted by the replay evaluator."""

    frames: list[ScoredReplayFrame] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                frames.append(ScoredReplayFrame.model_validate(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid replay result on JSONL line {line_number}: {exc}"
                ) from exc
    if not frames:
        raise ValueError("replay result JSONL must contain at least one frame")
    _validate_result_order(tuple(frames))
    return tuple(frames)


def score_ground_truth(
    annotations: GroundTruthAnnotationSet,
    results: tuple[ScoredReplayFrame, ...],
    config: GroundTruthScoringConfig | None = None,
    *,
    annotations_sha256: str | None = None,
    results_sha256: str | None = None,
) -> GroundTruthScoreReport:
    """Score one frame-aligned prediction sequence against independent labels."""

    settings = config or GroundTruthScoringConfig()
    _validate_alignment(annotations, results)
    matches = tuple(
        _match_frame(annotation_frame, result, settings)
        for annotation_frame, result in zip(annotations.frames, results, strict=True)
    )

    gt_object_count = sum(
        not item.ignore for frame in annotations.frames for item in frame.objects
    )
    predicted_track_count = sum(
        len(result.tracks) - len(match.ignored_track_ids)
        for result, match in zip(results, matches, strict=True)
    )
    matched_count = sum(len(match.gt_to_track) for match in matches)
    identity_switches, fragmentations = _identity_metrics(annotations, matches)

    depth_errors: list[float] = []
    position_errors: list[float] = []
    velocity_errors: list[float] = []
    tca_errors: list[float] = []
    dca_errors: list[float] = []
    collision_tp = 0
    collision_fp = 0
    collision_fn = 0

    for annotation_frame, result, match in zip(
        annotations.frames,
        results,
        matches,
        strict=True,
    ):
        objects_by_id = {
            item.object_id: item for item in annotation_frame.objects if not item.ignore
        }
        depth_by_track = {item.track_id: item for item in result.depth_associations}
        motion_by_track = {item.track_id: item for item in result.motion_estimates}
        risk_by_track = {item.track_id: item for item in result.risk_results}
        scored_predicted_collision_tracks: set[int] = set()

        for object_id, track in match.gt_to_track.items():
            ground_truth = objects_by_id[object_id]
            depth = depth_by_track.get(track.track_id)
            motion = motion_by_track.get(track.track_id)
            risk = risk_by_track.get(track.track_id)

            if (
                ground_truth.depth_m is not None
                and depth is not None
                and depth.status is DepthStatus.VALID
                and depth.depth_m is not None
            ):
                depth_errors.append(depth.depth_m - ground_truth.depth_m)

            if (
                ground_truth.position is not None
                and motion is not None
                and motion.position is not None
            ):
                position_errors.append(
                    _vector_error(
                        (
                            motion.position.x_m,
                            motion.position.y_m,
                            motion.position.z_m,
                        ),
                        (
                            ground_truth.position.x_m,
                            ground_truth.position.y_m,
                            ground_truth.position.z_m,
                        ),
                    )
                )

            if (
                ground_truth.velocity is not None
                and motion is not None
                and motion.status is MotionStatus.VALID
                and motion.velocity is not None
            ):
                velocity_errors.append(
                    _vector_error(
                        (
                            motion.velocity.x_mps,
                            motion.velocity.y_mps,
                            motion.velocity.z_mps,
                        ),
                        (
                            ground_truth.velocity.x_mps,
                            ground_truth.velocity.y_mps,
                            ground_truth.velocity.z_mps,
                        ),
                    )
                )

            truth_risk = _ground_truth_risk(
                ground_truth,
                annotation_frame.timestamp_ns,
                settings.risk_config,
            )
            if truth_risk is None:
                continue
            predicted_collision = bool(
                risk is not None
                and risk.status is PerceptionRiskStatus.VALID
                and risk.assessment is not None
                and risk.assessment.predicted_collision
            )
            if risk is not None and risk.assessment is not None:
                scored_predicted_collision_tracks.add(track.track_id)
                if risk.status is PerceptionRiskStatus.VALID:
                    tca_errors.append(
                        risk.assessment.time_to_closest_approach_s
                        - truth_risk.time_to_closest_approach_s
                    )
                    dca_errors.append(
                        risk.assessment.distance_at_closest_approach_m
                        - truth_risk.distance_at_closest_approach_m
                    )
            if truth_risk.predicted_collision and predicted_collision:
                collision_tp += 1
            elif truth_risk.predicted_collision:
                collision_fn += 1
            elif predicted_collision:
                collision_fp += 1

        for risk in result.risk_results:
            if (
                risk.track_id in scored_predicted_collision_tracks
                or risk.track_id in match.ignored_track_ids
            ):
                continue
            if (
                risk.status is PerceptionRiskStatus.VALID
                and risk.assessment is not None
                and risk.assessment.predicted_collision
            ):
                collision_fp += 1

    predicted_events = _predicted_events(results, matches)
    event_true_positives, detected_event_ids = _score_events(
        predicted_events,
        annotations.events,
    )
    warning_leads, false_warning_count = _score_warnings(
        annotations.events,
        results,
        matches,
    )

    duration_s = _duration_s(results)
    collision_frame_precision = _safe_ratio(collision_tp, collision_tp + collision_fp)
    collision_frame_recall = _safe_ratio(collision_tp, collision_tp + collision_fn)
    collision_event_precision = _safe_ratio(event_true_positives, len(predicted_events))
    collision_event_recall = _safe_ratio(len(detected_event_ids), len(annotations.events))
    warning_event_recall = _safe_ratio(len(warning_leads), len(annotations.events))

    notes = (
        "Frame matching is greedy, class-aware IoU matching unless configured otherwise.",
        "CPA ground truth is derived only from independently annotated 3D position and velocity.",
        "Event precision groups contiguous collision-positive frames by predicted track ID.",
        "A warning is successful only when emitted by the critical timestamp with sufficient severity.",
    )
    return GroundTruthScoreReport(
        sequence_id=annotations.sequence_id,
        frame_count=len(results),
        duration_s=duration_s,
        annotations_sha256=annotations_sha256,
        results_sha256=results_sha256,
        ground_truth_object_frame_count=int(gt_object_count),
        predicted_track_observation_count=predicted_track_count,
        matched_object_frame_count=matched_count,
        track_precision=_safe_ratio(matched_count, predicted_track_count),
        track_recall=_safe_ratio(matched_count, int(gt_object_count)),
        identity_switches=identity_switches,
        fragmentations=fragmentations,
        depth_error_m=_distribution(depth_errors),
        position_error_m=_distribution(position_errors),
        velocity_error_mps=_distribution(velocity_errors),
        time_to_closest_approach_error_s=_distribution(tca_errors),
        closest_approach_distance_error_m=_distribution(dca_errors),
        collision_frame_true_positives=collision_tp,
        collision_frame_false_positives=collision_fp,
        collision_frame_false_negatives=collision_fn,
        collision_frame_precision=collision_frame_precision,
        collision_frame_recall=collision_frame_recall,
        collision_frame_f1=_f1(collision_frame_precision, collision_frame_recall),
        ground_truth_event_count=len(annotations.events),
        predicted_event_count=len(predicted_events),
        detected_event_count=len(detected_event_ids),
        collision_event_precision=collision_event_precision,
        collision_event_recall=collision_event_recall,
        collision_event_f1=_f1(collision_event_precision, collision_event_recall),
        warning_event_recall=warning_event_recall,
        warning_lead_time_s=_distribution(warning_leads),
        false_warning_count=false_warning_count,
        false_warnings_per_minute=(
            false_warning_count / (duration_s / 60.0) if duration_s > 0 else 0.0
        ),
        abstention_frame_count=sum(
            result.warning_decision is not None
            and result.warning_decision.action is AlertAction.ABSTAIN
            for result in results
        ),
        notes=notes,
    )


def score_evaluation_files(
    annotations_path: Path,
    results_path: Path,
    output_path: Path,
    config: GroundTruthScoringConfig | None = None,
) -> GroundTruthScoreReport:
    """Load, score, and export one annotation/result pair."""

    annotations = load_ground_truth(annotations_path)
    results = load_scored_replay(results_path)
    report = score_ground_truth(
        annotations,
        results,
        config,
        annotations_sha256=_sha256(annotations_path),
        results_sha256=_sha256(results_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _validate_result_order(results: tuple[ScoredReplayFrame, ...]) -> None:
    frame_ids = [result.frame.frame_id for result in results]
    timestamps = [result.frame.timestamp_ns for result in results]
    if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("replay result frames must have unique increasing frame IDs")
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise ValueError("replay result frames must have unique increasing timestamps")


def _validate_alignment(
    annotations: GroundTruthAnnotationSet,
    results: tuple[ScoredReplayFrame, ...],
) -> None:
    if len(annotations.frames) != len(results):
        raise ValueError("ground truth and replay results must contain the same frame count")
    for annotation, result in zip(annotations.frames, results, strict=True):
        if annotation.frame_id != result.frame.frame_id:
            raise ValueError("ground truth frame IDs do not align with replay results")
        if annotation.timestamp_ns != result.frame.timestamp_ns:
            raise ValueError("ground truth timestamps do not align with replay results")


def _match_frame(
    ground_truth: GroundTruthFrame,
    result: ScoredReplayFrame,
    config: GroundTruthScoringConfig,
) -> _FrameMatch:
    candidates: list[tuple[float, str, int]] = []
    scorable_objects = tuple(item for item in ground_truth.objects if not item.ignore)
    ignored_objects = tuple(item for item in ground_truth.objects if item.ignore)
    for item in scorable_objects:
        for track in result.tracks:
            if not config.class_agnostic_matching and item.class_name != track.class_name:
                continue
            iou = item.bbox.intersection_over_union(track.bbox)
            if iou >= config.match_iou_threshold:
                candidates.append((iou, item.object_id, track.track_id))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))

    tracks_by_id = {track.track_id: track for track in result.tracks}
    gt_to_track: dict[str, TrackObservation] = {}
    track_to_gt: dict[int, str] = {}
    for _iou, object_id, track_id in candidates:
        if object_id in gt_to_track or track_id in track_to_gt:
            continue
        gt_to_track[object_id] = tracks_by_id[track_id]
        track_to_gt[track_id] = object_id

    ignored_track_ids = {
        track.track_id
        for track in result.tracks
        if track.track_id not in track_to_gt
        and any(
            ignored.bbox.intersection_over_union(track.bbox) >= config.match_iou_threshold
            for ignored in ignored_objects
        )
    }
    return _FrameMatch(
        gt_to_track=gt_to_track,
        track_to_gt=track_to_gt,
        ignored_track_ids=frozenset(ignored_track_ids),
    )


def _identity_metrics(
    annotations: GroundTruthAnnotationSet,
    matches: tuple[_FrameMatch, ...],
) -> tuple[int, int]:
    states: dict[str, _TrackHistoryState] = defaultdict(_TrackHistoryState)
    identity_switches = 0
    fragmentations = 0
    for frame, match in zip(annotations.frames, matches, strict=True):
        visible_ids = {item.object_id for item in frame.objects if not item.ignore}
        for object_id in visible_ids:
            state = states[object_id]
            track = match.gt_to_track.get(object_id)
            if track is None:
                state.matched_previous_frame = False
                continue
            if state.previous_track_id is not None and state.previous_track_id != track.track_id:
                identity_switches += 1
            if state.was_matched and not state.matched_previous_frame:
                fragmentations += 1
            state.previous_track_id = track.track_id
            state.was_matched = True
            state.matched_previous_frame = True
    return identity_switches, fragmentations


def _ground_truth_risk(
    item: GroundTruthObject,
    timestamp_ns: int,
    config: PerceptionRiskConfig,
) -> RiskAssessment | None:
    if item.position is None or item.velocity is None:
        return None
    obstacle_radius = config.class_radius_m.get(
        item.class_name,
        config.default_obstacle_radius_m,
    )
    boundary = config.observer_radius_m + obstacle_radius + config.safety_margin_m
    return assess_relative_risk(
        obstacle_id=f"ground-truth-{item.object_id}",
        timestamp_s=timestamp_ns / 1_000_000_000,
        relative_position=Vector2(x=item.position.x_m, z=item.position.z_m),
        relative_velocity=Vector2(x=item.velocity.x_mps, z=item.velocity.z_mps),
        prediction_horizon_s=config.prediction_horizon_s,
        collision_boundary_m=boundary,
    )


def _predicted_events(
    results: tuple[ScoredReplayFrame, ...],
    matches: tuple[_FrameMatch, ...],
) -> tuple[_PredictedEvent, ...]:
    positives: dict[int, list[tuple[int, int, str | None]]] = defaultdict(list)
    for frame_index, (result, match) in enumerate(zip(results, matches, strict=True)):
        for risk in result.risk_results:
            if (
                risk.status is PerceptionRiskStatus.VALID
                and risk.assessment is not None
                and risk.assessment.predicted_collision
                and risk.track_id not in match.ignored_track_ids
            ):
                positives[risk.track_id].append(
                    (
                        frame_index,
                        result.frame.timestamp_ns,
                        match.track_to_gt.get(risk.track_id),
                    )
                )

    events: list[_PredictedEvent] = []
    for track_id, samples in positives.items():
        segment: list[tuple[int, int, str | None]] = []
        for sample in samples:
            if segment and sample[0] != segment[-1][0] + 1:
                events.append(_build_predicted_event(track_id, segment))
                segment = []
            segment.append(sample)
        if segment:
            events.append(_build_predicted_event(track_id, segment))
    return tuple(sorted(events, key=lambda event: (event.start_timestamp_ns, event.track_id)))


def _build_predicted_event(
    track_id: int,
    samples: list[tuple[int, int, str | None]],
) -> _PredictedEvent:
    object_ids = frozenset(item[2] for item in samples if item[2] is not None)
    return _PredictedEvent(
        track_id=track_id,
        start_frame_index=samples[0][0],
        end_frame_index=samples[-1][0],
        start_timestamp_ns=samples[0][1],
        end_timestamp_ns=samples[-1][1],
        object_ids=object_ids,
    )


def _score_events(
    predicted: tuple[_PredictedEvent, ...],
    ground_truth: tuple[GroundTruthEvent, ...],
) -> tuple[int, frozenset[str]]:
    true_positive_predictions = 0
    detected: set[str] = set()
    for prediction in predicted:
        overlaps = [
            event
            for event in ground_truth
            if event.object_id in prediction.object_ids
            and prediction.start_timestamp_ns <= event.end_timestamp_ns
            and prediction.end_timestamp_ns >= event.start_timestamp_ns
        ]
        if overlaps:
            true_positive_predictions += 1
            detected.update(event.event_id for event in overlaps)
    return true_positive_predictions, frozenset(detected)


def _score_warnings(
    events: tuple[GroundTruthEvent, ...],
    results: tuple[ScoredReplayFrame, ...],
    matches: tuple[_FrameMatch, ...],
) -> tuple[list[float], int]:
    lead_times: list[float] = []
    for event in events:
        successful_timestamps: list[int] = []
        required_severity = _warning_severity(event.minimum_warning_action)
        for result, match in zip(results, matches, strict=True):
            decision = result.warning_decision
            timestamp = result.frame.timestamp_ns
            if (
                decision is None
                or not decision.should_emit
                or decision.track_id is None
                or timestamp < event.start_timestamp_ns
                or timestamp > event.critical_timestamp_ns
                or _warning_severity(decision.action) < required_severity
                or match.track_to_gt.get(decision.track_id) != event.object_id
            ):
                continue
            successful_timestamps.append(timestamp)
        if successful_timestamps:
            first = min(successful_timestamps)
            lead_times.append((event.critical_timestamp_ns - first) / 1_000_000_000)

    false_warning_count = 0
    for result, match in zip(results, matches, strict=True):
        decision = result.warning_decision
        if (
            decision is None
            or not decision.should_emit
            or _warning_severity(decision.action) <= 0
        ):
            continue
        object_id = (
            match.track_to_gt.get(decision.track_id)
            if decision.track_id is not None
            else None
        )
        active = any(
            event.object_id == object_id
            and event.start_timestamp_ns <= result.frame.timestamp_ns <= event.end_timestamp_ns
            for event in events
        )
        if not active:
            false_warning_count += 1
    return lead_times, false_warning_count


def _warning_severity(action: AlertAction) -> int:
    if action is AlertAction.STOP:
        return 3
    if action is AlertAction.SLOW:
        return 2
    if action in {
        AlertAction.AWARENESS_LEFT,
        AlertAction.AWARENESS_CENTER,
        AlertAction.AWARENESS_RIGHT,
    }:
        return 1
    return 0


def _distribution(errors: list[float]) -> ErrorDistribution | None:
    if not errors:
        return None
    values = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(values)
    return ErrorDistribution(
        count=int(values.size),
        mean_error=float(np.mean(values)),
        mean_absolute_error=float(np.mean(absolute)),
        median_absolute_error=float(np.median(absolute)),
        root_mean_square_error=float(np.sqrt(np.mean(values**2))),
        p95_absolute_error=float(np.percentile(absolute, 95)),
        maximum_absolute_error=float(np.max(absolute)),
    )


def _vector_error(
    predicted: tuple[float, float, float],
    ground_truth: tuple[float, float, float],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(predicted, dtype=np.float64)
            - np.asarray(ground_truth, dtype=np.float64)
        )
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _duration_s(results: tuple[ScoredReplayFrame, ...]) -> float:
    if len(results) < 2:
        return 0.0
    return (
        results[-1].frame.timestamp_ns - results[0].frame.timestamp_ns
    ) / 1_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
