from __future__ import annotations

import pytest

from sightly_assist.depth_association import CameraPoint, DepthAssociation, DepthStatus
from sightly_assist.ground_truth import (
    GroundTruthAnnotationSet,
    GroundTruthEvent,
    GroundTruthFrame,
    GroundTruthObject,
)
from sightly_assist.ground_truth_scoring import ScoredReplayFrame, score_ground_truth
from sightly_assist.motion_estimation import MotionEstimate, MotionStatus, Velocity3D
from sightly_assist.perception import BoundingBox, FramePacket, TrackObservation
from sightly_assist.perception_risk import PerceptionRiskResult, PerceptionRiskStatus
from sightly_assist.risk import assess_relative_risk
from sightly_assist.schemas import Vector2
from sightly_assist.warning_policy import AlertAction, WarningDecision

GT_BOX = BoundingBox(x_min=10, y_min=8, x_max=30, y_max=54)
FALSE_BOX = BoundingBox(x_min=36, y_min=8, x_max=58, y_max=54)
COLLISION_BOUNDARY_M = 1.0


def _frame(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=frame_id * 1_000_000_000,
        width_px=64,
        height_px=64,
        rgb_path=f"rgb/{frame_id:06d}.png",
        depth_path=f"depth/{frame_id:06d}.npy",
    )


def _track(frame_id: int, track_id: int, bbox: BoundingBox) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        frame_id=frame_id,
        class_name="person",
        confidence=0.95,
        bbox=bbox,
        age_frames=frame_id + 1,
        missed_frames=0,
    )


def _risk(frame_id: int, track_id: int, z_m: float, z_mps: float) -> PerceptionRiskResult:
    assessment = assess_relative_risk(
        obstacle_id=f"track-{track_id}",
        timestamp_s=float(frame_id),
        relative_position=Vector2(x=0.0, z=z_m),
        relative_velocity=Vector2(x=0.0, z=z_mps),
        prediction_horizon_s=5.0,
        collision_boundary_m=COLLISION_BOUNDARY_M,
    )
    return PerceptionRiskResult(
        track_id=track_id,
        class_name="person",
        status=PerceptionRiskStatus.VALID,
        motion_status=MotionStatus.VALID,
        assessment=assessment,
    )


def test_geometry_and_event_metrics_distinguish_true_and_false_hazards() -> None:
    ground_truth_frames: list[GroundTruthFrame] = []
    result_frames: list[ScoredReplayFrame] = []
    for frame_id, z_m in enumerate((3.0, 2.0, 1.0)):
        ground_truth_frames.append(
            GroundTruthFrame(
                frame_id=frame_id,
                timestamp_ns=frame_id * 1_000_000_000,
                objects=(
                    GroundTruthObject(
                        object_id="person-a",
                        class_name="person",
                        bbox=GT_BOX,
                        depth_m=z_m,
                        position=CameraPoint(x_m=0.0, y_m=0.0, z_m=z_m),
                        velocity=Velocity3D(x_mps=0.0, y_mps=0.0, z_mps=-1.0),
                    ),
                ),
            )
        )
        tracks = [_track(frame_id, 7, GT_BOX)]
        risks = [_risk(frame_id, 7, z_m, -0.9)]
        if frame_id == 1:
            tracks.append(_track(frame_id, 99, FALSE_BOX))
            risks.append(_risk(frame_id, 99, 1.5, -1.0))
        result_frames.append(
            ScoredReplayFrame(
                frame=_frame(frame_id),
                tracks=tuple(tracks),
                depth_associations=(
                    DepthAssociation(
                        track_id=7,
                        frame_id=frame_id,
                        status=DepthStatus.VALID,
                        roi_sample_count=100,
                        valid_sample_count=100,
                        inlier_sample_count=100,
                        valid_fraction=1.0,
                        depth_m=z_m + 0.1,
                        median_absolute_deviation_m=0.01,
                        pixel_u=20,
                        pixel_v=31,
                        camera_point=CameraPoint(x_m=0.0, y_m=0.0, z_m=z_m + 0.1),
                    ),
                ),
                motion_estimates=(
                    MotionEstimate(
                        track_id=7,
                        frame_id=frame_id,
                        timestamp_ns=frame_id * 1_000_000_000,
                        status=MotionStatus.VALID,
                        sample_count=3,
                        inlier_count=3,
                        time_span_s=2.0,
                        position=CameraPoint(x_m=0.0, y_m=0.0, z_m=z_m),
                        velocity=Velocity3D(x_mps=0.0, y_mps=0.0, z_mps=-0.9),
                        residual_rms_m=0.01,
                    ),
                ),
                risk_results=tuple(risks),
                warning_decision=(
                    WarningDecision(
                        timestamp_s=0.0,
                        action=AlertAction.SLOW,
                        should_emit=True,
                        reason="synthetic early warning",
                        track_id=7,
                    )
                    if frame_id == 0
                    else WarningDecision(
                        timestamp_s=float(frame_id),
                        action=AlertAction.SLOW,
                        should_emit=False,
                        reason="repeat suppressed",
                        track_id=7,
                    )
                ),
            )
        )

    annotations = GroundTruthAnnotationSet(
        sequence_id="geometry-score",
        annotator="instrumented-test",
        frames=tuple(ground_truth_frames),
        events=(
            GroundTruthEvent(
                event_id="person-approach",
                object_id="person-a",
                start_timestamp_ns=0,
                critical_timestamp_ns=2_000_000_000,
                end_timestamp_ns=2_000_000_000,
                minimum_warning_action=AlertAction.SLOW,
            ),
        ),
    )

    report = score_ground_truth(annotations, tuple(result_frames))

    assert report.track_recall == pytest.approx(1.0)
    assert report.track_precision == pytest.approx(0.75)
    assert report.depth_error_m is not None
    assert report.depth_error_m.mean_absolute_error == pytest.approx(0.1)
    assert report.position_error_m is not None
    assert report.position_error_m.mean_absolute_error == pytest.approx(0.0)
    assert report.velocity_error_mps is not None
    assert report.velocity_error_mps.mean_absolute_error == pytest.approx(0.1)
    assert report.time_to_closest_approach_error_s is not None
    assert report.closest_approach_distance_error_m is not None
    assert report.collision_frame_true_positives == 3
    assert report.collision_frame_false_positives == 1
    assert report.collision_frame_false_negatives == 0
    assert report.collision_frame_precision == pytest.approx(0.75)
    assert report.collision_frame_recall == pytest.approx(1.0)
    assert report.predicted_event_count == 2
    assert report.detected_event_count == 1
    assert report.collision_event_precision == pytest.approx(0.5)
    assert report.collision_event_recall == pytest.approx(1.0)
    assert report.warning_event_recall == pytest.approx(1.0)
    assert report.warning_lead_time_s is not None
    assert report.warning_lead_time_s.mean_error == pytest.approx(2.0)
    assert report.false_warning_count == 0
