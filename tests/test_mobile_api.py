from __future__ import annotations

from fastapi.testclient import TestClient

from sightly_assist.depth_association import CameraPoint, DepthAssociation, DepthStatus
from sightly_assist.mobile_api import (
    DemoStateGenerator,
    MobileStateStore,
    create_mobile_app,
    idle_mobile_state,
    mobile_state_from_replay_result,
)
from sightly_assist.motion_estimation import MotionEstimate, MotionStatus, Velocity3D
from sightly_assist.perception import BoundingBox, FramePacket, TrackObservation
from sightly_assist.perception_risk import PerceptionRiskResult, PerceptionRiskStatus
from sightly_assist.replay_pipeline import ReplayResult, ReplayStageTimings
from sightly_assist.schemas import RiskAssessment, RiskLevel, Vector2
from sightly_assist.warning_policy import AlertAction, WarningDecision


def test_state_store_bounds_history_and_versions() -> None:
    store = MobileStateStore(history_size=2)
    initial_version, initial = store.snapshot()
    assert initial_version == 0
    assert initial.warning_action is AlertAction.ABSTAIN

    generator = DemoStateGenerator()
    first_version = store.publish(generator.state_at(3.0))
    second_version = store.publish(generator.state_at(7.8))

    assert first_version == 1
    assert second_version == 2
    assert len(store.history(10)) == 2
    assert store.snapshot()[1].warning_action is AlertAction.STOP


def test_http_and_websocket_expose_latest_state() -> None:
    state = DemoStateGenerator().state_at(7.8)
    store = MobileStateStore(state)
    app = create_mobile_app(store)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["schema_version"] == "1.0"

        current = client.get("/api/v1/state")
        assert current.status_code == 200
        assert current.json()["warning_action"] == "stop"

        with client.websocket_connect("/ws/live") as socket:
            payload = socket.receive_json()
            assert payload["frame_id"] == state.frame_id
            assert payload["warning_action"] == "stop"


def test_demo_generator_covers_clear_slow_stop_and_directional_states() -> None:
    generator = DemoStateGenerator()
    assert generator.state_at(0.5).warning_action is AlertAction.NO_ALERT
    assert generator.state_at(4.0).warning_action in {
        AlertAction.AWARENESS_CENTER,
        AlertAction.SLOW,
    }
    assert generator.state_at(7.8).warning_action is AlertAction.STOP
    assert generator.state_at(10.0).warning_action is AlertAction.AWARENESS_LEFT


def test_replay_result_converts_to_normalized_mobile_state() -> None:
    frame = FramePacket(
        frame_id=7,
        timestamp_ns=700_000_000,
        width_px=100,
        height_px=80,
        rgb_path="frame.png",
        depth_path="depth.npy",
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    track = TrackObservation(
        track_id=4,
        frame_id=7,
        class_name="person",
        confidence=0.93,
        bbox=BoundingBox(x_min=20, y_min=10, x_max=60, y_max=70),
        age_frames=5,
        missed_frames=0,
    )
    depth = DepthAssociation(
        track_id=4,
        frame_id=7,
        status=DepthStatus.VALID,
        roi_sample_count=100,
        valid_sample_count=95,
        inlier_sample_count=90,
        valid_fraction=0.95,
        depth_m=2.2,
        median_absolute_deviation_m=0.02,
        pixel_u=40,
        pixel_v=40,
        camera_point=CameraPoint(x_m=0.1, y_m=0.0, z_m=2.2),
    )
    motion = MotionEstimate(
        track_id=4,
        frame_id=7,
        timestamp_ns=700_000_000,
        status=MotionStatus.VALID,
        sample_count=5,
        inlier_count=5,
        time_span_s=0.4,
        position=CameraPoint(x_m=0.1, y_m=0.0, z_m=2.2),
        velocity=Velocity3D(x_mps=0.0, y_mps=0.0, z_mps=-1.1),
        residual_rms_m=0.03,
    )
    assessment = RiskAssessment(
        obstacle_id="track-4",
        timestamp_s=0.7,
        relative_position_m=Vector2(x=0.1, z=2.2),
        relative_velocity_mps=Vector2(x=0.0, z=-1.1),
        time_to_closest_approach_s=2.0,
        distance_at_closest_approach_m=0.1,
        closing_speed_mps=1.1,
        collision_boundary_m=1.0,
        predicted_collision=True,
        risk_score=0.82,
        risk_level=RiskLevel.CRITICAL,
    )
    risk = PerceptionRiskResult(
        track_id=4,
        class_name="person",
        status=PerceptionRiskStatus.VALID,
        motion_status=MotionStatus.VALID,
        assessment=assessment,
    )
    warning = WarningDecision(
        timestamp_s=0.7,
        action=AlertAction.STOP,
        should_emit=True,
        reason="predicted collision",
        track_id=4,
        risk_score=0.82,
        time_to_closest_approach_s=2.0,
    )
    timings = ReplayStageTimings(
        image_load_ms=1,
        detector_ms=12,
        tracker_ms=2,
        depth_load_ms=1,
        depth_association_ms=2,
        free_space_ms=1,
        motion_ms=1,
        risk_ms=1,
        warning_ms=1,
        total_ms=25,
    )

    state = mobile_state_from_replay_result(
        ReplayResult(
            frame=frame,
            detections=(),
            tracks=(track,),
            depth_associations=(depth,),
            motion_estimates=(motion,),
            risk_results=(risk,),
            warning_decision=warning,
            stage_timings=timings,
        ),
        sequence_id="test-sequence",
    )

    assert state.warning_action is AlertAction.STOP
    assert state.orientation_available
    assert state.metrics.processing_fps == 40
    assert state.hazards[0].bbox.x_min == 0.2
    assert state.hazards[0].bbox.y_max == 0.875
    assert state.hazards[0].predicted_collision is True


def test_idle_state_is_explicitly_unavailable() -> None:
    state = idle_mobile_state()
    assert state.warning_action is AlertAction.ABSTAIN
    assert not state.synchronized
    assert state.hazards == ()
