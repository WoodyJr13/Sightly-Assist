"""Tests for the constrained warning-policy state machine."""

from __future__ import annotations

import pytest

from sightly_assist.motion_estimation import MotionStatus
from sightly_assist.perception_risk import PerceptionRiskResult, PerceptionRiskStatus
from sightly_assist.schemas import RiskAssessment, RiskLevel, Vector2
from sightly_assist.warning_policy import (
    AlertAction,
    WarningPolicy,
    WarningPolicyConfig,
)


def _risk(
    *,
    score: float,
    x_m: float = 0.0,
    z_m: float = 5.0,
    vx_mps: float = 0.0,
    vz_mps: float = -1.0,
    ttc_s: float = 4.0,
    predicted_collision: bool = False,
) -> PerceptionRiskResult:
    level = RiskLevel.CRITICAL if predicted_collision else RiskLevel.MODERATE
    return PerceptionRiskResult(
        track_id=2,
        class_name="person",
        status=PerceptionRiskStatus.VALID,
        motion_status=MotionStatus.VALID,
        assessment=RiskAssessment(
            obstacle_id="track-2",
            timestamp_s=1.0,
            relative_position_m=Vector2(x=x_m, z=z_m),
            relative_velocity_mps=Vector2(x=vx_mps, z=vz_mps),
            time_to_closest_approach_s=ttc_s,
            distance_at_closest_approach_m=0.5,
            closing_speed_mps=max(0.0, -vz_mps),
            collision_boundary_m=1.0,
            predicted_collision=predicted_collision,
            risk_score=score,
            risk_level=level,
        ),
    )


def _unavailable() -> PerceptionRiskResult:
    return PerceptionRiskResult(
        track_id=2,
        class_name="person",
        status=PerceptionRiskStatus.MOTION_UNAVAILABLE,
        motion_status=MotionStatus.WARMING_UP,
    )


def test_predicted_collision_escalates_immediately_to_stop() -> None:
    policy = WarningPolicy()

    decision = policy.evaluate(
        (_risk(score=0.9, ttc_s=1.0, predicted_collision=True),),
        timestamp_s=0.0,
    )

    assert decision.action is AlertAction.STOP
    assert decision.should_emit
    assert decision.track_id == 2


def test_awareness_direction_uses_relative_lateral_position() -> None:
    left_policy = WarningPolicy()
    right_policy = WarningPolicy()

    left = left_policy.evaluate((_risk(score=0.4, x_m=-1.0),), timestamp_s=0.0)
    right = right_policy.evaluate((_risk(score=0.4, x_m=1.0),), timestamp_s=0.0)

    assert left.action is AlertAction.AWARENESS_LEFT
    assert right.action is AlertAction.AWARENESS_RIGHT


def test_unavailable_motion_causes_abstention() -> None:
    decision = WarningPolicy().evaluate((_unavailable(),), timestamp_s=0.0)

    assert decision.action is AlertAction.ABSTAIN
    assert decision.should_emit


def test_minimum_hold_prevents_stop_chatter() -> None:
    policy = WarningPolicy(WarningPolicyConfig(minimum_hold_s=0.6))
    policy.evaluate(
        (_risk(score=0.9, ttc_s=1.0, predicted_collision=True),),
        timestamp_s=0.0,
    )

    held = policy.evaluate((_risk(score=0.05, vz_mps=1.0),), timestamp_s=0.2)
    released = policy.evaluate((_risk(score=0.05, vz_mps=1.0),), timestamp_s=0.7)

    assert held.action is AlertAction.STOP
    assert not held.should_emit
    assert released.action is AlertAction.NO_ALERT
    assert not released.should_emit


def test_repeated_alerts_are_rate_limited() -> None:
    policy = WarningPolicy(
        WarningPolicyConfig(
            repeat_interval_s=1.5,
            minimum_hold_s=0.0,
        )
    )
    risk = _risk(score=0.7, ttc_s=2.5)

    first = policy.evaluate((risk,), timestamp_s=0.0)
    suppressed = policy.evaluate((risk,), timestamp_s=0.5)
    repeated = policy.evaluate((risk,), timestamp_s=1.6)

    assert first.action is AlertAction.SLOW
    assert first.should_emit
    assert not suppressed.should_emit
    assert repeated.should_emit


def test_lower_risk_uncertain_track_forces_abstention() -> None:
    uncertain = PerceptionRiskResult(
        track_id=7,
        class_name="car",
        status=PerceptionRiskStatus.EXCESSIVE_UNCERTAINTY,
        motion_status=MotionStatus.VALID,
    )

    decision = WarningPolicy().evaluate(
        (_risk(score=0.4, x_m=1.0), uncertain),
        timestamp_s=0.0,
    )

    assert decision.action is AlertAction.ABSTAIN


def test_warning_timestamps_must_be_nondecreasing() -> None:
    policy = WarningPolicy()
    policy.evaluate((), timestamp_s=1.0)

    with pytest.raises(ValueError, match="nondecreasing"):
        policy.evaluate((), timestamp_s=0.9)
