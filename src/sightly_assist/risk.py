"""Transparent baseline risk assessment."""

from __future__ import annotations

from math import exp

from sightly_assist.geometry import (
    closing_speed,
    collision_boundary,
    distance_at_closest_approach,
    relative_state,
    time_to_closest_approach,
)
from sightly_assist.schemas import MovingBody, RiskAssessment, RiskLevel


def _risk_level(score: float, predicted_collision: bool) -> RiskLevel:
    if predicted_collision and score >= 0.8:
        return RiskLevel.CRITICAL
    if score >= 0.6:
        return RiskLevel.HIGH
    if score >= 0.35:
        return RiskLevel.MODERATE
    if score >= 0.1:
        return RiskLevel.LOW
    return RiskLevel.NONE


def assess_risk(
    observer: MovingBody,
    obstacle: MovingBody,
    timestamp_s: float,
    prediction_horizon_s: float,
    safety_margin_m: float,
) -> RiskAssessment:
    """Assess deterministic collision risk for one obstacle.

    The score is intentionally interpretable. It combines predicted clearance,
    urgency, and positive closing speed. This is a baseline for later calibrated
    probabilistic risk estimation; it is not presented as a collision probability.
    """

    relative_position, relative_velocity = relative_state(observer, obstacle, timestamp_s)
    closest_time = time_to_closest_approach(
        relative_position,
        relative_velocity,
        prediction_horizon_s,
    )
    closest_distance = distance_at_closest_approach(
        relative_position,
        relative_velocity,
        closest_time,
    )
    radial_closing_speed = closing_speed(relative_position, relative_velocity)
    boundary = collision_boundary(observer, obstacle, safety_margin_m)
    predicted_collision = closest_distance <= boundary and closest_time > 0

    clearance_ratio = max(0.0, 1.0 - closest_distance / max(boundary * 2.0, 1e-9))
    urgency = exp(-closest_time / max(prediction_horizon_s / 2.0, 1e-9))
    approach_factor = min(max(radial_closing_speed / 3.0, 0.0), 1.0)

    score = 0.45 * clearance_ratio + 0.35 * urgency + 0.20 * approach_factor
    if predicted_collision:
        score = max(score, 0.8)
    if radial_closing_speed <= 0 and not predicted_collision:
        score *= 0.25
    score = min(max(score, 0.0), 1.0)

    return RiskAssessment(
        obstacle_id=obstacle.id,
        timestamp_s=timestamp_s,
        relative_position_m=relative_position,
        relative_velocity_mps=relative_velocity,
        time_to_closest_approach_s=closest_time,
        distance_at_closest_approach_m=closest_distance,
        closing_speed_mps=radial_closing_speed,
        collision_boundary_m=boundary,
        predicted_collision=predicted_collision,
        risk_score=score,
        risk_level=_risk_level(score, predicted_collision),
    )
