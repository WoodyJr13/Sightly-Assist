"""Ground-plane collision geometry for constant-velocity motion."""

from __future__ import annotations

from math import isfinite

from sightly_assist.schemas import MovingBody, Vector2

_EPSILON = 1e-9


def position_at(body: MovingBody, timestamp_s: float) -> Vector2:
    """Return a body's position after ``timestamp_s`` under constant velocity."""

    if timestamp_s < 0 or not isfinite(timestamp_s):
        raise ValueError("timestamp_s must be finite and non-negative")
    return body.position_m + body.velocity_mps.scale(timestamp_s)


def relative_state(
    observer: MovingBody,
    obstacle: MovingBody,
    timestamp_s: float,
) -> tuple[Vector2, Vector2]:
    """Return obstacle position and velocity relative to the observer."""

    relative_position = position_at(obstacle, timestamp_s) - position_at(
        observer,
        timestamp_s,
    )
    relative_velocity = obstacle.velocity_mps - observer.velocity_mps
    return relative_position, relative_velocity


def time_to_closest_approach(
    relative_position: Vector2,
    relative_velocity: Vector2,
    horizon_s: float,
) -> float:
    """Return clipped time to closest approach within ``[0, horizon_s]``."""

    if horizon_s <= 0 or not isfinite(horizon_s):
        raise ValueError("horizon_s must be finite and positive")

    speed_squared = relative_velocity.dot(relative_velocity)
    if speed_squared <= _EPSILON:
        return 0.0

    unconstrained = -relative_position.dot(relative_velocity) / speed_squared
    return min(max(unconstrained, 0.0), horizon_s)


def distance_at_closest_approach(
    relative_position: Vector2,
    relative_velocity: Vector2,
    closest_time_s: float,
) -> float:
    """Return separation at a previously calculated closest-approach time."""

    if closest_time_s < 0 or not isfinite(closest_time_s):
        raise ValueError("closest_time_s must be finite and non-negative")
    return (relative_position + relative_velocity.scale(closest_time_s)).magnitude


def closing_speed(relative_position: Vector2, relative_velocity: Vector2) -> float:
    """Return radial closing speed; positive means separation is decreasing."""

    distance = relative_position.magnitude
    if distance <= _EPSILON:
        return 0.0
    return -relative_position.dot(relative_velocity) / distance


def collision_boundary(
    observer: MovingBody,
    obstacle: MovingBody,
    safety_margin_m: float,
) -> float:
    """Return the combined circular collision boundary."""

    if safety_margin_m < 0 or not isfinite(safety_margin_m):
        raise ValueError("safety_margin_m must be finite and non-negative")
    return observer.radius_m + obstacle.radius_m + safety_margin_m
