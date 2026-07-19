"""Analytical and property tests for collision geometry."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sightly_assist.geometry import (
    closing_speed,
    distance_at_closest_approach,
    time_to_closest_approach,
)
from sightly_assist.schemas import Vector2


def test_head_on_closest_approach() -> None:
    position = Vector2(x=0.0, z=10.0)
    velocity = Vector2(x=0.0, z=-2.0)

    closest_time = time_to_closest_approach(position, velocity, horizon_s=10.0)
    closest_distance = distance_at_closest_approach(position, velocity, closest_time)

    assert closest_time == pytest.approx(5.0)
    assert closest_distance == pytest.approx(0.0)
    assert closing_speed(position, velocity) == pytest.approx(2.0)


def test_receding_object_is_closest_now() -> None:
    position = Vector2(x=0.0, z=4.0)
    velocity = Vector2(x=0.0, z=1.0)

    assert time_to_closest_approach(position, velocity, horizon_s=5.0) == 0.0
    assert closing_speed(position, velocity) == pytest.approx(-1.0)


def test_crossing_trajectory() -> None:
    position = Vector2(x=-4.0, z=4.0)
    velocity = Vector2(x=1.2, z=-1.0)

    closest_time = time_to_closest_approach(position, velocity, horizon_s=5.0)
    closest_distance = distance_at_closest_approach(position, velocity, closest_time)

    assert closest_time == pytest.approx(3.606557377, rel=1e-7)
    assert closest_distance == pytest.approx(0.51214752, rel=1e-6)


@given(
    px=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    pz=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    vx=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    vz=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    horizon=st.floats(0.001, 20, allow_nan=False, allow_infinity=False),
)
def test_closest_approach_is_bounded_and_nonnegative(
    px: float,
    pz: float,
    vx: float,
    vz: float,
    horizon: float,
) -> None:
    position = Vector2(x=px, z=pz)
    velocity = Vector2(x=vx, z=vz)

    closest_time = time_to_closest_approach(position, velocity, horizon)
    distance = distance_at_closest_approach(position, velocity, closest_time)

    assert 0.0 <= closest_time <= horizon
    assert distance >= 0.0
