"""Shared typed schemas for simulation and risk evaluation."""

from __future__ import annotations

from enum import StrEnum
from math import hypot

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskLevel(StrEnum):
    """Coarse risk classification used by the baseline policy."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Vector2(BaseModel):
    """Two-dimensional ground-plane vector in meters or meters per second."""

    model_config = ConfigDict(frozen=True)

    x: float
    z: float

    @property
    def magnitude(self) -> float:
        return hypot(self.x, self.z)

    def __add__(self, other: Vector2) -> Vector2:
        return Vector2(x=self.x + other.x, z=self.z + other.z)

    def __sub__(self, other: Vector2) -> Vector2:
        return Vector2(x=self.x - other.x, z=self.z - other.z)

    def scale(self, scalar: float) -> Vector2:
        return Vector2(x=self.x * scalar, z=self.z * scalar)

    def dot(self, other: Vector2) -> float:
        return self.x * other.x + self.z * other.z


class MovingBody(BaseModel):
    """A circular body moving with constant velocity on the ground plane."""

    id: str = Field(min_length=1)
    position_m: Vector2
    velocity_mps: Vector2
    radius_m: float = Field(gt=0)


class ScenarioExpectation(BaseModel):
    """Expected outcome used for scenario regression checks."""

    hazard: bool
    primary_obstacle_id: str | None = None


class ScenarioConfig(BaseModel):
    """Deterministic simulation scenario loaded from YAML."""

    scenario_id: str = Field(min_length=1)
    duration_s: float = Field(gt=0)
    time_step_s: float = Field(gt=0)
    prediction_horizon_s: float = Field(gt=0, default=5.0)
    safety_margin_m: float = Field(gt=0, default=0.35)
    observer: MovingBody
    obstacles: list[MovingBody] = Field(min_length=1)
    expected: ScenarioExpectation

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ScenarioConfig:
        obstacle_ids = [obstacle.id for obstacle in self.obstacles]
        if len(obstacle_ids) != len(set(obstacle_ids)):
            raise ValueError("Obstacle IDs must be unique")
        if self.observer.id in obstacle_ids:
            raise ValueError("Observer ID cannot also be used by an obstacle")
        return self


class RiskAssessment(BaseModel):
    """Risk result for one obstacle at one simulation timestamp."""

    obstacle_id: str
    timestamp_s: float = Field(ge=0)
    relative_position_m: Vector2
    relative_velocity_mps: Vector2
    time_to_closest_approach_s: float = Field(ge=0)
    distance_at_closest_approach_m: float = Field(ge=0)
    closing_speed_mps: float
    collision_boundary_m: float = Field(gt=0)
    predicted_collision: bool
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel


class SimulationStep(BaseModel):
    """All obstacle assessments produced at a simulation timestamp."""

    timestamp_s: float = Field(ge=0)
    observer_position_m: Vector2
    assessments: list[RiskAssessment]
