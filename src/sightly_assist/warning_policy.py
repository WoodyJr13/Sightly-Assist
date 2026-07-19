"""Constrained, uncertainty-aware warning policy with temporal hysteresis."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from sightly_assist.perception_risk import (
    PerceptionRiskResult,
    PerceptionRiskStatus,
)


class AlertAction(StrEnum):
    """Allowed user-facing actions; no directional movement commands are included."""

    NO_ALERT = "no_alert"
    AWARENESS_LEFT = "awareness_left"
    AWARENESS_CENTER = "awareness_center"
    AWARENESS_RIGHT = "awareness_right"
    SLOW = "slow"
    STOP = "stop"
    ABSTAIN = "abstain"


class WarningPolicyConfig(BaseModel):
    """Thresholds, hysteresis, and repeat timing for warning decisions."""

    awareness_score: float = Field(ge=0, le=1, default=0.35)
    slow_score: float = Field(ge=0, le=1, default=0.60)
    stop_score: float = Field(ge=0, le=1, default=0.80)
    slow_time_to_collision_s: float = Field(gt=0, default=3.0)
    stop_time_to_collision_s: float = Field(gt=0, default=1.5)
    direction_deadband_m: float = Field(ge=0, default=0.35)
    minimum_hold_s: float = Field(ge=0, default=0.60)
    repeat_interval_s: float = Field(gt=0, default=1.50)
    release_margin: float = Field(ge=0, le=0.5, default=0.10)

    @model_validator(mode="after")
    def validate_thresholds(self) -> WarningPolicyConfig:
        if not self.awareness_score <= self.slow_score <= self.stop_score:
            raise ValueError("risk thresholds must be ordered awareness <= slow <= stop")
        if self.stop_time_to_collision_s > self.slow_time_to_collision_s:
            raise ValueError("stop TTC threshold cannot exceed slow TTC threshold")
        return self


class WarningDecision(BaseModel):
    """One stateful warning-policy decision."""

    timestamp_s: float = Field(ge=0)
    action: AlertAction
    should_emit: bool
    reason: str = Field(min_length=1)
    track_id: int | None = Field(default=None, ge=0)
    risk_score: float | None = Field(default=None, ge=0, le=1)
    time_to_closest_approach_s: float | None = Field(default=None, ge=0)


class WarningPolicy:
    """Select and stabilize warnings from per-track collision-risk results."""

    def __init__(self, config: WarningPolicyConfig | None = None) -> None:
        self.config = config or WarningPolicyConfig()
        self._current_action = AlertAction.NO_ALERT
        self._current_since_s = 0.0
        self._last_emitted_s: float | None = None
        self._last_timestamp_s: float | None = None

    def evaluate(
        self,
        risk_results: tuple[PerceptionRiskResult, ...],
        timestamp_s: float,
    ) -> WarningDecision:
        """Choose an allowed action and apply temporal hysteresis."""

        if timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative")
        if self._last_timestamp_s is not None and timestamp_s < self._last_timestamp_s:
            raise ValueError("warning timestamps must be nondecreasing")
        self._last_timestamp_s = timestamp_s

        desired, selected, reason = self._desired_action(risk_results)
        action = self._stabilize(desired, timestamp_s)
        should_emit = self._should_emit(action, timestamp_s)

        if action is not desired:
            reason = f"holding {action.value} to prevent alert chatter"

        assessment = selected.assessment if selected is not None else None
        return WarningDecision(
            timestamp_s=timestamp_s,
            action=action,
            should_emit=should_emit,
            reason=reason,
            track_id=selected.track_id if selected is not None else None,
            risk_score=assessment.risk_score if assessment is not None else None,
            time_to_closest_approach_s=(
                assessment.time_to_closest_approach_s if assessment is not None else None
            ),
        )

    def _desired_action(
        self,
        risk_results: tuple[PerceptionRiskResult, ...],
    ) -> tuple[AlertAction, PerceptionRiskResult | None, str]:
        valid = [
            result
            for result in risk_results
            if result.status is PerceptionRiskStatus.VALID and result.assessment is not None
        ]
        uncertain = any(
            result.status is PerceptionRiskStatus.EXCESSIVE_UNCERTAINTY for result in risk_results
        )

        if not valid:
            if risk_results:
                return AlertAction.ABSTAIN, None, "insufficient trustworthy motion for risk scoring"
            return AlertAction.NO_ALERT, None, "no tracked hazards"

        selected = max(
            valid,
            key=lambda result: (
                bool(result.assessment and result.assessment.predicted_collision),
                result.assessment.risk_score if result.assessment else 0.0,
                -(result.assessment.time_to_closest_approach_s if result.assessment else 1e9),
            ),
        )
        assessment = selected.assessment
        if assessment is None:
            raise RuntimeError("valid perception risk result is missing an assessment")

        if assessment.predicted_collision and (
            assessment.risk_score >= self.config.stop_score
            or assessment.time_to_closest_approach_s <= self.config.stop_time_to_collision_s
        ):
            return AlertAction.STOP, selected, "predicted collision requires immediate stop"

        if uncertain and assessment.risk_score < self.config.slow_score:
            return AlertAction.ABSTAIN, selected, "another active track has excessive uncertainty"

        if assessment.closing_speed_mps > 0 and (
            assessment.risk_score >= self.config.slow_score
            or assessment.time_to_closest_approach_s <= self.config.slow_time_to_collision_s
        ):
            return AlertAction.SLOW, selected, "approaching hazard requires reduced speed"

        if assessment.risk_score >= self.config.awareness_score:
            x_m = assessment.relative_position_m.x
            if x_m < -self.config.direction_deadband_m:
                action = AlertAction.AWARENESS_LEFT
            elif x_m > self.config.direction_deadband_m:
                action = AlertAction.AWARENESS_RIGHT
            else:
                action = AlertAction.AWARENESS_CENTER
            return action, selected, "hazard awareness threshold exceeded"

        return AlertAction.NO_ALERT, selected, "risk remains below the awareness threshold"

    def _stabilize(self, desired: AlertAction, timestamp_s: float) -> AlertAction:
        current = self._current_action
        if desired is current:
            return current

        current_severity = _severity(current)
        desired_severity = _severity(desired)
        held_s = timestamp_s - self._current_since_s

        immediate_change = (
            desired_severity > current_severity
            or desired is AlertAction.STOP
            or desired is AlertAction.ABSTAIN
            and current is AlertAction.NO_ALERT
        )
        if immediate_change or held_s >= self.config.minimum_hold_s:
            self._current_action = desired
            self._current_since_s = timestamp_s
            return desired
        return current

    def _should_emit(self, action: AlertAction, timestamp_s: float) -> bool:
        if action is AlertAction.NO_ALERT:
            return False
        if self._last_emitted_s is None:
            self._last_emitted_s = timestamp_s
            return True
        action_just_changed = timestamp_s == self._current_since_s
        repeat_due = timestamp_s - self._last_emitted_s >= self.config.repeat_interval_s
        if action_just_changed or repeat_due:
            self._last_emitted_s = timestamp_s
            return True
        return False


def _severity(action: AlertAction) -> int:
    if action is AlertAction.STOP:
        return 4
    if action is AlertAction.SLOW:
        return 3
    if action in {
        AlertAction.AWARENESS_LEFT,
        AlertAction.AWARENESS_CENTER,
        AlertAction.AWARENESS_RIGHT,
    }:
        return 2
    if action is AlertAction.ABSTAIN:
        return 1
    return 0
