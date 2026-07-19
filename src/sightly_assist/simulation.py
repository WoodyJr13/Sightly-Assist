"""Deterministic scenario loading, execution, and export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import yaml

from sightly_assist.geometry import position_at
from sightly_assist.risk import assess_risk
from sightly_assist.schemas import ScenarioConfig, SimulationStep


def load_scenario(path: Path) -> ScenarioConfig:
    """Load and validate a YAML simulation scenario."""

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return ScenarioConfig.model_validate(raw)


def run_scenario(scenario: ScenarioConfig) -> list[SimulationStep]:
    """Run a deterministic constant-velocity scenario."""

    steps: list[SimulationStep] = []
    step_count = int(scenario.duration_s / scenario.time_step_s) + 1

    for index in range(step_count):
        timestamp_s = min(index * scenario.time_step_s, scenario.duration_s)
        assessments = [
            assess_risk(
                observer=scenario.observer,
                obstacle=obstacle,
                timestamp_s=timestamp_s,
                prediction_horizon_s=scenario.prediction_horizon_s,
                safety_margin_m=scenario.safety_margin_m,
            )
            for obstacle in scenario.obstacles
        ]
        steps.append(
            SimulationStep(
                timestamp_s=timestamp_s,
                observer_position_m=position_at(scenario.observer, timestamp_s),
                assessments=assessments,
            )
        )

    return steps


def summarize_run(scenario: ScenarioConfig, steps: Iterable[SimulationStep]) -> dict[str, object]:
    """Create a compact machine-readable run summary."""

    materialized = list(steps)
    assessments = [assessment for step in materialized for assessment in step.assessments]
    peak = max(assessments, key=lambda item: item.risk_score)
    predicted_hazards = sorted(
        {item.obstacle_id for item in assessments if item.predicted_collision}
    )
    expectation_met = scenario.expected.hazard == bool(predicted_hazards)
    if scenario.expected.primary_obstacle_id is not None and predicted_hazards:
        expectation_met = expectation_met and peak.obstacle_id == scenario.expected.primary_obstacle_id

    return {
        "scenario_id": scenario.scenario_id,
        "duration_s": scenario.duration_s,
        "time_step_s": scenario.time_step_s,
        "step_count": len(materialized),
        "predicted_hazard_obstacle_ids": predicted_hazards,
        "peak_risk_obstacle_id": peak.obstacle_id,
        "peak_risk_score": peak.risk_score,
        "peak_risk_level": peak.risk_level.value,
        "expected_hazard": scenario.expected.hazard,
        "expectation_met": expectation_met,
    }


def export_run(
    scenario: ScenarioConfig,
    steps: list[SimulationStep],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Export summary JSON and per-obstacle timeline CSV."""

    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    timeline_path = output_directory / "timeline.csv"

    summary_path.write_text(
        json.dumps(summarize_run(scenario, steps), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with timeline_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "timestamp_s",
                "obstacle_id",
                "relative_x_m",
                "relative_z_m",
                "relative_vx_mps",
                "relative_vz_mps",
                "time_to_closest_approach_s",
                "distance_at_closest_approach_m",
                "closing_speed_mps",
                "predicted_collision",
                "risk_score",
                "risk_level",
            ],
        )
        writer.writeheader()
        for step in steps:
            for item in step.assessments:
                writer.writerow(
                    {
                        "timestamp_s": item.timestamp_s,
                        "obstacle_id": item.obstacle_id,
                        "relative_x_m": item.relative_position_m.x,
                        "relative_z_m": item.relative_position_m.z,
                        "relative_vx_mps": item.relative_velocity_mps.x,
                        "relative_vz_mps": item.relative_velocity_mps.z,
                        "time_to_closest_approach_s": item.time_to_closest_approach_s,
                        "distance_at_closest_approach_m": item.distance_at_closest_approach_m,
                        "closing_speed_mps": item.closing_speed_mps,
                        "predicted_collision": item.predicted_collision,
                        "risk_score": item.risk_score,
                        "risk_level": item.risk_level.value,
                    }
                )

    return summary_path, timeline_path
