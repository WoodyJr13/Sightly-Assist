"""Regression tests for the controlled scenario suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from sightly_assist.simulation import export_run, load_scenario, run_scenario, summarize_run

SCENARIO_DIRECTORY = Path("scenarios")


@pytest.mark.parametrize(
    "scenario_name",
    [
        "static_ahead.yaml",
        "static_beside.yaml",
        "approaching_head_on.yaml",
        "receding.yaml",
        "crossing.yaml",
        "two_object_conflict.yaml",
    ],
)
def test_scenario_expectation(scenario_name: str) -> None:
    scenario = load_scenario(SCENARIO_DIRECTORY / scenario_name)
    steps = run_scenario(scenario)
    summary = summarize_run(scenario, steps)

    assert summary["expectation_met"] is True
    assert summary["step_count"] > 1


def test_export_creates_machine_readable_outputs(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIO_DIRECTORY / "crossing.yaml")
    steps = run_scenario(scenario)

    summary_path, timeline_path = export_run(scenario, steps, tmp_path)

    assert summary_path.exists()
    assert timeline_path.exists()
    assert '"scenario_id": "crossing"' in summary_path.read_text(encoding="utf-8")
    assert "time_to_closest_approach_s" in timeline_path.read_text(encoding="utf-8")
