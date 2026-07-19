"""Tests for simulation visualizations."""

from pathlib import Path

from sightly_assist.simulation import load_scenario, run_scenario
from sightly_assist.visualization import plot_top_down


def test_top_down_plot_is_created(tmp_path: Path) -> None:
    scenario = load_scenario(Path("scenarios/crossing.yaml"))
    steps = run_scenario(scenario)
    output_path = tmp_path / "top_down.png"

    returned_path = plot_top_down(scenario, steps, output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
