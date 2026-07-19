"""Command-line interface for Sightly Assist."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import typer

from sightly_assist.simulation import export_run, load_scenario, run_scenario, summarize_run

app = typer.Typer(no_args_is_help=True, help="Sightly Assist research tools.")


@app.command()
def doctor() -> None:
    """Print the local development environment."""

    typer.echo(f"Python: {sys.version.split()[0]}")
    typer.echo(f"Platform: {platform.platform()}")
    typer.echo("Core package import: OK")


@app.command()
def simulate(
    scenario_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_directory: Path = typer.Option(Path("reports/runs/latest"), "--output", "-o"),
) -> None:
    """Run one deterministic YAML scenario and export its results."""

    scenario = load_scenario(scenario_path)
    steps = run_scenario(scenario)
    summary_path, timeline_path = export_run(scenario, steps, output_directory)
    summary = summarize_run(scenario, steps)

    typer.echo(f"Scenario: {scenario.scenario_id}")
    typer.echo(f"Expectation met: {summary['expectation_met']}")
    typer.echo(f"Peak risk: {summary['peak_risk_score']:.3f}")
    typer.echo(f"Summary: {summary_path}")
    typer.echo(f"Timeline: {timeline_path}")

    if not bool(summary["expectation_met"]):
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
