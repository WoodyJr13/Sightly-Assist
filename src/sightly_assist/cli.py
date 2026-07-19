"""Command-line interface for Sightly Assist."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated, cast

import typer

from sightly_assist.simulation import export_run, load_scenario, run_scenario, summarize_run
from sightly_assist.visualization import plot_top_down

app = typer.Typer(no_args_is_help=True, help="Sightly Assist research tools.")


@app.command()
def doctor() -> None:
    """Print the local development environment."""

    typer.echo(f"Python: {sys.version.split()[0]}")
    typer.echo(f"Platform: {platform.platform()}")
    typer.echo("Core package import: OK")


@app.command()
def simulate(
    scenario_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output", "-o"),
    ] = Path("reports/runs/latest"),
    plot: Annotated[
        bool,
        typer.Option("--plot/--no-plot", help="Export a top-down trajectory PNG."),
    ] = True,
) -> None:
    """Run one deterministic YAML scenario and export its results."""

    scenario = load_scenario(scenario_path)
    steps = run_scenario(scenario)
    summary_path, timeline_path = export_run(scenario, steps, output_directory)
    plot_path: Path | None = None
    if plot:
        plot_path = plot_top_down(scenario, steps, output_directory / "top_down.png")

    summary = summarize_run(scenario, steps)
    expectation_met = bool(summary["expectation_met"])
    peak_risk_score = cast(float, summary["peak_risk_score"])

    typer.echo(f"Scenario: {scenario.scenario_id}")
    typer.echo(f"Expectation met: {expectation_met}")
    typer.echo(f"Peak risk: {peak_risk_score:.3f}")
    typer.echo(f"Summary: {summary_path}")
    typer.echo(f"Timeline: {timeline_path}")
    if plot_path is not None:
        typer.echo(f"Plot: {plot_path}")

    if not expectation_met:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
