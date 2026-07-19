"""Command-line interface for Sightly Assist."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated, cast

import typer

from sightly_assist.dataset_recorder import DatasetRecorderConfig, record_rgbd_dataset
from sightly_assist.oakd_source import OakDConfig, OakDSource
from sightly_assist.simulation import export_run, load_scenario, run_scenario, summarize_run
from sightly_assist.tracking_report import export_locked_tracking_benchmark
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


@app.command("benchmark-tracking")
def benchmark_tracking(
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="JSON benchmark report path."),
    ] = Path("reports/benchmarks/tracking.json"),
) -> None:
    """Compare the IoU baseline and ByteTrack on locked synthetic sequences."""

    results = export_locked_tracking_benchmark(output_path)
    typer.echo(f"Report: {output_path}")
    for result in results:
        metrics = result.metrics
        typer.echo(
            f"{result.tracker_name}: recall={metrics.recall:.3f} "
            f"IDSW={metrics.identity_switches} fragments={metrics.fragmentations} "
            f"risk_F1={metrics.collision_f1:.3f} "
            f"p95={metrics.latency_p95_ms:.3f} ms"
        )


@app.command("record-oakd")
def record_oakd(
    sequence_id: Annotated[
        str,
        typer.Option("--sequence-id", help="Stable identifier for this recording."),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination dataset directory."),
    ] = Path("datasets/oakd/latest"),
    duration_s: Annotated[
        float,
        typer.Option("--duration", help="Maximum recording duration in seconds."),
    ] = 10.0,
    maximum_frames: Annotated[
        int | None,
        typer.Option("--max-frames", help="Optional additional frame-count limit."),
    ] = None,
    width_px: Annotated[int, typer.Option("--width")] = 640,
    height_px: Annotated[int, typer.Option("--height")] = 400,
    fps: Annotated[float, typer.Option("--fps")] = 30.0,
    orientation: Annotated[
        bool,
        typer.Option(
            "--orientation/--no-orientation",
            help="Request a fused IMU orientation when the device supports it.",
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing destination directory."),
    ] = False,
) -> None:
    """Record synchronized OAK-D RGB, metric depth, and IMU data for replay."""

    source_config = OakDConfig(
        width_px=width_px,
        height_px=height_px,
        fps=fps,
        enable_orientation=orientation,
    )
    recorder_config = DatasetRecorderConfig(
        sequence_id=sequence_id,
        maximum_frames=maximum_frames,
        maximum_duration_s=duration_s,
        overwrite=overwrite,
    )
    with OakDSource(source_config) as source:
        summary = record_rgbd_dataset(source, output_directory, recorder_config)

    typer.echo(f"Sequence: {summary.sequence_id}")
    typer.echo(f"Frames: {summary.frame_count}")
    typer.echo(f"IMU samples: {summary.imu_sample_count}")
    typer.echo(f"Duration: {summary.duration_s:.3f} s")
    typer.echo("Synchronized frames: " f"{summary.synchronized_frame_count}/{summary.frame_count}")
    typer.echo(f"Manifest: {summary.manifest_path}")


if __name__ == "__main__":
    app()
