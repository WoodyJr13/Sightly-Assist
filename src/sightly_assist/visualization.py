"""Top-down plots for deterministic simulation runs."""

from __future__ import annotations

from pathlib import Path

from sightly_assist.geometry import position_at
from sightly_assist.schemas import ScenarioConfig, SimulationStep


def plot_top_down(
    scenario: ScenarioConfig,
    steps: list[SimulationStep],
    output_path: Path,
) -> Path:
    """Render observer and obstacle trajectories to a PNG file.

    Matplotlib is imported lazily so core simulation remains usable without the
    optional visualization dependency.
    """

    if not steps:
        raise ValueError("steps must not be empty")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Visualization requires the optional dependency: "
            "python -m pip install -e '.[visualization]'"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = [step.timestamp_s for step in steps]

    figure, axis = plt.subplots(figsize=(7, 7))

    observer_x = [position_at(scenario.observer, time).x for time in times]
    observer_z = [position_at(scenario.observer, time).z for time in times]
    axis.plot(observer_x, observer_z, linewidth=2.5, label="observer")
    axis.scatter(observer_x[0], observer_z[0], marker="o", s=70)

    for obstacle in scenario.obstacles:
        obstacle_x = [position_at(obstacle, time).x for time in times]
        obstacle_z = [position_at(obstacle, time).z for time in times]
        axis.plot(obstacle_x, obstacle_z, linewidth=2, label=obstacle.id)
        axis.scatter(obstacle_x[0], obstacle_z[0], marker="x", s=70)

        boundary = obstacle.radius_m + scenario.observer.radius_m + scenario.safety_margin_m
        axis.add_patch(
            Circle(
                (obstacle_x[-1], obstacle_z[-1]),
                radius=boundary,
                fill=False,
                linestyle="--",
                alpha=0.35,
            )
        )

    axis.set_title(f"Sightly Assist — {scenario.scenario_id}")
    axis.set_xlabel("Lateral position x (m)")
    axis.set_ylabel("Forward position z (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
