"""Run and export the locked IoU-versus-ByteTrack benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from sightly_assist.bytetrack_adapter import ByteTrackAdapter
from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.tracking_benchmark import TrackingBenchmarkResult, benchmark_trackers
from sightly_assist.tracking_scenarios import locked_tracking_sequences


BYTETRACK_REPORT_NAME = "bytetrack-supervision-0.27.0"


def run_locked_tracking_benchmark() -> tuple[TrackingBenchmarkResult, ...]:
    """Benchmark the transparent baseline and real ByteTrack on identical data."""

    return benchmark_trackers(
        {
            BYTETRACK_REPORT_NAME: ByteTrackAdapter,
            "iou-baseline": IoUTracker,
        },
        locked_tracking_sequences(),
    )


def export_locked_tracking_benchmark(output_path: Path) -> tuple[TrackingBenchmarkResult, ...]:
    """Run the benchmark and write a stable JSON report."""

    results = run_locked_tracking_benchmark()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "locked-synthetic-tracking-v1",
        "results": [result.model_dump(mode="json") for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results
