from __future__ import annotations

import json

import pytest

from sightly_assist.bytetrack_adapter import ByteTrackAdapter, ByteTrackConfig
from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.tracking_benchmark import benchmark_trackers
from sightly_assist.tracking_report import export_locked_tracking_benchmark
from sightly_assist.tracking_scenarios import locked_tracking_sequences


def test_bytetrack_config_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="high_conf_det_threshold"):
        ByteTrackConfig(
            track_activation_threshold=0.8,
            high_conf_det_threshold=0.6,
        )


def test_real_bytetrack_preserves_id_on_smooth_approach() -> None:
    sequence = locked_tracking_sequences()[0]
    tracker = ByteTrackAdapter()
    observed_ids: list[int] = []

    for item in sequence.frames:
        tracks = tracker.update(item.frame, item.detections)
        assert len(tracks) == 1
        observed_ids.append(tracks[0].track_id)

    assert len(set(observed_ids)) == 1


def test_locked_benchmark_runs_real_bytetrack_and_iou() -> None:
    results = benchmark_trackers(
        {
            "bytetrack-2.5.0": ByteTrackAdapter,
            "iou-baseline": IoUTracker,
        },
        locked_tracking_sequences(),
    )
    by_name = {result.tracker_name: result.metrics for result in results}

    assert set(by_name) == {"bytetrack-2.5.0", "iou-baseline"}
    for metrics in by_name.values():
        assert metrics.sequences == 3
        assert metrics.frames == 34
        assert metrics.ground_truth_observations == 46
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert metrics.latency_p95_ms >= metrics.latency_median_ms
        assert metrics.risk_eligible_observations > 0
        assert 0.0 <= metrics.collision_f1 <= 1.0

    bytetrack = by_name["bytetrack-2.5.0"]
    baseline = by_name["iou-baseline"]
    assert bytetrack.fragmentations <= baseline.fragmentations
    assert bytetrack.identity_switches <= baseline.identity_switches
    assert bytetrack.collision_recall >= baseline.collision_recall


def test_tracking_benchmark_exports_json(tmp_path) -> None:
    output_path = tmp_path / "tracking.json"
    results = export_locked_tracking_benchmark(output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(results) == 2
    assert payload["benchmark"] == "locked-synthetic-tracking-v1"
    assert {item["tracker_name"] for item in payload["results"]} == {
        "bytetrack-2.5.0",
        "iou-baseline",
    }
