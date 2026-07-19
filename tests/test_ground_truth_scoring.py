from __future__ import annotations

import json
from pathlib import Path

import pytest

from sightly_assist.ground_truth import (
    GroundTruthAnnotationSet,
    GroundTruthEvent,
    GroundTruthFrame,
    GroundTruthObject,
    create_annotation_template,
    load_ground_truth,
    save_ground_truth,
    validate_annotations_against_manifest,
)
from sightly_assist.ground_truth_scoring import (
    ScoredReplayFrame,
    load_scored_replay,
    score_evaluation_files,
    score_ground_truth,
)
from sightly_assist.perception import BoundingBox, FramePacket, TrackObservation
from sightly_assist.replay import ReplayManifest
from sightly_assist.warning_policy import AlertAction, WarningDecision


BOX = BoundingBox(x_min=10, y_min=10, x_max=30, y_max=50)


def _frame(frame_id: int, timestamp_ns: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        width_px=64,
        height_px=64,
        rgb_path=f"rgb/{frame_id:06d}.png",
        depth_path=f"depth/{frame_id:06d}.npy",
    )


def _track(frame_id: int, track_id: int) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        frame_id=frame_id,
        class_name="person",
        confidence=0.95,
        bbox=BOX,
        age_frames=frame_id + 1,
        missed_frames=0,
    )


def _object() -> GroundTruthObject:
    return GroundTruthObject(
        object_id="person-a",
        class_name="person",
        bbox=BOX,
    )


def test_annotation_template_round_trip_and_manifest_alignment(tmp_path: Path) -> None:
    manifest = ReplayManifest(
        sequence_id="template-sequence",
        source="unit-test",
        frames=[_frame(0, 0), _frame(1, 100_000_000)],
    )
    annotations = create_annotation_template(manifest, "annotator-1")
    output = tmp_path / "annotations.json"

    save_ground_truth(annotations, output)
    loaded = load_ground_truth(output)
    validate_annotations_against_manifest(loaded, manifest)

    assert loaded.sequence_id == manifest.sequence_id
    assert loaded.annotator == "annotator-1"
    assert len(loaded.frames) == 2
    assert loaded.frames[0].objects == ()


def test_annotations_reject_unknown_event_object() -> None:
    with pytest.raises(ValueError, match="unknown object"):
        GroundTruthAnnotationSet(
            sequence_id="bad-event",
            annotator="tester",
            frames=(GroundTruthFrame(frame_id=0, timestamp_ns=0, objects=(_object(),)),),
            events=(
                GroundTruthEvent(
                    event_id="event-1",
                    object_id="not-present",
                    start_timestamp_ns=0,
                    critical_timestamp_ns=0,
                    end_timestamp_ns=0,
                ),
            ),
        )


def test_identity_switch_fragmentation_warning_lead_and_false_warning_rate() -> None:
    timestamps = (0, 1_000_000_000, 2_000_000_000, 3_000_000_000)
    annotations = GroundTruthAnnotationSet(
        sequence_id="identity-and-warning",
        annotator="tester",
        frames=tuple(
            GroundTruthFrame(
                frame_id=frame_id,
                timestamp_ns=timestamp,
                objects=(_object(),),
            )
            for frame_id, timestamp in enumerate(timestamps)
        ),
        events=(
            GroundTruthEvent(
                event_id="approach",
                object_id="person-a",
                start_timestamp_ns=1_000_000_000,
                critical_timestamp_ns=2_000_000_000,
                end_timestamp_ns=3_000_000_000,
                minimum_warning_action=AlertAction.SLOW,
            ),
        ),
    )
    results = (
        ScoredReplayFrame(
            frame=_frame(0, timestamps[0]),
            tracks=(_track(0, 1),),
            warning_decision=WarningDecision(
                timestamp_s=0,
                action=AlertAction.STOP,
                should_emit=True,
                reason="synthetic false warning",
                track_id=1,
            ),
        ),
        ScoredReplayFrame(
            frame=_frame(1, timestamps[1]),
            tracks=(),
            warning_decision=WarningDecision(
                timestamp_s=1,
                action=AlertAction.NO_ALERT,
                should_emit=False,
                reason="track temporarily missing",
            ),
        ),
        ScoredReplayFrame(
            frame=_frame(2, timestamps[2]),
            tracks=(_track(2, 2),),
            warning_decision=WarningDecision(
                timestamp_s=2,
                action=AlertAction.SLOW,
                should_emit=True,
                reason="synthetic valid warning",
                track_id=2,
            ),
        ),
        ScoredReplayFrame(
            frame=_frame(3, timestamps[3]),
            tracks=(_track(3, 2),),
            warning_decision=WarningDecision(
                timestamp_s=3,
                action=AlertAction.NO_ALERT,
                should_emit=False,
                reason="event complete",
            ),
        ),
    )

    report = score_ground_truth(annotations, results)

    assert report.track_recall == pytest.approx(0.75)
    assert report.track_precision == pytest.approx(1.0)
    assert report.identity_switches == 1
    assert report.fragmentations == 1
    assert report.warning_event_recall == pytest.approx(1.0)
    assert report.warning_lead_time_s is not None
    assert report.warning_lead_time_s.mean_error == pytest.approx(0.0)
    assert report.false_warning_count == 1
    assert report.false_warnings_per_minute == pytest.approx(20.0)


def test_score_files_export_hashes_and_reject_misaligned_results(tmp_path: Path) -> None:
    annotations = GroundTruthAnnotationSet(
        sequence_id="file-score",
        annotator="tester",
        frames=(GroundTruthFrame(frame_id=0, timestamp_ns=0, objects=(_object(),)),),
    )
    annotations_path = save_ground_truth(annotations, tmp_path / "annotations.json")
    results_path = tmp_path / "frame_results.jsonl"
    result = ScoredReplayFrame(frame=_frame(0, 0), tracks=(_track(0, 7),))
    results_path.write_text(
        json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "score.json"

    report = score_evaluation_files(annotations_path, results_path, output_path)

    assert report.annotations_sha256 is not None
    assert report.results_sha256 is not None
    assert report.track_recall == pytest.approx(1.0)
    assert output_path.is_file()
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["sequence_id"] == "file-score"

    bad_path = tmp_path / "bad-results.jsonl"
    bad_result = ScoredReplayFrame(frame=_frame(1, 1), tracks=())
    bad_path.write_text(
        json.dumps(bad_result.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    loaded = load_scored_replay(bad_path)
    with pytest.raises(ValueError, match="frame IDs"):
        score_ground_truth(annotations, loaded)
