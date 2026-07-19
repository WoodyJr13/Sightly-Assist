from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.iou_tracker import IoUTracker
from sightly_assist.perception import BoundingBox, Detection, FramePacket
from sightly_assist.replay import ReplayManifest
from sightly_assist.replay_evaluation import ReplayEvaluationConfig, evaluate_replay


class ApproachingPersonDetector:
    def predict(self, frame: FramePacket, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        del image_bgr
        return (
            Detection(
                frame_id=frame.frame_id,
                class_id=0,
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(x_min=4, y_min=2, x_max=16, y_max=14),
            ),
        )


def _manifest(root: Path, *, hashes: bool = False) -> ReplayManifest:
    frames: list[FramePacket] = []
    for frame_id in range(7):
        rgb_path = root / f"rgb-{frame_id}.bin"
        depth_path = root / f"depth-{frame_id}.bin"
        rgb_path.write_bytes(b"rgb")
        depth_path.write_bytes(b"depth")
        frames.append(
            FramePacket(
                frame_id=frame_id,
                timestamp_ns=frame_id * 100_000_000,
                width_px=20,
                height_px=16,
                rgb_path=rgb_path.name,
                depth_path=depth_path.name,
                rgb_sha256="0" * 64 if hashes else None,
                depth_sha256="0" * 64 if hashes else None,
            )
        )
    return ReplayManifest(
        sequence_id="approaching-person",
        source="unit-test",
        frames=frames,
        intrinsics=CameraIntrinsics(
            width_px=20,
            height_px=16,
            fx_px=18,
            fy_px=18,
            cx_px=9.5,
            cy_px=7.5,
        ),
    )


def _image_loader(frame: FramePacket, root: Path) -> np.ndarray:
    del root
    return np.zeros((frame.height_px, frame.width_px, 3), dtype=np.uint8)


def _depth_loader(frame: FramePacket, root: Path) -> np.ndarray:
    del root
    depth_m = 5.0 - 0.4 * frame.frame_id
    return np.full((frame.height_px, frame.width_px), depth_m, dtype=np.float32)


def test_evaluation_exports_results_metrics_and_audio(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = _manifest(root)
    output = tmp_path / "evaluation"

    summary = evaluate_replay(
        manifest,
        root,
        ApproachingPersonDetector(),
        IoUTracker(minimum_iou=0.1),
        output,
        ReplayEvaluationConfig(
            verify_checksums=False,
            render_video=False,
            render_audio=True,
        ),
        model_sha256="a" * 64,
        image_loader=_image_loader,
        depth_loader=_depth_loader,
    )

    assert summary.frame_count == 7
    assert summary.detection_count == 7
    assert summary.track_observation_count == 7
    assert summary.valid_depth_association_count == 7
    assert summary.valid_motion_estimate_count >= 5
    assert summary.valid_risk_result_count >= 5
    assert summary.predicted_collision_frame_count >= 1
    assert summary.emitted_warning_count >= 1
    assert summary.stage_latencies["total_ms"].maximum_ms >= 0
    assert summary.mean_processing_fps >= 0
    assert summary.model_sha256 == "a" * 64
    assert summary.annotated_video_path is None
    assert summary.audio_directory == str(output.resolve() / "audio")

    frame_lines = (output / "frame_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(frame_lines) == 7
    last_result = json.loads(frame_lines[-1])
    assert last_result["stage_timings"]["total_ms"] >= 0
    assert last_result["risk_results"]
    assert (output / "summary.json").is_file()
    assert (output / "audio" / "audio_manifest.json").is_file()


def test_evaluation_rejects_checksum_mismatch_before_processing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = _manifest(root, hashes=True)

    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate_replay(
            manifest,
            root,
            ApproachingPersonDetector(),
            IoUTracker(),
            tmp_path / "evaluation",
            ReplayEvaluationConfig(render_video=False, render_audio=False),
            image_loader=_image_loader,
            depth_loader=_depth_loader,
        )


def test_evaluation_requires_intrinsics(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = _manifest(root).model_copy(update={"intrinsics": None})

    with pytest.raises(ValueError, match="camera intrinsics"):
        evaluate_replay(
            manifest,
            root,
            ApproachingPersonDetector(),
            IoUTracker(),
            tmp_path / "evaluation",
            ReplayEvaluationConfig(
                verify_checksums=False,
                render_video=False,
                render_audio=False,
            ),
            image_loader=_image_loader,
            depth_loader=_depth_loader,
        )
