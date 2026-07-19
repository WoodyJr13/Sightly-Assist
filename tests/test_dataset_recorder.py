from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from sightly_assist.capture import CaptureSourceMetadata, ImuSample, RgbdCaptureFrame
from sightly_assist.dataset_recorder import (
    DatasetRecorderConfig,
    record_rgbd_dataset,
    verify_dataset_checksums,
)
from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.replay import load_manifest, validate_replay_files


class FakeCaptureSource:
    def __init__(self, frames: tuple[RgbdCaptureFrame, ...]) -> None:
        self._frames = frames
        self._intrinsics = CameraIntrinsics(
            width_px=6,
            height_px=4,
            fx_px=5.0,
            fy_px=5.0,
            cx_px=2.5,
            cy_px=1.5,
        )
        self._metadata = CaptureSourceMetadata(
            source_name="fake-rgbd",
            device_id="fake-001",
            device_model="synthetic",
            width_px=6,
            height_px=4,
            requested_fps=10.0,
            imu_report_rate_hz=100,
            orientation_available=True,
        )

    @property
    def source_name(self) -> str:
        return "fake-rgbd"

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @property
    def metadata(self) -> CaptureSourceMetadata:
        return self._metadata

    def __iter__(self) -> Iterator[RgbdCaptureFrame]:
        yield from self._frames

    def close(self) -> None:
        return None


def _frame(frame_id: int, timestamp_ns: int, sync_error_ns: int = 1_000_000) -> RgbdCaptureFrame:
    orientation = (1.0, 0.0, 0.0, 0.0)
    imu = ImuSample(
        timestamp_ns=timestamp_ns,
        sequence_num=frame_id,
        acceleration_mps2=(0.0, 9.81, 0.0),
        angular_velocity_radps=(0.0, 0.0, 0.0),
        orientation_wxyz=orientation,
        orientation_accuracy_rad=0.02,
    )
    return RgbdCaptureFrame(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        rgb_bgr=np.full((4, 6, 3), frame_id * 20, dtype=np.uint8),
        depth_m=np.full((4, 6), 2.0 - 0.1 * frame_id, dtype=np.float32),
        imu_samples=(imu,),
        orientation_wxyz=orientation,
        device_sequence_num=frame_id + 100,
        sync_error_ns=sync_error_ns,
    )


def test_recorder_exports_replay_dataset_with_integrity_metadata(tmp_path: Path) -> None:
    source = FakeCaptureSource(
        (
            _frame(0, 1_000_000_000),
            _frame(1, 1_100_000_000),
            _frame(2, 1_200_000_000, sync_error_ns=25_000_000),
        )
    )
    output = tmp_path / "recording"

    summary = record_rgbd_dataset(
        source,
        output,
        DatasetRecorderConfig(
            sequence_id="unit-rgbd",
            maximum_frames=3,
            maximum_sync_error_ns=20_000_000,
        ),
    )

    manifest = load_manifest(output / "manifest.json")
    assert summary.frame_count == 3
    assert summary.imu_sample_count == 3
    assert summary.synchronized_frame_count == 2
    assert summary.duration_s == pytest.approx(0.2)
    assert manifest.sequence_id == "unit-rgbd"
    assert manifest.intrinsics == source.intrinsics
    assert manifest.imu_path == "imu/samples.jsonl"
    assert validate_replay_files(manifest, output) == []
    assert verify_dataset_checksums(manifest, output) == []
    assert manifest.frames[-1].synchronized is False
    assert manifest.frames[0].device_sequence_num == 100

    depth = np.load(output / manifest.frames[1].depth_path, allow_pickle=False)
    assert depth.dtype == np.float32
    assert depth.shape == (4, 6)
    assert depth[0, 0] == pytest.approx(1.9)
    imu_lines = (output / "imu" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(imu_lines) == 3


def test_checksum_verifier_detects_tampered_frame(tmp_path: Path) -> None:
    output = tmp_path / "recording"
    record_rgbd_dataset(
        FakeCaptureSource((_frame(0, 1_000_000_000),)),
        output,
        DatasetRecorderConfig(sequence_id="tamper", maximum_frames=1),
    )
    manifest = load_manifest(output / "manifest.json")
    rgb_path = output / manifest.frames[0].rgb_path
    rgb_path.write_bytes(rgb_path.read_bytes() + b"tampered")

    assert verify_dataset_checksums(manifest, output) == [manifest.frames[0].rgb_path]


def test_recorder_rejects_nonchronological_source_without_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "recording"
    source = FakeCaptureSource(
        (
            _frame(1, 1_100_000_000),
            _frame(0, 1_200_000_000),
        )
    )

    with pytest.raises(ValueError, match="frame IDs"):
        record_rgbd_dataset(
            source,
            output,
            DatasetRecorderConfig(sequence_id="bad-order", maximum_frames=2),
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".recording.partial-*"))


def test_recorder_requires_a_capture_limit() -> None:
    with pytest.raises(ValueError, match="frame or duration limit"):
        DatasetRecorderConfig(sequence_id="unbounded")


def test_imu_sample_rejects_empty_measurement() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        ImuSample(timestamp_ns=0)
