from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sightly_assist.depth_association import CameraIntrinsics
from sightly_assist.free_space import (
    CorridorDirection,
    CorridorStatus,
    FreeSpaceConfig,
    FreeSpaceContext,
    analyze_free_space,
)
from sightly_assist.perception import (
    Detection,
    EmptyDetector,
    FramePacket,
    TrackObservation,
)
from sightly_assist.replay import ReplayManifest
from sightly_assist.replay_pipeline import process_replay


class _EmptyTracker:
    def update(
        self,
        frame: FramePacket,
        detections: Sequence[Detection],
    ) -> Sequence[TrackObservation]:
        del frame, detections
        return ()


def _frame() -> FramePacket:
    return FramePacket(
        frame_id=7,
        timestamp_ns=500_000_000,
        width_px=90,
        height_px=60,
        rgb_path="rgb.png",
        depth_path="depth.npy",
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=90,
        height_px=60,
        fx_px=60.0,
        fy_px=60.0,
        cx_px=44.5,
        cy_px=29.5,
    )


def _config(**changes: float | int) -> FreeSpaceConfig:
    return FreeSpaceConfig(
        minimum_valid_samples=10,
        minimum_valid_fraction=0.10,
        **changes,
    )


def test_clear_depth_marks_all_corridors_clear() -> None:
    analysis = analyze_free_space(
        np.full((60, 90), 5.0),
        _frame(),
        _intrinsics(),
        _config(),
    )

    assert analysis.context is FreeSpaceContext.CENTER_CLEAR
    assert all(corridor.status is CorridorStatus.CLEAR for corridor in analysis.corridors)


def test_near_center_region_is_blocked_without_blocking_sides() -> None:
    depth = np.full((60, 90), 5.0)
    depth[:, 33:57] = 1.0

    analysis = analyze_free_space(depth, _frame(), _intrinsics(), _config())

    assert analysis.context is FreeSpaceContext.CENTER_BLOCKED
    assert analysis.corridor(CorridorDirection.CENTER).status is CorridorStatus.BLOCKED
    assert analysis.corridor(CorridorDirection.LEFT).status is CorridorStatus.CLEAR
    assert analysis.corridor(CorridorDirection.RIGHT).status is CorridorStatus.CLEAR


def test_midrange_center_region_is_constrained() -> None:
    depth = np.full((60, 90), 5.0)
    depth[:, 33:57] = 2.2

    analysis = analyze_free_space(depth, _frame(), _intrinsics(), _config())

    center = analysis.corridor(CorridorDirection.CENTER)
    assert analysis.context is FreeSpaceContext.CENTER_CONSTRAINED
    assert center.status is CorridorStatus.CONSTRAINED
    assert center.near_depth_m == pytest.approx(2.2)


def test_missing_center_depth_is_unknown_not_clear() -> None:
    depth = np.full((60, 90), 5.0)
    depth[:, 33:57] = 0.0

    analysis = analyze_free_space(depth, _frame(), _intrinsics(), _config())

    center = analysis.corridor(CorridorDirection.CENTER)
    assert analysis.context is FreeSpaceContext.UNKNOWN
    assert center.status is CorridorStatus.UNKNOWN
    assert center.valid_fraction == 0.0


def test_none_depth_marks_every_corridor_unknown() -> None:
    analysis = analyze_free_space(None, _frame(), _intrinsics(), _config())

    assert analysis.context is FreeSpaceContext.UNKNOWN
    assert all(corridor.status is CorridorStatus.UNKNOWN for corridor in analysis.corridors)


def test_depth_scale_converts_millimeters() -> None:
    analysis = analyze_free_space(
        np.full((60, 90), 5000.0),
        _frame(),
        _intrinsics(),
        _config(depth_scale_m=0.001),
    )

    center = analysis.corridor(CorridorDirection.CENTER)
    assert center.status is CorridorStatus.CLEAR
    assert center.median_depth_m == pytest.approx(5.0)


def test_depth_dimensions_must_match_frame() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        analyze_free_space(
            np.ones((30, 45)),
            _frame(),
            _intrinsics(),
            _config(),
        )


def test_replay_pipeline_includes_free_space_context() -> None:
    frame = _frame()
    manifest = ReplayManifest(
        sequence_id="free-space",
        source="synthetic",
        frames=[frame],
    )
    depth = np.full((60, 90), 5.0)
    depth[:, 33:57] = 1.0

    def image_loader(frame_packet: FramePacket, root: Path) -> np.ndarray:
        del root
        return np.zeros(
            (frame_packet.height_px, frame_packet.width_px, 3),
            dtype=np.uint8,
        )

    def depth_loader(frame_packet: FramePacket, root: Path) -> np.ndarray:
        del frame_packet, root
        return depth

    results = tuple(
        process_replay(
            manifest=manifest,
            root=Path("."),
            detector=EmptyDetector(),
            tracker=_EmptyTracker(),
            image_loader=image_loader,
            depth_loader=depth_loader,
            camera_intrinsics=_intrinsics(),
            free_space_config=_config(),
        )
    )

    assert len(results) == 1
    assert results[0].free_space_analysis is not None
    assert results[0].free_space_analysis.context is FreeSpaceContext.CENTER_BLOCKED


def test_replay_free_space_requires_depth_and_intrinsics() -> None:
    manifest = ReplayManifest(
        sequence_id="invalid-free-space",
        source="synthetic",
        frames=[_frame()],
    )

    with pytest.raises(ValueError, match="free_space_config"):
        tuple(
            process_replay(
                manifest=manifest,
                root=Path("."),
                detector=EmptyDetector(),
                tracker=_EmptyTracker(),
                image_loader=lambda frame, root: np.zeros(
                    (frame.height_px, frame.width_px, 3),
                    dtype=np.uint8,
                ),
                free_space_config=_config(),
            )
        )
