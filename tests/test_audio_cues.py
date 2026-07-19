"""Tests for deterministic stereo warning cue synthesis and WAV export."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from sightly_assist.audio_cues import (
    export_warning_audio_cues,
    phrase_for_action,
    render_action_audio,
    render_warning_audio,
    write_warning_wav,
)
from sightly_assist.warning_policy import AlertAction, WarningDecision


def _decision(
    action: AlertAction,
    *,
    timestamp_s: float = 1.0,
    should_emit: bool = True,
) -> WarningDecision:
    return WarningDecision(
        timestamp_s=timestamp_s,
        action=action,
        should_emit=should_emit,
        reason="unit-test decision",
    )


def _channel_rms(samples: np.ndarray, channel: int) -> float:
    return float(np.sqrt(np.mean(np.square(samples[:, channel], dtype=np.float64))))


def test_directional_awareness_cues_pan_toward_the_hazard() -> None:
    left = render_action_audio(AlertAction.AWARENESS_LEFT)
    right = render_action_audio(AlertAction.AWARENESS_RIGHT)

    assert left is not None
    assert right is not None
    assert _channel_rms(left.samples, 0) > _channel_rms(left.samples, 1)
    assert _channel_rms(right.samples, 1) > _channel_rms(right.samples, 0)


def test_stop_cue_is_longer_than_single_awareness_cue() -> None:
    stop = render_action_audio(AlertAction.STOP)
    awareness = render_action_audio(AlertAction.AWARENESS_CENTER)

    assert stop is not None
    assert awareness is not None
    assert stop.duration_s > awareness.duration_s
    assert stop.samples.shape[1] == 2
    assert np.max(np.abs(stop.samples)) <= 1.0
    assert phrase_for_action(AlertAction.STOP) == "Stop"


def test_no_alert_and_suppressed_decisions_produce_no_audio() -> None:
    assert render_action_audio(AlertAction.NO_ALERT) is None
    assert render_warning_audio(_decision(AlertAction.STOP, should_emit=False)) is None


def test_write_warning_wav_produces_stereo_pcm(tmp_path: Path) -> None:
    output = tmp_path / "stop.wav"
    metadata = write_warning_wav(_decision(AlertAction.STOP), output)

    assert metadata is not None
    assert output.is_file()
    assert metadata.action is AlertAction.STOP
    assert metadata.duration_s > 0

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnframes() > 0


def test_export_warning_audio_cues_writes_only_emitted_decisions(tmp_path: Path) -> None:
    decisions = [
        _decision(AlertAction.AWARENESS_LEFT, timestamp_s=0.5),
        _decision(AlertAction.AWARENESS_LEFT, timestamp_s=0.6, should_emit=False),
        _decision(AlertAction.SLOW, timestamp_s=1.0),
        _decision(AlertAction.NO_ALERT, timestamp_s=1.5, should_emit=False),
    ]

    exported = export_warning_audio_cues(decisions, tmp_path)

    assert [item.action for item in exported] == [
        AlertAction.AWARENESS_LEFT,
        AlertAction.SLOW,
    ]
    assert all(Path(item.output_path).is_file() for item in exported)
    manifest = json.loads((tmp_path / "audio_manifest.json").read_text(encoding="utf-8"))
    assert [item["action"] for item in manifest] == ["awareness_left", "slow"]


def test_export_rejects_out_of_order_decisions(tmp_path: Path) -> None:
    decisions = [
        _decision(AlertAction.SLOW, timestamp_s=2.0),
        _decision(AlertAction.STOP, timestamp_s=1.0),
    ]

    with pytest.raises(ValueError, match="ordered by timestamp"):
        export_warning_audio_cues(decisions, tmp_path)
